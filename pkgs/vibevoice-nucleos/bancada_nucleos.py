"""Bancada de los nucleos nativos: paridad, microbanco por capa y decode entero.

Tres ordenes, de mas barata a mas cara:

  python bancada_nucleos.py paridad
      Cada envoltorio contra su referencia torch EXACTA: se emula la misma
      cuantizacion (u7 por tensor, int8 por fila) con ops de torch y la salida
      debe coincidir salvo redondeo fp32. Caza errores de logica (saturacion,
      empaquetado, punto cero) sin necesitar pesos reales.

  python bancada_nucleos.py capas
      Microbanco con las formas REALES del decodificador: cada capa en
      fp32 / int8 dinamico de fbgemm (lo que hay hoy) / nucleo nativo.
      Es la vara de la puerta B (subidas >=2x vs fp32) y de la puerta C
      (FFN >=1,25x vs fbgemm; si fbgemm empata, fbgemm se queda).

  python bancada_nucleos.py decoder [--pesos model.safetensors] [--fotogramas 25]
      Decode en streaming completo (misma topologia y estados que el
      decodificador real, ver decoder_manual.py) en cinco variantes:
      fp32, produccion (dw rapida + fbgemm), subidas (solo convtr
      nativas), etapa0 (subidas + stem + FFN de T=1) y nativo (ambito
      todo). Da ms/fotograma con
      desglose por tramo y la correlacion del audio contra fp32. Sin
      --pesos usa pesos aleatorios: vale para rendimiento y paridad, no
      para juzgar calidad de audio.

Medir SIEMPRE con el entorno de produccion (docs/optimizacion.md):
  OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close

Si no hay .so a mano la bancada lo compila ella misma con g++ (util en el
devShell y en la VM); el servicio en cambio solo usa el .so del store de Nix.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------- utilidades

MS_POR_FRAME = 3200.0 / 24000.0 * 1000.0    # 133,3 ms de audio por fotograma


def asegurar_so():
    """Deja VIBEVOICE_NUCLEOS_SO apuntando a un .so utilizable."""
    if os.environ.get("VIBEVOICE_NUCLEOS_SO", "").strip():
        return
    aqui = os.path.dirname(os.path.abspath(__file__))
    vecino = os.path.join(aqui, "libnucleos_vibevoice.so")
    if os.path.exists(vecino):
        os.environ["VIBEVOICE_NUCLEOS_SO"] = vecino
        return
    gxx = shutil.which("g++") or shutil.which("c++")
    if not gxx:
        raise SystemExit("ni VIBEVOICE_NUCLEOS_SO ni g++ para compilarlo")
    destino = os.path.join(tempfile.mkdtemp(prefix="nucleos-"),
                           "libnucleos_vibevoice.so")
    subprocess.run(
        [gxx, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared", "-fPIC",
         "-Wall", os.path.join(aqui, "nucleos.cpp"), "-o", destino],
        check=True)
    print(f"[bancada] compilado {destino}")
    os.environ["VIBEVOICE_NUCLEOS_SO"] = destino


def cronometrar(fn, repeticiones=30, calentamiento=5):
    for _ in range(calentamiento):
        fn()
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(tiempos))


def motor_fbgemm():
    for m in ("x86", "fbgemm"):
        if m in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = m
            return m
    return None


# --------------------------------------------------- paridad contra torch

def _emular_u7(x):
    """La misma cuantizacion de activaciones que hace el nucleo, en torch fp32."""
    mn = float(min(x.min().item(), 0.0))
    mx = float(max(x.max().item(), 0.0))
    esc = torch.tensor(mx - mn, dtype=torch.float32) / 127.0
    if esc.item() <= 0:
        return torch.zeros_like(x), esc + 1.0, 0
    zp = int(torch.round(torch.tensor(-mn, dtype=torch.float32) / esc)
             .clamp(0, 127).item())
    inv = 1.0 / esc
    xq = (torch.round(x * inv) + zp).clamp(0, 127)
    return (xq - zp) * esc, esc, zp


def _dequant_filas(w):
    esc = w.abs().amax(dim=1) / 127.0
    esc = torch.where(esc > 0, esc, torch.ones_like(esc))
    return torch.round(w / esc[:, None]).clamp(-127, 127) * esc[:, None]


def paridad():
    from nucleos_torch import (ConvNativa, ConvTrNativa, LinealNativa,
                               cargar_nucleos)
    lib = cargar_nucleos()
    assert lib is not None, "no se pudo cargar el .so"
    torch.manual_seed(7)
    fallos = 0

    # Lineales: el nucleo debe dar lo MISMO que torch fp32 operando sobre los
    # tensores ya cuantizados-y-decuantizados (solo queda redondeo fp32).
    for t, k, n in [(1, 2048, 8192), (1, 8192, 2048), (8, 1024, 4096),
                    (8, 4096, 1024), (40, 512, 2048), (40, 2048, 512),
                    (3, 448, 2048), (11, 512, 64)]:
        lineal = nn.Linear(k, n)
        x = torch.randn(t, k) * 0.7
        nativa = LinealNativa(lineal, lib)
        y = nativa(x)
        xdq, _, _ = _emular_u7(x)
        ref = F.linear(xdq, _dequant_filas(lineal.weight.detach()),
                       lineal.bias.detach())
        d = (y - ref).abs().max().item()
        err = (y - lineal(x)).abs().max().item() / lineal(x).abs().max().item()
        ok = d < 1e-3 * max(1.0, ref.abs().max().item())
        fallos += not ok
        print(f"linear  T={t:3d} K={k:5d} N={n:5d}  exacta={d:.2e}  "
              f"vs_fp32={err:.3%}  {'ok' if ok else 'MAL'}")

    # Subidas: contra F.conv_transpose1d con pesos decuantizados; ademas la
    # forma de salida debe ser identica a la del modulo original.
    for t, ent, sal, k, s in [(16, 2048, 1024, 16, 8), (17, 1024, 512, 10, 5),
                              (49, 512, 256, 10, 5), (1, 512, 256, 10, 5)]:
        conv = nn.ConvTranspose1d(ent, sal, k, stride=s)
        x = torch.randn(1, ent, t) * 0.5
        nativa = ConvTrNativa(conv, lib)
        y = nativa(x)
        esc = conv.weight.detach().abs().amax(dim=(0, 2)) / 127.0
        esc = torch.where(esc > 0, esc, torch.ones_like(esc))
        wdq = (torch.round(conv.weight.detach() / esc[None, :, None])
               .clamp(-127, 127) * esc[None, :, None])
        xdq, _, _ = _emular_u7(x)
        ref = F.conv_transpose1d(xdq, wdq, conv.bias.detach(), stride=s)
        fp = F.conv_transpose1d(x, conv.weight.detach(), conv.bias.detach(),
                                stride=s)
        assert y.shape == fp.shape, (y.shape, fp.shape)
        d = (y - ref).abs().max().item()
        err = (y - fp).abs().max().item() / fp.abs().max().item()
        ok = d < 1e-3 * max(1.0, ref.abs().max().item())
        fallos += not ok
        print(f"convtr  T={t:3d} {ent:4d}->{sal:4d} k={k:2d} s={s}  "
              f"exacta={d:.2e}  vs_fp32={err:.3%}  {'ok' if ok else 'MAL'}")

    # Stem
    conv = nn.Conv1d(64, 2048, 7)
    x = torch.randn(1, 64, 7 + 4) * 0.5
    nativa = ConvNativa(conv, lib)
    y = nativa(x)
    xdq, _, _ = _emular_u7(
        x.unfold(2, 7, 1)[0].permute(1, 0, 2).reshape(-1, 448).contiguous())
    ref = F.linear(xdq, _dequant_filas(conv.weight.detach().reshape(2048, 448)),
                   conv.bias.detach()).t().unsqueeze(0)
    fp = F.conv1d(x, conv.weight.detach(), conv.bias.detach())
    assert y.shape == fp.shape, (y.shape, fp.shape)
    d = (y - ref).abs().max().item()
    err = (y - fp).abs().max().item() / fp.abs().max().item()
    ok = d < 1e-3 * max(1.0, ref.abs().max().item())
    fallos += not ok
    print(f"stem    64->2048 k=7             exacta={d:.2e}  "
          f"vs_fp32={err:.3%}  {'ok' if ok else 'MAL'}")
    if fallos:
        raise SystemExit(f"{fallos} pruebas MAL")
    print("paridad: todo ok")


# ------------------------------------------------------ microbanco por capa

def capas():
    from nucleos_torch import ConvTrNativa, LinealNativa, cargar_nucleos
    lib = cargar_nucleos()
    assert lib is not None, "no se pudo cargar el .so"
    torch.manual_seed(7)
    motor = motor_fbgemm()
    print(f"motor fbgemm: {motor}, hilos torch: {torch.get_num_threads()}")

    print("\n--- FFN (puerta C: nativo >= 1,25x fbgemm o se queda fbgemm) ---")
    for t, k, n in [(1, 2048, 8192), (1, 8192, 2048), (8, 1024, 4096),
                    (8, 4096, 1024), (40, 512, 2048), (40, 2048, 512)]:
        lineal = nn.Linear(k, n).eval()
        x = torch.randn(t, k)
        din = torch.ao.quantization.quantize_dynamic(
            nn.Sequential(nn.Linear(k, n)).eval(), {nn.Linear},
            dtype=torch.qint8)
        nativa = LinealNativa(lineal, lib)
        with torch.no_grad():
            ms_fp = cronometrar(lambda: lineal(x))
            ms_din = cronometrar(lambda: din(x))
            ms_nat = cronometrar(lambda: nativa(x))
        print(f"T={t:3d} K={k:5d} N={n:5d}  fp32 {ms_fp:7.3f}  "
              f"fbgemm {ms_din:7.3f}  nativo {ms_nat:7.3f} ms  "
              f"(x{ms_din / ms_nat:4.2f} vs fbgemm)")

    print("\n--- subidas (puerta B: nativo >= 2x fp32) ---")
    for t, ent, sal, k, s in [(16, 2048, 1024, 16, 8), (17, 1024, 512, 10, 5),
                              (49, 512, 256, 10, 5)]:
        conv = nn.ConvTranspose1d(ent, sal, k, stride=s).eval()
        x = torch.randn(1, ent, t)
        nativa = ConvTrNativa(conv, lib)
        with torch.no_grad():
            ms_fp = cronometrar(lambda: conv(x))
            ms_nat = cronometrar(lambda: nativa(x))
        print(f"T={t:3d} {ent:4d}->{sal:4d} k={k:2d} s={s}  "
              f"fp32 {ms_fp:7.3f}  nativo {ms_nat:7.3f} ms  "
              f"(x{ms_fp / ms_nat:4.2f})")


# ------------------------------------------- decodificador entero (streaming)

# Copia funcional de ConvDepthwiseRapida (voz_stream.py): la depthwise como
# suma de K desplazamientos. Aqui no se importa voz_stream porque arrastra
# fastapi/uvicorn; la logica son estas 12 lineas y la paridad esta medida
# alli (diff 4,77e-07).
class DwRapida(nn.Module):
    def __init__(self, conv):
        super().__init__()
        c, k = conv.out_channels, conv.kernel_size[0]
        self.k = k
        w = conv.weight.detach().reshape(c, k).t().contiguous().view(k, 1, c, 1)
        self.register_buffer("w", w)
        self.register_buffer("b", conv.bias.detach().reshape(1, c, 1).clone())

    def forward(self, x):
        largo = x.shape[2] - self.k + 1
        salida = x[:, :, :largo] * self.w[0]
        for j in range(1, self.k):
            salida = salida + x[:, :, j:j + largo] * self.w[j]
        return salida + self.b


# Misma topologia y estados que decoder_manual.py (que es la referencia fiel
# del decodificador real), pero con nn.Linear / nn.Conv1d / nn.ConvTranspose1d
# de verdad, para que quantize_dynamic y _acelerar_arbol le apliquen igual
# que al modelo de produccion.
RATIOS = [8, 5, 5, 4, 2, 2]
PROFUNDIDADES = [8, 3, 3, 3, 3, 3, 3]
DIMS = [2048, 1024, 512, 256, 128, 64, 32]
EPS = 1e-5


def _norma(x, peso):
    h = x.transpose(1, 2)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
    return (h * peso).transpose(1, 2)


class _Causal(nn.Module):
    def __init__(self, ent, sal, k=7):
        super().__init__()
        self.conv = nn.Conv1d(ent, sal, k)
        self.ctx = k - 1

    def forward(self, x, est):
        full = torch.cat([est, x], dim=2)
        return self.conv(full), full[:, :, -self.ctx:]


class _Bloque(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(d))
        self.dw = nn.Conv1d(d, d, 7, groups=d)
        self.gamma = nn.Parameter(torch.full((d,), 0.1))
        self.ffn_norm_w = nn.Parameter(torch.ones(d))
        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)
        self.ffn_gamma_ = nn.Parameter(torch.full((d,), 0.1))

    def forward(self, x, est):
        r = x
        h = _norma(x, self.norm_w)
        full = torch.cat([est, h], dim=2)
        nuevo = full[:, :, -6:]
        h = self.dw(full)
        x = r + h * self.gamma[None, :, None]
        r = x
        h = _norma(x, self.ffn_norm_w).transpose(1, 2)
        h = self.l2(F.gelu(self.l1(h))).transpose(1, 2)
        return r + h * self.ffn_gamma_[None, :, None], nuevo


class _Subida(nn.Module):
    def __init__(self, ent, sal, ratio):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(ent, sal, ratio * 2, stride=ratio)
        self.k, self.s = ratio * 2, ratio

    def forward(self, x, est):
        full = torch.cat([est, x], dim=2)
        nuevo = full[:, :, -(self.k - 1):]
        y = self.convtr(full)
        y = y[:, :, : -(self.k - self.s)]
        return y[:, :, -(x.shape[2] * self.s):], nuevo


class DecoderBancada(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = _Causal(64, 2048)
        self.subidas = nn.ModuleList(
            _Subida(DIMS[i], DIMS[i + 1], RATIOS[i]) for i in range(6))
        self.etapas = nn.ModuleList(
            nn.ModuleList(_Bloque(DIMS[i]) for _ in range(PROFUNDIDADES[i]))
            for i in range(7))
        self.head = _Causal(32, 1)

    def estados_cero(self):
        formas = [(64, 6)]
        for i in range(7):
            if i > 0:
                formas.append((DIMS[i - 1], RATIOS[i - 1] * 2 - 1))
            formas += [(DIMS[i], 6)] * PROFUNDIDADES[i]
        formas.append((32, 6))
        return [torch.zeros(1, c, t) for c, t in formas]

    def forward(self, lat, estados, cronos=None):
        nuevos = []
        j = 0

        def paso(mod, x, etiqueta):
            nonlocal j
            t0 = time.perf_counter() if cronos is not None else 0.0
            y, ne = mod(x, estados[j])
            if cronos is not None:
                cronos[etiqueta] = (cronos.get(etiqueta, 0.0)
                                    + (time.perf_counter() - t0) * 1000.0)
            nuevos.append(ne)
            j += 1
            return y

        x = paso(self.stem, lat, "stem")
        for i in range(7):
            if i > 0:
                x = paso(self.subidas[i - 1], x, f"subida{i - 1}")
            for blq in self.etapas[i]:
                x = paso(blq, x, f"etapa{i}")
        audio = paso(self.head, x, "head")
        return audio, nuevos


def _cargar_pesos(m, ruta):
    from safetensors import safe_open
    P = "model.acoustic_tokenizer.decoder."
    with safe_open(ruta, framework="pt") as f:
        def t(clave):
            return f.get_tensor(P + clave).float()
        m.stem.conv.weight.data = t("upsample_layers.0.0.conv.conv.weight")
        m.stem.conv.bias.data = t("upsample_layers.0.0.conv.conv.bias")
        m.head.conv.weight.data = t("head.conv.conv.weight")
        m.head.conv.bias.data = t("head.conv.conv.bias")
        for i in range(6):
            m.subidas[i].convtr.weight.data = \
                t("upsample_layers.%d.0.convtr.convtr.weight" % (i + 1))
            m.subidas[i].convtr.bias.data = \
                t("upsample_layers.%d.0.convtr.convtr.bias" % (i + 1))
        for i in range(7):
            for jj, blq in enumerate(m.etapas[i]):
                b = "stages.%d.%d." % (i, jj)
                blq.norm_w.data = t(b + "norm.weight")
                blq.dw.weight.data = t(b + "mixer.conv.conv.conv.weight")
                blq.dw.bias.data = t(b + "mixer.conv.conv.conv.bias")
                blq.gamma.data = t(b + "gamma")
                blq.ffn_norm_w.data = t(b + "ffn_norm.weight")
                blq.l1.weight.data = t(b + "ffn.linear1.weight")
                blq.l1.bias.data = t(b + "ffn.linear1.bias")
                blq.l2.weight.data = t(b + "ffn.linear2.weight")
                blq.l2.bias.data = t(b + "ffn.linear2.bias")
                blq.ffn_gamma_.data = t(b + "ffn_gamma")


def _construir(variante, pesos, lib):
    from nucleos_torch import _acelerar_arbol
    torch.manual_seed(1)
    m = DecoderBancada()
    if pesos:
        _cargar_pesos(m, pesos)
    else:
        # aleatorio pero acotado: solo vale para rendimiento y paridad
        for parametro in m.parameters():
            if parametro.dim() > 1:
                parametro.data = torch.randn_like(parametro) * 0.05
    m.eval()
    # las depthwise van SIEMPRE en dw rapida: es el estado de produccion
    for padre in m.modules():
        for nombre, hijo in list(padre.named_children()):
            if (isinstance(hijo, nn.Conv1d) and hijo.groups > 1
                    and hijo.groups == hijo.in_channels == hijo.out_channels):
                setattr(padre, nombre, DwRapida(hijo))
    if variante in ("nativo", "etapa0", "subidas"):
        ambito = "todo" if variante == "nativo" else variante
        n = _acelerar_arbol(m, lib, ambito)
        print(f"  [{variante}] capas nativas: {n}")
    if variante in ("produccion", "nativo", "etapa0", "subidas"):
        motor_fbgemm()
        torch.ao.quantization.quantize_dynamic(
            m, {nn.Linear}, dtype=torch.qint8, inplace=True)
    return m


def decoder(pesos, fotogramas):
    from nucleos_torch import cargar_nucleos
    lib = cargar_nucleos()
    assert lib is not None, "no se pudo cargar el .so"
    if not pesos:
        print("[bancada] SIN pesos reales: rendimiento y paridad valen, "
              "la calidad de audio no se puede juzgar aqui")
    torch.manual_seed(3)
    latentes = [torch.randn(1, 64, 1) * 0.5 for _ in range(fotogramas + 3)]
    resultados = {}
    for variante in ("fp32", "produccion", "subidas", "etapa0", "nativo"):
        m = _construir(variante, pesos, lib)
        estados = m.estados_cero()
        audio, tiempos, cronos = [], [], {}
        with torch.no_grad():
            for idx, lat in enumerate(latentes):
                t0 = time.perf_counter()
                y, estados = m(lat, estados, cronos if idx >= 3 else None)
                dt = (time.perf_counter() - t0) * 1000.0
                if idx >= 3:                     # 3 de calentamiento
                    tiempos.append(dt)
                    audio.append(y.reshape(-1).numpy().copy())
        ms = float(np.median(tiempos))
        resultados[variante] = (ms, np.concatenate(audio), cronos)
        del m
        print(f"{variante:10s} {ms:7.2f} ms/fotograma  "
              f"(RTF decoder {ms / MS_POR_FRAME:.3f})")
        por_tramo = sorted(cronos.items(), key=lambda kv: -kv[1])[:6]
        detalle = "  ".join(f"{k} {v / len(tiempos):6.2f}" for k, v in por_tramo)
        print(f"           tramos ms/fotograma: {detalle}")
    ref = resultados["fp32"][1]
    print()
    for variante in ("produccion", "subidas", "etapa0", "nativo"):
        a = resultados[variante][1]
        corr = float(np.corrcoef(ref, a)[0, 1])
        dmax = float(np.abs(ref - a).max())
        print(f"{variante:10s} correlacion vs fp32 {corr:.8f}  "
              f"diff max {dmax:.2e}  "
              f"({resultados['fp32'][0] / resultados[variante][0]:.2f}x fp32, "
              f"{resultados['produccion'][0] / resultados[variante][0]:.2f}x produccion)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="orden", required=True)
    sub.add_parser("paridad")
    sub.add_parser("capas")
    p = sub.add_parser("decoder")
    p.add_argument("--pesos", default=os.environ.get("VIBEVOICE_MODELO", "")
                   and os.path.join(os.environ["VIBEVOICE_MODELO"],
                                    "model.safetensors"))
    p.add_argument("--fotogramas", type=int, default=25)
    argumentos = parser.parse_args()
    asegurar_so()
    if argumentos.orden == "paridad":
        paridad()
    elif argumentos.orden == "capas":
        capas()
    else:
        pesos = argumentos.pesos if argumentos.pesos and \
            os.path.exists(argumentos.pesos) else ""
        decoder(pesos, argumentos.fotogramas)


if __name__ == "__main__":
    main()
