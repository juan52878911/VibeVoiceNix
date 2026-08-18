"""Envoltorios torch de los nucleos nativos int8 (nucleos.cpp, via ctypes).

Sustituyen modulos HOJA del decodificador acustico -- las nn.Linear de las
FFN grandes, las nn.ConvTranspose1d de las subidas y la nn.Conv1d del stem --
por versiones con pesos int8 servidas por libnucleos_vibevoice.so. Toda la
logica de streaming (la cache de 34 estados, la concatenacion de contexto, el
cebado con silencio) vive en los modulos que ENVUELVEN a estos y no se toca:
es la misma estrategia que ConvDepthwiseRapida en voz_stream.py, que ya
demostro que el cambio a nivel de hoja preserva la semantica.

Por que estos modulos y no otros: la VM va limitada por ancho de banda (80,7%
del bus, medido), asi que paga reducir bytes de peso leidos por fotograma.
Las subidas grandes son 162 MB fp32 por fotograma que quantize_dynamic no
cuantiza (solo toca nn.Linear); en int8 quedan en ~41 MB. Las FFN ya iban en
int8 fbgemm, pero el nucleo propio se ahorra el sobrecoste por llamada con
T pequeno (1, 8 o 40 fotogramas segun etapa). Las etapas de canal fino
(<512), las depthwise y las normas se quedan como estan: alli el limite son
las activaciones, no los pesos, e int8 no reduce el recurso escaso.

El acuerdo con el .so (mismos numeros que en nucleos.cpp):
  - pesos int8 simetricos por canal de salida, escala fp32 por fila;
  - activaciones u7 dinamicas por tensor (sin VNNI la ruta es VPMADDUBSW,
    que satura en i16; con u7 la saturacion es imposible por construccion);
  - K (dimension de reduccion) multiplo de 32: aqui se comprueba y si una
    capa no lo cumple sencillamente no se sustituye.

Interruptores:
  VIBEVOICE_NUCLEOS_SO      ruta del .so (la pone nix/modules/voz-stream.nix;
                            si falta se busca junto a este fichero)
  VIBEVOICE_SIN_NUCLEOS=1   apagado de emergencia sin rebuild
  VIBEVOICE_NUCLEOS_AMBITO  "todo" (defecto), "etapa0" (subidas + stem + solo
                            las FFN de la etapa 0, que van con T=1 y son el 88%
                            de los bytes; el resto sigue en fbgemm) o "subidas"
                            (solo las conv transpuestas). Los tres ambitos son
                            el A/B contra fbgemm: la bancada decide cual gana
                            EN LA MAQUINA DE DESTINO, no en la de desarrollo

En cualquier maquina sin el .so, sin x86_64 o sin AVX2, cargar_nucleos()
devuelve None y el modelo queda exactamente como estaba: mac (MPS), CUDA y
contenedores sin la libreria siguen funcionando igual.
"""
import ctypes
import os
import platform

import torch
import torch.nn as nn

VERSION_ESPERADA = 1

_lib = None
_intentado = False


def cargar_nucleos():
    """El CDLL de los nucleos, o None si aqui no procede usarlos."""
    global _lib, _intentado
    if _intentado:
        return _lib
    _intentado = True
    if os.environ.get("VIBEVOICE_SIN_NUCLEOS", "").strip() not in ("", "0"):
        return None
    if platform.machine() not in ("x86_64", "AMD64"):
        return None
    try:
        with open("/proc/cpuinfo") as f:
            if "avx2" not in f.read():
                return None
    except OSError:
        return None  # sin /proc no hay forma barata de saberlo: mejor no
    ruta = os.environ.get("VIBEVOICE_NUCLEOS_SO", "").strip()
    if not ruta:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "libnucleos_vibevoice.so")
    if not os.path.exists(ruta):
        return None
    try:
        lib = ctypes.CDLL(ruta)
    except OSError as e:
        print(f"[aviso] nucleos nativos ilegibles en {ruta}: {e}", flush=True)
        return None
    lib.vv_version.restype = ctypes.c_int
    lib.vv_version.argtypes = []
    version = lib.vv_version()
    if version != VERSION_ESPERADA:
        print(f"[aviso] nucleos nativos version {version}, esperada "
              f"{VERSION_ESPERADA}: se ignoran", flush=True)
        return None
    p, i64 = ctypes.c_void_p, ctypes.c_int64
    lib.vv_linear_din.restype = ctypes.c_int
    lib.vv_linear_din.argtypes = [p, p, p, p, p, p, i64, i64, i64]
    lib.vv_convtr_din.restype = ctypes.c_int
    lib.vv_convtr_din.argtypes = [p, p, p, p, p, p, i64, i64, i64, i64, i64]
    _lib = lib
    return lib


def _cuantizar_filas(w):
    """fp32 [N,K] -> (int8 [N,K], escalas fp32 [N], sumas i32 [N]).

    Simetrica por fila: escala = max|fila| / 127. La suma por fila se
    precalcula aqui para que el nucleo no relea los pesos en cada llamada
    solo para corregir el punto cero de las activaciones.
    """
    escalas = w.abs().amax(dim=1) / 127.0
    escalas = torch.where(escalas > 0, escalas, torch.ones_like(escalas))
    wq = torch.round(w / escalas[:, None]).clamp(-127, 127).to(torch.int8)
    sumas = wq.to(torch.int32).sum(dim=1, dtype=torch.int32)
    return wq.contiguous(), escalas.contiguous(), sumas.contiguous()


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


class LinealNativa(nn.Module):
    """nn.Linear con pesos int8 por fila, servida por vv_linear_din."""

    def __init__(self, lineal: nn.Linear, lib):
        super().__init__()
        self.ent, self.sal = lineal.in_features, lineal.out_features
        self._lib = lib
        wq, escalas, sumas = _cuantizar_filas(lineal.weight.detach().float())
        self.register_buffer("wq", wq)
        self.register_buffer("escalas", escalas)
        self.register_buffer("sumas", sumas)
        if lineal.bias is not None:
            self.register_buffer("sesgo", lineal.bias.detach().float().contiguous())
        else:
            self.sesgo = None

    def forward(self, x):
        forma = x.shape
        x2 = x.reshape(-1, self.ent)
        if x2.dtype != torch.float32 or not x2.is_contiguous():
            x2 = x2.float().contiguous()
        t = x2.shape[0]
        y = torch.empty((t, self.sal), dtype=torch.float32)
        r = self._lib.vv_linear_din(
            _ptr(x2), _ptr(self.wq), _ptr(self.escalas), _ptr(self.sumas),
            _ptr(self.sesgo) if self.sesgo is not None else None, _ptr(y),
            t, self.ent, self.sal)
        if r != 0:
            raise RuntimeError(f"vv_linear_din devolvio {r}")
        return y.reshape(*forma[:-1], self.sal)


class ConvTrNativa(nn.Module):
    """nn.ConvTranspose1d con pesos int8 por canal de salida.

    Devuelve la salida COMPLETA, la misma que F.conv_transpose1d: el recorte
    causal lo hace el modulo que envuelve (SConvTranspose1d), igual que antes.
    El W[Cin,Cout,k] de torch se reordena a [Cout*k, Cin] para que la GEMM
    reduzca sobre Cin contiguo; la escala por canal se repite k veces para
    que el nucleo la vea por columna, sin caso especial.
    """

    def __init__(self, conv: nn.ConvTranspose1d, lib):
        super().__init__()
        self.ent, self.sal = conv.in_channels, conv.out_channels
        self.k, self.s = conv.kernel_size[0], conv.stride[0]
        self._lib = lib
        w = conv.weight.detach().float()                      # [Cin, Cout, k]
        escalas = w.abs().amax(dim=(0, 2)) / 127.0            # por Cout
        escalas = torch.where(escalas > 0, escalas, torch.ones_like(escalas))
        wq = torch.round(w / escalas[None, :, None]).clamp(-127, 127).to(torch.int8)
        wr = wq.permute(1, 2, 0).reshape(self.sal * self.k, self.ent).contiguous()
        self.register_buffer("wq", wr)
        self.register_buffer("escalas",
                             escalas.repeat_interleave(self.k).contiguous())
        self.register_buffer("sumas",
                             wr.to(torch.int32).sum(dim=1, dtype=torch.int32)
                             .contiguous())
        if conv.bias is not None:
            self.register_buffer("sesgo", conv.bias.detach().float().contiguous())
        else:
            self.sesgo = None

    def forward(self, x):
        if x.dim() != 3 or x.shape[0] != 1:
            raise RuntimeError(f"ConvTrNativa espera [1,C,T], llego {tuple(x.shape)}")
        t = x.shape[2]
        x2 = x[0]
        if x2.dtype != torch.float32 or not x2.is_contiguous():
            x2 = x2.float().contiguous()
        largo = (t - 1) * self.s + self.k
        y = torch.empty((1, self.sal, largo), dtype=torch.float32)
        r = self._lib.vv_convtr_din(
            _ptr(x2), _ptr(self.wq), _ptr(self.escalas), _ptr(self.sumas),
            _ptr(self.sesgo) if self.sesgo is not None else None, _ptr(y),
            t, self.ent, self.sal, self.k, self.s)
        if r != 0:
            raise RuntimeError(f"vv_convtr_din devolvio {r}")
        return y


class ConvNativa(nn.Module):
    """nn.Conv1d densa (stem 64->2048 k7) como unfold + vv_linear_din.

    Una conv densa k7 s1 es una GEMM sobre la entrada desplegada: cada
    columna de salida ve K = Cin*k = 448 entradas. El despliegue son 3 KB
    por fotograma; los pesos (0,9M) se leen en int8.
    """

    def __init__(self, conv: nn.Conv1d, lib):
        super().__init__()
        self.ent, self.sal, self.k = (conv.in_channels, conv.out_channels,
                                      conv.kernel_size[0])
        self._lib = lib
        w = conv.weight.detach().float().reshape(self.sal, self.ent * self.k)
        wq, escalas, sumas = _cuantizar_filas(w)
        self.register_buffer("wq", wq)
        self.register_buffer("escalas", escalas)
        self.register_buffer("sumas", sumas)
        if conv.bias is not None:
            self.register_buffer("sesgo", conv.bias.detach().float().contiguous())
        else:
            self.sesgo = None

    def forward(self, x):
        if x.dim() != 3 or x.shape[0] != 1:
            raise RuntimeError(f"ConvNativa espera [1,C,T], llego {tuple(x.shape)}")
        t = x.shape[2] - self.k + 1
        # [1,C,L] -> [T, C*k]: fila t = la ventana [t, t+k) de todos los canales
        xu = (x.unfold(2, self.k, 1)[0]        # [C, T, k]
              .permute(1, 0, 2)                # [T, C, k]
              .reshape(t, self.ent * self.k))
        if xu.dtype != torch.float32 or not xu.is_contiguous():
            xu = xu.float().contiguous()
        y = torch.empty((t, self.sal), dtype=torch.float32)
        r = self._lib.vv_linear_din(
            _ptr(xu), _ptr(self.wq), _ptr(self.escalas), _ptr(self.sumas),
            _ptr(self.sesgo) if self.sesgo is not None else None, _ptr(y),
            t, self.ent * self.k, self.sal)
        if r != 0:
            raise RuntimeError(f"vv_linear_din devolvio {r}")
        return y.t().unsqueeze(0).contiguous()


def _acelerar_arbol(dec, lib, ambito):
    """Sustituye las hojas elegibles de un arbol de modulos. Devuelve contadores."""
    subidas = lineales = stems = 0
    for padre in dec.modules():
        for nombre, hijo in list(padre.named_children()):
            if (isinstance(hijo, nn.ConvTranspose1d)
                    and hijo.stride[0] > 1
                    and hijo.dilation[0] == 1
                    and hijo.groups == 1
                    and hijo.padding[0] == 0
                    and hijo.output_padding[0] == 0
                    and hijo.in_channels % 32 == 0
                    # solo las 3 subidas gordas (2048/1024/512 de entrada):
                    # en las de canal fino mandan las activaciones, no los pesos
                    and hijo.in_channels >= 512):
                setattr(padre, nombre, ConvTrNativa(hijo, lib))
                subidas += 1
            elif (ambito in ("todo", "etapa0")
                    and isinstance(hijo, nn.Linear)
                    # "etapa0": solo las FFN de canal 2048, que en streaming
                    # van con T=1 (donde el nucleo propio gana a fbgemm hasta
                    # en la maquina de desarrollo); con "todo" tambien las de
                    # 1024 y 512 (T=8 y 40), a decidir midiendo en el destino
                    and min(hijo.in_features, hijo.out_features)
                        >= (2048 if ambito == "etapa0" else 512)
                    and hijo.in_features % 32 == 0):
                setattr(padre, nombre, LinealNativa(hijo, lib))
                lineales += 1
            elif (ambito in ("todo", "etapa0")
                    and isinstance(hijo, nn.Conv1d)
                    and hijo.groups == 1
                    and hijo.stride[0] == 1
                    and hijo.dilation[0] == 1
                    and hijo.padding[0] == 0
                    and hijo.out_channels >= 512
                    and (hijo.in_channels * hijo.kernel_size[0]) % 32 == 0):
                setattr(padre, nombre, ConvNativa(hijo, lib))
                stems += 1
    return subidas, lineales, stems


def acelerar_decoder_nativo(modelo) -> int:
    """Pasa el decoder acustico a nucleos nativos int8. Devuelve cuantas capas.

    Llamar ANTES de quantize_dynamic: los modulos sustituidos ya no son
    nn.Linear y quantize_dynamic no los re-toca; los que se dejan en paz
    (etapas de canal fino) siguen siendo nn.Linear y se cuantizan como
    siempre. Con ambito "subidas" las FFN se quedan todas en fbgemm: es el
    A/B para decidir si el nucleo propio le gana o no.
    """
    lib = cargar_nucleos()
    if lib is None:
        return 0
    ambito = os.environ.get("VIBEVOICE_NUCLEOS_AMBITO", "").strip() or "todo"
    if ambito not in ("todo", "etapa0", "subidas"):
        print(f"[aviso] VIBEVOICE_NUCLEOS_AMBITO={ambito!r} no existe; "
              f"se usa 'todo'", flush=True)
        ambito = "todo"
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    dec = getattr(tok, "decoder", None)
    if dec is None:
        return 0
    subidas, lineales, stems = _acelerar_arbol(dec, lib, ambito)
    total = subidas + lineales + stems
    if total:
        print(f"[arranque] decoder acustico en nucleos nativos int8: "
              f"{subidas} subidas, {lineales} FFN, {stems} stem "
              f"(ambito {ambito})", flush=True)
    return total
