"""El nucleo int8 sobre habla real: la misma frase por los dos caminos.

QUE ES Y QUE NO ES
Tampoco esto es VibeVoice -- no carga el checkpoint. Coge habla ya
sintetizada (o cualquier WAV) y la hace pasar por el MISMO nucleo
(vv_convtr_din) con la MISMA geometria que la subida grande del
decodificador: k=16, stride 8. Sirve para oir si la cuantizacion ensucia la
voz, que es lo unico que este cambio puede estropear.

POR QUE IDA Y VUELTA, Y NO SOLO SUBIDA
El banco hermano (prueba_audio.py) baja la senal a 3 kHz y la sube: eso mide
la capa haciendo su trabajo real, pero deja el audio apagado de agudos en
AMBOS caminos, y con voz eso se confunde con dano de la cuantizacion. Aqui la
frase sube x8 por el nucleo (24 -> 192 kHz) y vuelve a bajar con el mismo
filtro por los dos caminos. En fp32 el viaje es practicamente transparente,
asi que lo unico que separa los dos ficheros es el int8. Nada de agudos
perdidos que achacarle por error.

El troceado es el de produccion: una llamada por fotograma de 133 ms, que es
como se cuantizan las activaciones de verdad (ver optimizacion.md, hito 5).

SALIDA
  frase_entrada.wav     lo que entro
  frase_fp32.wav        el camino de hoy
  frase_int8.wav        el camino con el nucleo
  frase_diferencia.wav  la resta, amplificada, para oir QUE anade el int8

Uso:
    python prueba_frase.py --texto "Hola, esto es una prueba."   # pide espeak-ng
    python prueba_frase.py --wav locucion.wav                    # cualquier WAV
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from prueba_audio import K, RATIO, RITMO, bajar_ritmo, capa_subida, escribir_wav

CANALES_ENT = 128        # profundidad de reduccion; menos que una subida real
CANALES_SAL = 8          # solo para que el ida y vuelta quepa en memoria
FOTOGRAMA = 3200         # muestras de entrada por llamada = 133 ms


# ------------------------------------------------------------------ entrada

def leer_wav(ruta):
    with wave.open(ruta, "rb") as f:
        if f.getsampwidth() != 2:
            raise SystemExit(f"{ruta}: solo WAV de 16 bits")
        canales, ritmo = f.getnchannels(), f.getframerate()
        crudo = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    x = crudo.astype(np.float32) / 32768.0
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    return x, ritmo


def remuestrear(x, de, a):
    """Sinc enventanado a posiciones arbitrarias. de == a no toca nada."""
    if de == a:
        return x.astype(np.float32)
    taps = 32
    razon = a / de
    n_sal = int(len(x) * razon)
    pos = np.arange(n_sal) / razon
    base = np.floor(pos).astype(np.int64)
    desfase = pos - base
    # ventana de taps centrada en cada posicion de origen
    idx = base[:, None] + np.arange(-taps // 2 + 1, taps // 2 + 1)[None, :]
    dist = desfase[:, None] - np.arange(-taps // 2 + 1, taps // 2 + 1)[None, :]
    corte = min(1.0, razon)          # anti-alias si se baja el ritmo
    h = np.sinc(dist * corte) * corte * np.hamming(taps)[None, :]
    h /= h.sum(axis=1, keepdims=True)
    xp = np.pad(x, taps, mode="constant")
    return (xp[idx + taps] * h).sum(axis=1).astype(np.float32)


def hablar(texto, voz, velocidad):
    """espeak-ng a WAV. Voz robotica, pero fonemas y prosodia de verdad."""
    if not shutil.which("espeak-ng"):
        raise SystemExit(
            "no hay espeak-ng; usa --wav con un fichero, o instalalo "
            "(nix-shell -p espeak-ng / apt install espeak-ng)")
    ruta = os.path.join(tempfile.mkdtemp(prefix="frase-"), "frase.wav")
    subprocess.run(["espeak-ng", "-v", voz, "-s", str(velocidad),
                    "-w", ruta, texto], check=True)
    return leer_wav(ruta)


# ------------------------------------------------------------------ camino

def pasar(x, conv, nativa):
    """La senal por el nucleo (o por fp32 si nativa es None), y de vuelta.

    Trocea en fotogramas con los k-1 ultimos muestreos de entrada como
    estado, igual que SConvTranspose1d en streaming (decoder_manual.py).
    """
    ent = torch.from_numpy(x)[None, None, :].repeat(1, CANALES_ENT, 1).contiguous()
    trozos, estado = [], torch.zeros(1, CANALES_ENT, K - 1)
    with torch.no_grad():
        for ini in range(0, ent.shape[2], FOTOGRAMA):
            trozo = ent[:, :, ini:ini + FOTOGRAMA]
            lleno = torch.cat([estado, trozo], dim=2)
            estado = lleno[:, :, -(K - 1):]
            if nativa is None:
                y = F.conv_transpose1d(lleno, conv.weight, conv.bias, stride=RATIO)
            else:
                y = nativa(lleno)
            y = y[:, :, : -(K - RATIO)]
            trozos.append(y[:, :, -(trozo.shape[2] * RATIO):])
    alto = torch.cat(trozos, dim=2)[0, 0].numpy()      # 192 kHz
    return bajar_ritmo(alto, RATIO)                    # de vuelta a 24 kHz


def main():
    from nucleos_torch import ConvTrNativa, cargar_nucleos
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texto", default="Hola, esto es una prueba de voz "
                                           "para los núcleos int8.")
    parser.add_argument("--wav", default="")
    parser.add_argument("--voz", default="es")
    parser.add_argument("--velocidad", type=int, default=150)
    parser.add_argument("--salida", default=".")
    argumentos = parser.parse_args()

    lib = cargar_nucleos()
    if lib is None:
        raise SystemExit("no hay nucleos nativos utilizables aqui")
    os.makedirs(argumentos.salida, exist_ok=True)

    if argumentos.wav:
        crudo, ritmo = leer_wav(argumentos.wav)
        print(f"entrada: {argumentos.wav} ({len(crudo) / ritmo:.2f} s a {ritmo} Hz)")
    else:
        crudo, ritmo = hablar(argumentos.texto, argumentos.voz,
                              argumentos.velocidad)
        print(f"entrada: espeak-ng '{argumentos.texto}' "
              f"({len(crudo) / ritmo:.2f} s a {ritmo} Hz)")
    x = remuestrear(crudo, ritmo, RITMO)
    pico = np.abs(x).max()
    if pico > 0:
        x = (x / pico * 0.85).astype(np.float32)
    print(f"         {len(x)} muestras a {RITMO} Hz -> x{RATIO} por el nucleo "
          f"-> vuelta a {RITMO} Hz, en fotogramas de "
          f"{FOTOGRAMA * 1000 // RITMO} ms")

    conv = capa_subida(canales_ent=CANALES_ENT, canales_sal=CANALES_SAL)
    nativa = ConvTrNativa(conv, lib)
    a = pasar(x, conv, None)
    b = pasar(x, conv, nativa)
    n = min(len(a), len(b), len(x))
    borde = K * RATIO
    a, b, orig = a[borde:n], b[borde:n], x[borde:n]

    # fp32 e int8 comparten camino, asi que se comparan tal cual. Contra la
    # ENTRADA hay que descontar antes el retardo de grupo del filtro (~1
    # muestra) y su ganancia (~0,6%): son del remuestreo, no del int8, y sin
    # descontarlos el numero sale ~22 dB peor y parece dano que no existe.
    ruido = a - b
    snr = 10 * np.log10(float((a ** 2).sum()) / float((ruido ** 2).sum() + 1e-30))
    margen = 2000
    desfase, mejor = 0, -9.0
    for lag in range(-16, 17):
        trozo = a[margen + lag:len(a) - margen + lag]
        cc = float(np.corrcoef(orig[margen:len(a) - margen], trozo)[0, 1])
        if cc > mejor:
            desfase, mejor = lag, cc
    o = orig[margen:len(a) - margen]
    f = a[margen + desfase:len(a) - margen + desfase]
    ganancia = float((o * f).sum() / (f * f).sum())
    transparencia = 10 * np.log10(
        float((o ** 2).sum()) / float(((o - f * ganancia) ** 2).sum() + 1e-30))
    print(f"\n  el viaje en fp32, ya de por si:     {transparencia:6.2f} dB "
          f"(entrada vs fp32, descontando {desfase} muestra de retardo "
          f"y x{ganancia:.4f})")
    print(f"  lo que anade el nucleo int8:        {snr:6.2f} dB   "
          f"correlacion {float(np.corrcoef(a, b)[0, 1]):.8f}")
    print(f"  ruido rms {np.sqrt((ruido ** 2).mean()):.2e}  "
          f"pico de la senal {np.abs(a).max():.3f}")

    # amplificar la resta hasta que se oiga, y decir cuanto se amplifico
    tope = np.abs(ruido).max()
    ganancia = min(200.0, 0.7 / tope) if tope > 0 else 1.0
    for nombre, datos in (("frase_entrada.wav", orig),
                          ("frase_fp32.wav", a),
                          ("frase_int8.wav", b),
                          ("frase_diferencia.wav", ruido * ganancia)):
        ruta = os.path.join(argumentos.salida, nombre)
        escribir_wav(ruta, datos)
        extra = f"  (x{ganancia:.0f})" if "diferencia" in nombre else ""
        print(f"  escrito {ruta}{extra}")


if __name__ == "__main__":
    main()
