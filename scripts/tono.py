#!/usr/bin/env python
"""Corregir el TONO de un clon a la salida, sin tocar la duracion.

    python scripts/tono.py clon.wav --st +1.9 -o corregido.wav
    python scripts/tono.py clon.wav --referencia ref.wav -o corregido.wav   # mide y corrige

POR QUE
El clonado sale desplazado de tono y el desplazamiento es SISTEMATICO por voz:
MEDIDO en el banco de semillas (docs/clonado-de-voz.md §7.11), Laura sale
entre -1,9 y -6,2 st de su tono real segun la semilla, y Juan Pablo -2,9 a
-4,9 st; en §7.8 las voces graves salian +2,5 st. El juez de identidad
(ECAPA) apenas lo nota; una persona si.

COMO
Desplazar tono = remuestrear (cambia tono Y duracion) + WSOLA (devuelve la
duracion sin tocar el tono). Subir `st` semitonos: se remuestrea la senal a
1/r con r = 2^(st/12) -- queda mas corta y mas aguda -- y se estira x r con
el WSOLA de pkgs/vibevoice-cli/estirar.py. Solo numpy, como todo lo que
corre en la VM.

El remuestreo es sinc con ventana (16 lobulos), no np.interp: la lineal
apaga los agudos y se oye. Las formantes se desplazan con el tono, asi que
por encima de +-3 st la voz suena mas pequena o mas grande; para el sesgo
medido (2-4 st) es un precio aceptable, y se mide (scripts/banco_tono.py).
"""
import argparse
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np

_aqui = Path(__file__).resolve().parent
sys.path[:0] = [str(_aqui), str(_aqui.parent / "pkgs" / "vibevoice-cli")]
from estirar import estirar  # noqa: E402

RITMO = 24000


def remuestrear_sinc(x, factor, lobulos=16):
    """y[n] = x(n * factor): factor > 1 acorta (y sube el tono). Sinc con
    ventana Hann, `lobulos` a cada lado. Para factor > 1 se filtra a la
    nueva Nyquist para no aliasear."""
    x = np.asarray(x, dtype=np.float64)
    n_out = int(round(x.size / factor))
    corte = min(1.0, 1.0 / factor)          # fraccion de Nyquist que sobrevive
    pos = np.arange(n_out) * factor
    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    y = np.zeros(n_out)
    for k in range(-lobulos + 1, lobulos + 1):
        idx = base + k
        d = k - frac                           # distancia en muestras
        nucleo = corte * np.sinc(corte * d)
        vent = 0.5 + 0.5 * np.cos(np.pi * d / lobulos)
        vent[np.abs(d) >= lobulos] = 0.0
        valido = (idx >= 0) & (idx < x.size)
        y[valido] += x[idx[valido]] * nucleo[valido] * vent[valido]
    return y.astype(np.float32)


def desplazar_remuestreo(x, st, hz=RITMO):
    """Remuestreo + WSOLA: mueve el tono Y las formantes (la voz suena mas
    pequena o mas grande). MEDIDO: +1,86 st sobre un clon baja su ECAPA
    contra si mismo a 0,58: el juez de identidad ve las formantes. Se deja
    como referencia; el camino es PSOLA."""
    if abs(st) < 0.05:
        return np.asarray(x, dtype=np.float32)
    r = 2.0 ** (st / 12.0)
    agudo = remuestrear_sinc(x, r)             # mas corto (1/r) y r veces mas agudo
    y = estirar(agudo, 1.0 / r)                # WSOLA: factor < 1 alarga
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size, np.float32)])
    return y[: x.size]


def marcas_de_periodo(x, hz=RITMO, f0_ref=None):
    """Marcas de tono: una por periodo en lo sonoro (alineadas al pico local),
    cada 5 ms en lo sordo. Devuelve (marcas, periodos) en muestras. Con
    f0_ref, el detector va acotado a la banda de esa voz (sin octavas)."""
    from prosodia import SALTO, contorno, corregir_octavas
    f_min, f_max = banda(f0_ref)
    f0 = contorno(x, hz, f_min=f_min, f_max=f_max)
    if f0_ref is None:
        f0 = corregir_octavas(f0)
    marcas, periodos = [], []
    n, i = x.size, 0
    T_sordo = int(0.005 * hz)
    while i < n - 1:
        k = min(len(f0) - 1, i // SALTO)
        f = f0[k] if len(f0) else np.nan
        if np.isnan(f) or f <= 0:
            i += T_sordo
            marcas.append(i); periodos.append(T_sordo)
            continue
        T = int(round(hz / f))
        # alinear al pico de energia local: los periodos se cortan en el mismo sitio
        lo, hi = max(0, i + T - T // 4), min(n, i + T + T // 4)
        if hi - lo > 2:
            i = lo + int(np.argmax(np.abs(x[lo:hi])))
        else:
            i += T
        marcas.append(i); periodos.append(T)
    return np.array(marcas), np.array(periodos)


def desplazar_psola(x, st, hz=RITMO, f0_ref=None):
    """TD-PSOLA: el tono sube `st` semitonos y las formantes (el timbre) se
    quedan donde estaban. Cada periodo de la senal, con ventana Hann de dos
    periodos, se vuelve a colocar con el espaciado nuevo (T/r); la duracion
    no cambia porque el reloj de salida avanza en tiempo real y solo cambia
    que trozo se copia en cada marca."""
    x = np.asarray(x, dtype=np.float32)
    if abs(st) < 0.05:
        return x
    r = 2.0 ** (st / 12.0)
    marcas, periodos = marcas_de_periodo(x, hz, f0_ref)
    if marcas.size < 3:
        return x
    y = np.zeros(x.size + 4 * int(periodos.max()), np.float64)
    norma = np.zeros_like(y)
    pos = float(marcas[0])
    j = 0
    while pos < x.size:
        # la marca de entrada mas cercana en el tiempo (la duracion no cambia)
        while j + 1 < marcas.size and abs(marcas[j + 1] - pos) < abs(marcas[j] - pos):
            j += 1
        T = int(periodos[j])
        m = int(marcas[j])
        lo, hi = m - T, m + T
        if lo >= 0 and hi <= x.size and T > 2:
            seg = x[lo:hi] * np.hanning(2 * T)
            ini = int(round(pos)) - T
            if ini >= 0 and ini + 2 * T <= y.size:
                y[ini:ini + 2 * T] += seg
                norma[ini:ini + 2 * T] += np.hanning(2 * T)
        pos += T / r                            # el tono nuevo: periodo T/r
    y = y[: x.size] / np.maximum(norma[: x.size], 1e-3)
    # donde la norma no cubre (bordes) se conserva la senal original
    hueco = norma[: x.size] < 1e-3
    y[hueco] = x[hueco]
    return y.astype(np.float32)


def desplazar_tono(x, st, hz=RITMO, metodo="psola", f0_ref=None):
    """x con el tono `st` semitonos mas alto (negativo = mas grave) y la misma duracion."""
    if metodo == "remuestreo":
        return desplazar_remuestreo(x, st, hz)
    return desplazar_psola(x, st, hz, f0_ref)


def banda(f0_ref, ancho=1.5):
    """La banda de busqueda de f0 alrededor de la voz: [f0/1,5, f0*1,5] no
    deja sitio al error de octava (que necesita un factor 2)."""
    if not f0_ref:
        return 60, 400
    return max(50, f0_ref / ancho), min(500, f0_ref * ancho)


def f0_mediana(x, hz=RITMO, f0_ref=None):
    """Mediana de f0 en lo sonoro. Con f0_ref, acotada a su banda."""
    from prosodia import contorno, corregir_octavas
    f_min, f_max = banda(f0_ref)
    f = contorno(np.asarray(x, dtype=np.float32), hz, f_min=f_min, f_max=f_max)
    if f0_ref is None:
        f = corregir_octavas(f)
    v = f[~np.isnan(f)]
    return float(np.median(v)) if len(v) >= 5 else 0.0


def sesgo_st(x, f0_ref, hz=RITMO):
    f0 = f0_mediana(x, hz, f0_ref)
    return 12 * math.log2(f0 / f0_ref) if f0 and f0_ref else None


def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x, w.getframerate()


def escribir_wav(ruta, x, hz=RITMO):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(hz)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("-o", required=True)
    ap.add_argument("--st", type=float, default=None, help="semitonos a subir (negativo baja)")
    ap.add_argument("--referencia", default=None,
                    help="wav de la voz real: se mide el sesgo y se corrige entero")
    args = ap.parse_args()
    x, hz = leer_wav(args.entrada)
    st = args.st
    if st is None:
        if not args.referencia:
            raise SystemExit("hace falta --st o --referencia")
        ref, hz_r = leer_wav(args.referencia)
        f0_ref = f0_mediana(ref, hz_r)
        s = sesgo_st(x, f0_ref, hz)
        if s is None:
            raise SystemExit("no se pudo medir el tono")
        st = -s
        print(f"sesgo medido {s:+.2f} st -> correccion {st:+.2f} st")
    y = desplazar_tono(x, st, hz, f0_ref=locals().get("f0_ref"))
    escribir_wav(args.o, y, hz)
    print(f"{args.o}: {st:+.2f} st, f0 {f0_mediana(x, hz):.0f} -> {f0_mediana(y, hz):.0f} Hz")


if __name__ == "__main__":
    main()
