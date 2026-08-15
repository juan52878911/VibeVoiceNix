#!/usr/bin/env python
"""Espectro de la zona de pausa, contra una pausa NATURAL del propio modelo.

La pregunta que cierra el diagnostico: lo que se mete entre frases, ¿se parece
al aire que el modelo genera cuando pausa de verdad? Se compara por bandas de
octava el interior de una pausa larga de ref_saltolinea.wav (el modelo pausando
por su cuenta) con el material insertado por cada variante.
"""
import pathlib

import numpy as np

from analizar import FOT, RITMO, fotogramas, leer, rms
from variantes import VARIANTES, respirar

AQUI = pathlib.Path(__file__).parent
BANDAS = [(0, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000),
          (2000, 4000), (4000, 8000), (8000, 12000)]


def bandas_db(x):
    """Nivel por banda de octava, en dBFS."""
    if len(x) < 1024:
        return [float("nan")] * len(BANDAS)
    n = 2048
    trozos = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    pot = np.zeros(n // 2 + 1)
    for t in trozos:
        pot += np.abs(np.fft.rfft(t * np.hanning(n))) ** 2
    pot /= max(len(trozos), 1)
    f = np.fft.rfftfreq(n, 1 / RITMO)
    out = []
    for a, b in BANDAS:
        sel = (f >= a) & (f < b)
        out.append(10 * np.log10(pot[sel].sum() / (np.hanning(n) ** 2).sum()
                                 + 1e-20))
    return out


def fila(nombre, x):
    b = bandas_db(x)
    print(f"{nombre:<26} {rms(x):>8.5f} " +
          " ".join(f"{v:>7.1f}" for v in b))


def main():
    ref = fotogramas(leer(AQUI / "ref_saltolinea.wav"))
    # interior de la pausa larga del modelo (fotogramas 105..126), sin los
    # bordes, que llevan la cola de la palabra anterior
    natural = np.concatenate([ref[k] for k in range(105, 126)])

    base = leer(AQUI / "base_espacio.wav")
    fs = fotogramas(base)

    print("nivel por banda de octava, dBFS\n")
    cab = f"{'tramo':<26} {'rms':>8} " + " ".join(
        f"{a//1000 if a >= 1000 else a}-{b//1000 if b >= 1000 else b}"
        .rjust(7) for a, b in BANDAS)
    print(cab)
    print("-" * len(cab))
    fila("pausa NATURAL del modelo", natural)
    for nombre, kw, _ in VARIANTES:
        if kw.get("material") in (None, "ninguno"):
            continue
        x, metidos = respirar(fs, **kw)
        aire = np.concatenate([x[i:f] for i, f in metidos])
        fila(f"aire de {nombre}", aire)

    print("\nY el HABLA, para tener la escala:")
    habla = np.concatenate([f for f in fs if rms(f) >= 0.02][:20])
    fila("habla", habla)

    print("\nLo que hay que mirar: cuanto se aparta cada aire del de la pausa "
          "natural.\nUn aire 8-9 dB mas alto y con energia de mas en 1-4 kHz "
          "no es suelo de sala:\nes el ataque de una palabra metido dentro.")


if __name__ == "__main__":
    main()
