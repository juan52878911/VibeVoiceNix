#!/usr/bin/env python
"""Lupa sobre el fotograma que el respiro repite en espejo.

La pregunta: ¿es suelo de sala plano, o tiene estructura? Si tiene estructura
temporal (energia concentrada al final = el ataque de la palabra siguiente),
el espejo la reproduce al reves y ESO es lo que suena antinatural.

Mide, en sub-bloques de 10 ms:
  - envolvente del fotograma tal cual
  - centroide TEMPORAL de la energia (0 = todo al principio, 1 = todo al final)
  - lo que queda emitido con el respiro actual, muestra a muestra en la junta
"""
import pathlib

import numpy as np

from analizar import FOT, RITMO, UMBRAL, centroide, fotogramas, leer, rms

AQUI = pathlib.Path(__file__).parent
SUB = 240   # 10 ms


def envolvente(v, sub=SUB):
    n = len(v) // sub
    e = np.sqrt((v[:n * sub].reshape(n, sub).astype(np.float64) ** 2).mean(1))
    return 20 * np.log10(e + 1e-12)


def centro_temporal(v):
    """0 = la energia esta toda al principio; 1 = toda al final."""
    e = v.astype(np.float64) ** 2
    if e.sum() <= 0:
        return 0.5
    return float((e * np.arange(len(e))).sum() / e.sum() / (len(e) - 1))


def main():
    x = leer(AQUI / "base_espacio.wav")
    fs = fotogramas(x)
    for k in (21, 42, 62):
        v = fs[k]
        print(f"\n===== fotograma {k}  rms {rms(v):.5f}  pico {np.abs(v).max():.5f}")
        env = envolvente(v)
        print("envolvente por 10 ms (dBFS):")
        print("  " + " ".join(f"{d:6.1f}" for d in env))
        print(f"centro temporal de la energia: {centro_temporal(v):.3f} "
              f"(0,5 = plano; >0,6 = crece hacia el final = hay un ataque dentro)")
        fuertes = np.nonzero(np.abs(v) >= 0.03)[0]
        if len(fuertes):
            print(f"primera muestra |x|>=0,03 en {fuertes[0]} "
                  f"({fuertes[0]/RITMO*1000:.0f} ms de los 133) "
                  f"-> el respiro la repite DOS veces mas")
        print(f"espejo: centro temporal {centro_temporal(v[::-1]):.3f}, "
              f"envolvente al reves")
        # lo que de verdad se emite: v, flip(v), v, y luego el fotograma
        # siguiente (que es el que lleva la palabra entera)
        emitido = np.concatenate([v, v[::-1], v, fs[k + 1]])
        print("cadena emitida por el respiro actual, envolvente 10 ms:")
        e = envolvente(emitido)
        for i in range(0, len(e), 14):
            marca = {0: " <- fotograma real", 14: " <- ESPEJO (invertido)",
                     28: " <- copia", 42: " <- sigue el habla"}.get(i, "")
            print("  " + " ".join(f"{d:6.1f}" for d in e[i:i + 14]) + marca)
        # saltos en las juntas
        for nombre, a, b in (("real|espejo", v[-1], v[::-1][0]),
                             ("espejo|copia", v[::-1][-1], v[0]),
                             ("copia|habla", v[-1], fs[k + 1][0])):
            print(f"  salto en {nombre:<14} {abs(float(a)-float(b)):.5f}")

    # ¿y si en vez del fotograma se metiera silencio? ¿cuanto suelo tiene el
    # modelo de verdad en una pausa suya?
    ref = fotogramas(leer(AQUI / "ref_saltolinea.wav"))
    interior = np.concatenate([ref[k] for k in range(105, 126)])
    print(f"\n===== suelo de sala REAL del modelo (interior de una pausa suya)")
    print(f"rms {rms(interior):.5f}  pico {np.abs(interior).max():.5f}  "
          f"centroide {centroide(interior[:FOT]):.0f} Hz")
    print(f"y el 'suelo' que el respiro inserta hoy: "
          f"rms {rms(fs[21]):.5f} / {rms(fs[42]):.5f}  -> "
          f"{20*np.log10(rms(fs[21])/rms(interior)):+.1f} dB por encima")


if __name__ == "__main__":
    main()
