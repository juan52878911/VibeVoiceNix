#!/usr/bin/env python
"""Que hay REALMENTE en los fotogramas que el respiro inserta, y que les hace
el espejo.

No reproduce audio. Mide:
  - donde caen las pausas del modelo (rachas de fotogramas bajo RESPIRO_UMBRAL)
  - que es cada fotograma de esa pausa: RMS, pico, PENDIENTE de la envolvente
    (¿es ruido plano o es una cola que se apaga?), centroide espectral
  - que le hace el espejo: correlacion entre el fotograma y su inversa, y si
    la envolvente cambia de signo (una cola que baja se convierte en un
    crescendo, y eso es lo que el oido llama "audio al reves")
  - lo mismo sobre una pausa NATURAL del modelo (ref_saltolinea.wav) para
    tener con que comparar
"""
import pathlib
import sys
import wave

import numpy as np

RITMO = 24_000
FOT = 3200                 # muestras por fotograma (un latente)
UMBRAL = 0.006             # RESPIRO_UMBRAL
PICO = 0.03                # RESPIRO_PICO
AQUI = pathlib.Path(__file__).parent


def leer(p):
    with wave.open(str(p), "rb") as w:
        cru = w.readframes(w.getnframes())
    return np.frombuffer(cru, "<i2").astype(np.float32) / 32768.0


def fotogramas(x):
    n = len(x) // FOT
    return x[:n * FOT].reshape(n, FOT)


def rms(v):
    return float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))


def centroide(v):
    """Centroide espectral en Hz. Dice si el trozo es grave (cola de voz) o
    agudo (siseo de suelo de sala)."""
    if rms(v) < 1e-9:
        return 0.0
    esp = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    f = np.fft.rfftfreq(len(v), 1 / RITMO)
    return float((esp * f).sum() / max(esp.sum(), 1e-12))


def pendiente_db(v, trozos=8):
    """dB que sube o baja la envolvente a lo largo del fragmento, por regresion
    sobre el RMS de `trozos` sub-bloques. Negativo = se esta apagando."""
    sub = v[:len(v) // trozos * trozos].reshape(trozos, -1)
    e = 20 * np.log10(np.sqrt((sub.astype(np.float64) ** 2).mean(1)) + 1e-12)
    return float(np.polyfit(np.arange(trozos), e, 1)[0] * (trozos - 1))


def rachas(fs, umbral=UMBRAL):
    """(inicio, largo) de cada racha de fotogramas callados."""
    callado = np.array([rms(f) < umbral for f in fs])
    out, i = [], 0
    while i < len(callado):
        if callado[i]:
            j = i
            while j < len(callado) and callado[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def salto_max(x):
    return float(np.abs(np.diff(x)).max())


def informe(nombre, ruta):
    x = leer(ruta)
    fs = fotogramas(x)
    print(f"\n===== {nombre}  ({len(x)/RITMO:.2f} s, {len(fs)} fotogramas, "
          f"resto {len(x) % FOT} muestras)")
    rs = rachas(fs)
    print(f"rachas calladas: {len(rs)}  ->  " +
          ", ".join(f"@{i} x{n}" for i, n in rs))
    return x, fs, rs


def main():
    base, fs, rs = informe("BASE (costura de espacio)", AQUI / "base_espacio.wav")
    print("\n--- el fotograma que el respiro copiaria en espejo, uno por pausa")
    print(f"{'pausa':>6} {'fot':>4} {'rms':>9} {'pico':>8} {'centr Hz':>9} "
          f"{'pend dB':>8} {'corr(x,flip)':>12} {'zc/ms':>7}")
    victimas = []
    for i, n in rs:
        if n < 2:
            continue
        # el respiro inserta cuando _callado_seguido == RESPIRO_FOTOGRAMAS (2),
        # o sea justo despues de emitir el SEGUNDO callado: el trozo copiado es
        # ese segundo fotograma de la racha.
        k = i + 1
        v = fs[k]
        victimas.append(k)
        c = float(np.corrcoef(v, v[::-1])[0, 1]) if rms(v) > 1e-9 else 0.0
        zc = float((np.diff(np.signbit(v)) != 0).sum()) / (FOT / RITMO * 1000)
        print(f"{i:>6} {k:>4} {rms(v):>9.5f} {np.abs(v).max():>8.5f} "
              f"{centroide(v):>9.0f} {pendiente_db(v):>8.2f} {c:>12.3f} "
              f"{zc:>7.2f}")

    print("\n--- por comparacion: fotogramas de HABLA (los 5 primeros sonoros)")
    sonoros = [k for k in range(len(fs)) if rms(fs[k]) >= UMBRAL][:5]
    for k in sonoros:
        v = fs[k]
        print(f"       {k:>4} {rms(v):>9.5f} {np.abs(v).max():>8.5f} "
              f"{centroide(v):>9.0f} {pendiente_db(v):>8.2f}")

    # --- la pausa NATURAL del modelo, para comparar
    ref, fr, rr = informe("REFERENCIA (costura \\n\\n: el modelo pausa)",
                          AQUI / "ref_saltolinea.wav")
    largas = [(i, n) for i, n in rr if n >= 4]
    print("\n--- dentro de una pausa larga del modelo, fotograma a fotograma")
    print(f"{'fot':>5} {'rms':>9} {'pico':>8} {'centr Hz':>9} {'pend dB':>8}")
    for i, n in largas[:4]:
        print(f"  pausa @{i} de {n} fotogramas ({n*FOT/RITMO:.2f} s)")
        for k in range(i, i + n):
            v = fr[k]
            print(f"{k:>5} {rms(v):>9.5f} {np.abs(v).max():>8.5f} "
                  f"{centroide(v):>9.0f} {pendiente_db(v):>8.2f}")

    # --- estadistica agregada: suelo de pausa natural vs. lo que se inserta
    def agrega(fsx, idxs, etiqueta):
        if not idxs:
            return
        r = [rms(fsx[k]) for k in idxs]
        c = [centroide(fsx[k]) for k in idxs]
        p = [pendiente_db(fsx[k]) for k in idxs]
        print(f"{etiqueta:<38} n={len(idxs):>3}  rms {np.median(r):.5f}  "
              f"centroide {np.median(c):>6.0f} Hz  pendiente {np.median(p):>+6.2f} dB "
              f"(min {min(p):+.2f} / max {max(p):+.2f})")

    print("\n--- resumen")
    agrega(fs, victimas, "fotograma copiado por el respiro")
    interiores = [k for i, n in rr if n >= 4 for k in range(i + 1, i + n - 1)]
    agrega(fr, interiores, "interior de pausa natural del modelo")
    bordes = [i for i, n in rr if n >= 4]
    agrega(fr, bordes, "primer fotograma de pausa natural")

    print(f"\nsalto maximo entre muestras, base: {salto_max(base):.4f}")


if __name__ == "__main__":
    sys.exit(main())
