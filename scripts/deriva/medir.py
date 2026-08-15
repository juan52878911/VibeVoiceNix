#!/usr/bin/env python
"""Mide la deriva de tono de cada WAV: cuartos, Hz, semitonos y RMS.

tono() es copia literal de scripts/sondeo_voz.py (autocorrelacion por ventanas,
60-400 Hz, descarta ventanas sin pico claro) para que los numeros sean
comparables con los medidos en la VM.
"""
import glob
import json
import os
import sys
import wave

import numpy as np

RITMO = 24000


def tono(x, ritmo=RITMO):
    ventana, salto = 1024, 256
    minimo, maximo = ritmo // 400, ritmo // 60
    f0 = []
    for i in range(0, len(x) - ventana, salto):
        t = x[i:i + ventana]
        if np.sqrt(np.mean(t ** 2)) < 0.01:
            continue
        t = t - t.mean()
        r = np.correlate(t, t, mode="full")[ventana - 1:]
        if r[0] <= 0:
            continue
        pico = int(np.argmax(r[minimo:maximo])) + minimo
        if r[pico] / r[0] > 0.3:
            f0.append(ritmo / pico)
    if len(f0) < 5:
        return {"hz": 0.0, "semitonos": 0.0, "tonal": 0.0}
    f0 = np.array(f0)
    bajo, alto = np.percentile(f0, [5, 95])
    return {"hz": float(np.median(f0)),
            "semitonos": float(12 * np.log2(alto / bajo)) if bajo > 0 else 0.0,
            "tonal": len(f0) * salto / len(x)}


def leer(ruta):
    with wave.open(ruta) as w:
        crudo = w.readframes(w.getnframes())
    return np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0


def db(v):
    return 20 * np.log10(v + 1e-12)


def mide(ruta):
    x = leer(ruta)
    cuartos = np.array_split(x, 4)
    hz = [tono(c)["hz"] for c in cuartos]
    rms = [float(np.sqrt(np.mean(c ** 2))) for c in cuartos]
    semis = 12 * np.log2(hz[-1] / hz[0]) if hz[0] > 0 and hz[-1] > 0 else 0.0
    fila = {
        "clip": os.path.basename(ruta),
        "seg": len(x) / RITMO,
        "hz": [round(h, 1) for h in hz],
        "semitonos": round(float(semis), 2),
        "rms_db": [round(db(v), 1) for v in rms],
        "rms_db_delta": round(float(db(rms[-1]) - db(rms[0])), 1),
    }
    lat = ruta.replace(".wav", "-latentes.npy")
    if os.path.exists(lat):
        L = np.load(lat)
        cl = np.array_split(L, 4)
        fila["lat_norma"] = [round(float(np.mean(np.linalg.norm(c, axis=1))), 2) for c in cl]
        fila["lat_std"] = [round(float(np.mean(c.std(axis=1))), 3) for c in cl]
        fila["n_lat"] = int(len(L))
    return fila


def main():
    rutas = []
    for patron in sys.argv[1:]:
        rutas.extend(sorted(glob.glob(patron)))
    filas = [mide(r) for r in rutas]
    cab = f"{'clip':40s} {'seg':>5s} {'1er cuarto':>10s} {'ultimo':>7s} {'semis':>6s} {'dRMS':>6s}"
    print(cab)
    print("-" * len(cab))
    for f in filas:
        print(f"{f['clip']:40s} {f['seg']:5.1f} {f['hz'][0]:10.1f} {f['hz'][-1]:7.1f} "
              f"{f['semitonos']:+6.2f} {f['rms_db_delta']:+6.1f}"
              + (f"   lat {f['lat_norma']}" if "lat_norma" in f else ""))
    print()
    print(json.dumps(filas, indent=1))


if __name__ == "__main__":
    main()
