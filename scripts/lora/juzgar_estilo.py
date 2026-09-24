#!/usr/bin/env python3
"""Personalidad del clon frente a la PERSONA diciendo el mismo texto (F8).

Por cada frase en espanol de la evaluacion hay un segmento real de la persona con ESE texto (banco de
reconstruccion): el clon y el original se comparan descriptor a descriptor (perfil_vocal: tono, recorrido,
desviacion, movimiento, microvariacion, silabas/s, pausas/min, rango de energia, inclinacion, HNR) y por
el contorno de entonacion (prosodia.comparar, mismo texto). Cada diferencia se divide por lo que ese
descriptor varia entre los segmentos reales de la persona: la "distancia de estilo" es la media de esas
diferencias en unidades de la propia persona (0 = igual que ella; 1 = una desviacion tipica suya).

  python3 juzgar_estilo.py --voces voces.json --nombre carlos eval/base eval/r16 ...
Por modelo: distancia de estilo media, correlacion de contorno media y el detalle por descriptor, y la
comparacion pareada (mismo segmento y semilla) de cada modelo frente al primero, con IC 95 %.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import perfil_vocal as PV  # noqa: E402
import prosodia as PR  # noqa: E402


def ic(v, n=4000):
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(n))
    return sum(v) / len(v), b[int(0.025 * n)], b[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voces", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("modelos", nargs="+", help="carpetas de evaluar.py (la primera es la referencia)")
    a = ap.parse_args()
    v = json.loads(Path(a.voces).read_text())[a.nombre]
    textos, reales = v["frases"]["es"], v["reales"]
    perf_real = []
    for w, t in zip(reales, textos):
        x, hz = sf.read(w, dtype="float32")
        perf_real.append((x, hz, PV.perfil(x, hz, t)))
    escala = {k: (np.nanstd([p[k] for _, _, p in perf_real]) or 1.0) for k in PV.CLAVES}
    resultado = {}
    for carpeta in a.modelos:
        filas = {}
        for w in sorted(Path(carpeta, "wav").glob(f"{a.nombre}__es*__s*.wav")):
            k = int(w.stem.split("__")[1][2:])
            if k >= len(perf_real):
                continue
            x, hz = sf.read(str(w), dtype="float32")
            xr, hzr, pr = perf_real[k]
            p = PV.perfil(x, hz, textos[k])
            dif = {c: abs(p[c] - pr[c]) / escala[c] for c in PV.CLAVES
                   if not (np.isnan(p[c]) or np.isnan(pr[c]))}
            if hz != hzr:
                import librosa
                x = librosa.resample(x, orig_sr=hz, target_sr=hzr)
            comp = PR.comparar(xr, x, hzr, mismo_texto=True)
            filas[w.stem] = {"distancia": float(np.mean(list(dif.values()))), "corr_contorno": comp["corr_contorno"],
                             **{f"d_{c}": dif.get(c) for c in PV.CLAVES}}
        resultado[carpeta] = filas
        d = [f["distancia"] for f in filas.values()]
        cc = [f["corr_contorno"] for f in filas.values() if not np.isnan(f["corr_contorno"])]
        detalle = "  ".join(f"{c} {np.nanmean([f['d_' + c] for f in filas.values() if f['d_' + c] is not None]):.2f}"
                            for c in PV.CLAVES)
        print(f"{carpeta:28s} n {len(d):3d} · distancia de estilo {np.mean(d):.3f} · contorno {np.mean(cc):.3f}\n    {detalle}",
              flush=True)
    base = resultado[a.modelos[0]]
    for carpeta in a.modelos[1:]:
        comunes = [k for k in base if k in resultado[carpeta]]
        if not comunes:
            continue
        dd = ic([resultado[carpeta][k]["distancia"] - base[k]["distancia"] for k in comunes])
        par = [(base[k]["corr_contorno"], resultado[carpeta][k]["corr_contorno"]) for k in comunes]
        par = [(x, y) for x, y in par if not (np.isnan(x) or np.isnan(y))]
        dc = ic([y - x for x, y in par]) if par else (float("nan"),) * 3
        print(f"{carpeta} - {a.modelos[0]} ({len(comunes)} pares): distancia {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] "
              f"(negativo = mas parecido a la persona) · contorno {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}]", flush=True)
    Path(a.modelos[-1], "estilo.json").write_text(json.dumps(resultado, indent=1))


if __name__ == "__main__":
    main()
