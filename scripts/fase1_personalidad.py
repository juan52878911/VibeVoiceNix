#!/usr/bin/env python
"""Fase 1 del plan de personalidad: cada persona contra SU clon, diciendo lo mismo.

    python scripts/fase1_personalidad.py --dataset /var/lib/taller/dataset-voces --trabajo /var/lib/taller/fase1

Lo que hace, por persona del manifiesto (docs/plan-personalidad-voz.md):
  1. REFERENCIA: sus clips más largos hasta ~30 s, con su transcripción anotada -> clonar_voz.py -> .pt
  2. PRUEBA: el resto de sus clips (nunca los de la referencia: si no, el clon se evalúa con lo
     que ya ha oído). Para cada uno, el clon dice su MISMO texto con decir.py (semilla, cfg y pasos
     de producción).
  3. MEDIDA: scripts/perfil_vocal.py sobre el clip real y sobre el del clon, y la diferencia
     clon − real de cada descriptor, con media e IC 95 % por bootstrap.

Las personas con menos de --min-prueba clips de prueba se clonan igual pero no se evalúan: sus
números no significarían nada. Cada paso se salta si su salida ya existe, así que se puede
relanzar a mitad sin repetir lo hecho (en CPU esto va para horas).
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
try:
    import perfil_vocal as PV  # noqa: E402  (necesita librosa; con --sin-medir no hace falta)
except ImportError:
    PV = None


def ic95(dif, n=2000):
    d = np.asarray([v for v in dif if v == v], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    medias = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="carpeta con manifiesto.csv y los clips")
    ap.add_argument("--trabajo", required=True, help="carpeta de salida (voces, clips del clon, informe)")
    ap.add_argument("--segundos-ref", type=float, default=30.0)
    ap.add_argument("--min-prueba", type=int, default=5)
    ap.add_argument("--semilla", type=int, default=101)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sin-medir", action="store_true",
                    help="solo clona y genera: la medida se hace en otra maquina (la VM no tiene librosa)")
    a = ap.parse_args()

    ds, tr = Path(a.dataset), Path(a.trabajo)
    (tr / "voces").mkdir(parents=True, exist_ok=True)
    filas = list(csv.DictReader(open(ds / "manifiesto.csv", encoding="utf-8")))
    por = {}
    for f in filas:
        por.setdefault(f["hablante"], []).append(f)

    informe = {}
    for persona, clips in sorted(por.items(), key=lambda kv: -len(kv[1])):
        clips = sorted(clips, key=lambda f: -float(f["duracion_s"]))
        ref, acum = [], 0.0
        for f in clips:
            if acum >= a.segundos_ref:
                break
            ref.append(f)
            acum += float(f["duracion_s"])
        prueba = [f for f in clips if f not in ref]
        pt = tr / "voces" / f"{persona}.pt"
        print(f"\n== {persona}: referencia {len(ref)} clips ({acum:.0f} s) · prueba {len(prueba)} clips", flush=True)

        if not pt.exists():
            orden = [a.python, str(RAIZ / "clonar_voz.py"), "--salida", str(pt), "--semilla", "11"]
            for f in ref:
                orden += ["--audio", str(ds / f["fichero"]), "--transcripcion", f["texto"]]
            subprocess.run(orden, check=True)

        if len(prueba) < a.min_prueba:
            print(f"   {len(prueba)} clips de prueba (< {a.min_prueba}): clonado, sin evaluar", flush=True)
            continue

        destino = tr / "clon" / persona
        destino.mkdir(parents=True, exist_ok=True)
        pendientes = [f for f in prueba if not (destino / Path(f["fichero"]).name).exists()]
        if pendientes:
            # UNA llamada a decir.py por persona con todos sus textos: el modelo tarda ~40 s en
            # cargar y por clip serian horas solo cargando. decir.py escribe <salida>-1.wav,
            # <salida>-2.wav... en el orden de los --texto, y aqui se renombran al clip real.
            orden = [a.python, str(RAIZ / "decir.py"), "--voz", persona, "--voces", str(tr / "voces"),
                     "--salida", str(destino / "tanda"), "--semilla", str(a.semilla),
                     "--cfg", str(a.cfg), "--pasos", str(a.pasos)]
            for f in pendientes:
                orden += ["--texto", f["texto"]]
            subprocess.run(orden, check=True)
            for i, f in enumerate(pendientes, 1):
                (destino / f"tanda-{i}.wav").rename(destino / Path(f["fichero"]).name)

        if a.sin_medir or PV is None:
            print(f"   {len(prueba)} clips del clon en {destino} (sin medir)", flush=True)
            continue

        reales, clones = [], []
        for f in prueba:
            xr, sr = PV.leer(ds / f["fichero"])
            xc, src = PV.leer(destino / Path(f["fichero"]).name)
            reales.append(PV.perfil(xr, sr, f["texto"]))
            clones.append(PV.perfil(xc, src, f["texto"]))
        filas_p = {}
        for k in PV.CLAVES:
            m, lo, hi = ic95([c[k] - r[k] for r, c in zip(reales, clones)])
            filas_p[k] = {"real": statistics.median([r[k] for r in reales if r[k] == r[k]] or [float("nan")]),
                          "clon": statistics.median([c[k] for c in clones if c[k] == c[k]] or [float("nan")]),
                          "dif": m, "ic": [lo, hi], "significativo": not (lo <= 0 <= hi)}
        informe[persona] = {"n": len(prueba), "descriptores": filas_p}
        print(f"   {'descriptor':22s} {'real':>8s} {'clon':>8s} {'clon-real [IC 95 %]':>28s}")
        for k, v in filas_p.items():
            marca = " <-" if v["significativo"] else ""
            print(f"   {k:22s} {v['real']:8.2f} {v['clon']:8.2f} {v['dif']:+9.2f} [{v['ic'][0]:+.2f}, {v['ic'][1]:+.2f}]{marca}", flush=True)

    with open(tr / "informe_fase1.json", "w") as fh:
        json.dump(informe, fh, ensure_ascii=False, indent=1)
    print(f"\ninforme en {tr / 'informe_fase1.json'}")


if __name__ == "__main__":
    main()
