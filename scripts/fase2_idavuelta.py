#!/usr/bin/env python
"""Pasa TODOS los clips reales del dataset por el códec de VibeVoice (codificador -> decodificador).

    python scripts/fase2_idavuelta.py --dataset /var/lib/taller/dataset-voces --salida /var/lib/taller/idavuelta

Para que el juez real/clon de la fase 2 no pueda apoyarse en el CÓDEC: el control de canal
(scripts/fase2_control_canal.py) midió que el juez marcaba como clon el audio real pasado por el
códec (AUC 0,996 real frente a ida y vuelta). Con estos clips etiquetados como reales, lo único que
separa real de clon es lo que cambia la GENERACIÓN. Deja <salida>/<persona>/<clip>.wav y se salta los
que ya existen.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fase2_control_canal as CC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=os.environ.get("VIBEVOICE_MODELO"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    import soundfile as sf
    import torch
    torch.set_num_threads(6)
    filas = list(csv.DictReader(open(Path(a.dataset) / "manifiesto.csv", encoding="utf-8")))
    pendientes = [f for f in filas if not (Path(a.salida) / f["fichero"]).exists()]
    print(f"{len(filas)} clips, {len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return
    codec = CC.cargar_codec(a.modelo, a.cache)
    t0, seg = time.time(), 0.0
    for n, f in enumerate(pendientes, 1):
        x, sr = sf.read(str(Path(a.dataset) / f["fichero"]), dtype="float32")
        y = CC.ida_y_vuelta(codec, x)
        destino = Path(a.salida) / f["fichero"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), y, sr, subtype="PCM_16")
        seg += len(x) / sr
        if n % 10 == 0 or n == len(pendientes):
            dt = time.time() - t0
            print(f"  {n}/{len(pendientes)} · {seg / 60:.1f} min de audio en {dt / 60:.1f} min (RTF {dt / seg:.2f})", flush=True)


if __name__ == "__main__":
    main()
