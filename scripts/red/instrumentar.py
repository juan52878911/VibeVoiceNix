#!/usr/bin/env python3
"""I0 · Generar con la red a la vista: por cada clip, el audio y, por fotograma, la condicion, la negativa, el
residual de las 20 capas, el latente, p_fin y la ventana de texto leida. Es la materia prima de sondas.py.

    python3 scripts/red/instrumentar.py --modelo <ruta> --voces ~/.cache/vibevoice-nix/voces --salida red/inst \
        --voz sp-Spk1_man --voz sp-Spk0_woman --corpus scripts/corpus_mejora.json --grupos es,en --semillas 11,101

Deja <salida>/<voz>__<grupo><k>__s<semilla>.wav (24 kHz) y .npz con cond [T,896], neg [T,896], res [T,20,896]
(float16), lat [T,64], p_fin [T], ventana [T]. Con --aleatorio genera con pesos al azar (solo para probar la tuberia).
Con pesos reales en CPU cuesta lo que generate() (RTF ~2-3 en torch fp32 en un i7; en el M4 menos).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import modelo as MO  # noqa: E402
from bucle import Generador  # noqa: E402


def guardar_wav(ruta, onda, hz=24000):
    import soundfile as sf
    sf.write(str(ruta), onda.clamp(-1, 1).numpy(), hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--aleatorio", action="store_true")
    ap.add_argument("--voces", required=True, help="carpeta con los .pt")
    ap.add_argument("--voz", action="append", required=True)
    ap.add_argument("--corpus", default=str(AQUI.parent / "corpus_mejora.json"))
    ap.add_argument("--grupos", default="es,en")
    ap.add_argument("--semillas", default="11,101")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--max-fotogramas", type=int, default=450)
    ap.add_argument("--sin-residuales", action="store_true")
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    m, tok = MO.cargar(a.modelo, aleatorio=a.aleatorio)
    corpus = json.loads(Path(a.corpus).read_text())
    frases = [(g, k, t) for g in a.grupos.split(",") for k, t in enumerate(corpus[g])]
    for voz in a.voz:
        base = MO.prefijo(Path(a.voces) / f"{voz}.pt")
        for g, k, texto in frases:
            for s in (int(x) for x in a.semillas.split(",")):
                nombre = f"{voz}__{g}{k}__s{s}"
                if (sal / f"{nombre}.npz").exists():
                    continue
                t0 = time.time()
                torch.manual_seed(s)
                gen = Generador(m, base, MO.fichas(texto, tok), cfg_scale=a.cfg, registrar=True, residuales=not a.sin_residuales)
                onda = gen.correr(max_fotogramas=a.max_fotogramas)
                r = gen.reg
                np.savez_compressed(sal / f"{nombre}.npz",
                                    cond=torch.stack([x["cond"] for x in r]).numpy(),
                                    neg=torch.stack([x["neg"] for x in r]).numpy(),
                                    res=(torch.stack([x["res"] for x in r]).half().numpy() if not a.sin_residuales else np.zeros(0)),
                                    lat=torch.stack([x["lat"] for x in r]).numpy(),
                                    p_fin=np.array([x["p_fin"] for x in r]), ventana=np.array([x["ventana"] for x in r]),
                                    texto=texto, voz=voz, semilla=s)
                guardar_wav(sal / f"{nombre}.wav", onda)
                print(f"{nombre}: {len(r)} fotogramas, {onda.shape[0] / 24000:.1f} s, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
