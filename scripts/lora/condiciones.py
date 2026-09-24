#!/usr/bin/env python3
"""Condiciones de la cabeza de difusion, precalculadas para destilar la guia (destilar_guia.py).

Por cada ejemplo de datos.py (referencia de voz + objetivo del mismo lector), la pasada forzada de
forzado.estados() da, fotograma a fotograma, la condicion POSITIVA (el tts_lm con texto y voz) y la
NEGATIVA (la rama <|image_pad|> del CFG), mas el latente real. Es lo unico que necesita la cabeza:
con esto guardado, la destilacion no vuelve a pasar por el transformador.

  python3 condiciones.py --datos datos/ --salida condiciones/
Deja condiciones/<idioma>.pt: lista de {idioma, hablante, lat [T,64], cond [T,896], cond_neg [T,896]} en fp16.
"""
import argparse
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    a = ap.parse_args()
    from entrenar import cargar_modelo
    d = "cuda" if torch.cuda.is_available() else "cpu"
    proc, modelo = cargar_modelo(a.modelo, d)
    modelo.eval()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.datos).glob("*.pt")):
        if (sal / f.name).exists():
            print(f"[cond] {f.name}: ya esta", flush=True)
            continue
        filas, fotogramas = [], 0
        with torch.no_grad():
            for ej in torch.load(f, map_location="cpu"):
                r = FZ.estados(modelo, proc.tokenizer, ej)
                if r is None:
                    continue
                lat, cond, cond_neg, _, _ = r
                filas.append({"idioma": ej["idioma"], "hablante": ej["hablante"], "lat": lat.half().cpu(),
                              "cond": cond.half().cpu(), "cond_neg": cond_neg.half().cpu()})
                fotogramas += lat.shape[0]
        torch.save(filas, sal / f.name)
        print(f"[cond] {f.name}: {len(filas)} ejemplos, {fotogramas} fotogramas", flush=True)


if __name__ == "__main__":
    main()
