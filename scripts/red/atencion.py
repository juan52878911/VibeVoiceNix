#!/usr/bin/env python3
"""I4 · Que mira cada cabeza del tts_lm al generar: masa de atencion sobre [latentes del prefijo, texto del prefijo,
lo generado (texto leido + latentes propios)], por capa y cabeza, promediada en los fotogramas de una locucion.

    python3 scripts/red/atencion.py --modelo <ruta> --voces ~/.cache/vibevoice-nix/voces --voz sp-Spk1_man \
        --texto "..." --salida red/atencion.json

Exige atencion 'eager' (el IR de OpenVINO no expone atenciones: esto es de laboratorio, en torch). Las cabezas con
mas masa en los latentes del prefijo son las que sostienen la identidad (no dirigir ahi); las que miran el texto
generado son el alineador fotograma -> ficha del plan de la red (§5.6).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import modelo as MO  # noqa: E402
from bucle import Generador  # noqa: E402


def mejores(att, region, n=10):
    """Las n cabezas (capa, cabeza, masa) con mas masa en esa region."""
    idx = torch.topk(att[..., region].flatten(), n).indices
    return [(int(i // att.shape[1]), int(i % att.shape[1]), round(float(att[i // att.shape[1], i % att.shape[1], region]), 3)) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--aleatorio", action="store_true")
    ap.add_argument("--voces", required=True)
    ap.add_argument("--voz", required=True)
    ap.add_argument("--texto", default="Hoy vamos a hablar de como la nube cambia la manera en que una empresa maneja sus datos.")
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--max-fotogramas", type=int, default=120)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    m, tok = MO.cargar(a.modelo, aleatorio=a.aleatorio, atencion="eager")
    base = MO.prefijo(Path(a.voces) / f"{a.voz}.pt")
    torch.manual_seed(a.semilla)
    g = Generador(m, base, MO.fichas(a.texto, tok), registrar=True, atenciones=True)
    g.correr(max_fotogramas=a.max_fotogramas)
    att = torch.stack([r["att"] for r in g.reg if r["att"] is not None]).mean(0)   # [capas, cabezas, 3]
    res = dict(voz=a.voz, fotogramas=len(g.reg), regiones=["prefijo_latentes", "prefijo_texto", "generado"],
               por_capa=[[round(float(x), 3) for x in att[c].mean(0)] for c in range(att.shape[0])],
               cabezas_identidad=mejores(att, 0), cabezas_texto=mejores(att, 2))
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.salida, "w"), indent=1)
    print("masa media por capa [prefijo_latentes, prefijo_texto, generado]:")
    for c, fila in enumerate(res["por_capa"]):
        print(f"  capa {c:2d}: {fila}")
    print("cabezas que mas miran el prefijo de voz (capa, cabeza, masa):", res["cabezas_identidad"][:5])
    print("cabezas que mas miran lo generado:", res["cabezas_texto"][:5])


if __name__ == "__main__":
    main()
