#!/usr/bin/env python3
"""Base frente a LoRA, pareado por clip (mismo hablante, frase y semilla), por idioma y en las voces con
consentimiento. Las medidas son las de juez_lote.py sobre la salida de evaluar.py.

  python3 comparar.py eval/base/medidas.json eval/c1/medidas.json

Puerta (plan de mejora, F7): WER medio -30 % relativo con IC < 0; UTMOS con IC inferior >= -0,02; identidad
+-0,005 global o mejor; ningun idioma con WER peor con IC > 0.
"""
import json
import random
import sys
from collections import defaultdict


def ic(v, n=4000):
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(n))
    return sum(v) / len(v), b[int(0.025 * n)], b[int(0.975 * n)]


def main():
    base, nuevo = (json.load(open(p)) for p in sys.argv[1:3])
    grupos = defaultdict(lambda: defaultdict(list))
    for k, b in base.items():
        x = nuevo.get(k)
        if not x:
            continue
        ident = b["identidad"]
        grupo = ident.split("-")[0] if "-" in ident and len(ident.split("-")[0]) == 2 else f"voz:{ident}:{b['idioma']}"
        for met in ("wer_norm", "wer", "utmos", "ecapa", "per"):
            if b.get(met) is not None and x.get(met) is not None:
                grupos[grupo][met].append((b[met], x[met]))
                grupos["TODO"][met].append((b[met], x[met]))
    print(f"{'grupo':28s} {'n':>3s} | {'WER base':>8s} {'WER lora':>8s} {'dif (IC 95 %)':>24s} | {'UTMOS dif':>22s} | {'ECAPA dif':>22s}")
    for g in sorted(grupos, key=lambda g: (g == "TODO", g)):
        m = grupos[g]
        w = m.get("wer_norm") or m.get("wer")
        wb = sum(a for a, _ in w) / len(w)
        wl = sum(b for _, b in w) / len(w)
        dw = ic([b - a for a, b in w])
        du = ic([b - a for a, b in m["utmos"]])
        de = ic([b - a for a, b in m["ecapa"]]) if m.get("ecapa") else None
        rel = f"{100 * (wl / wb - 1):+.0f} %" if wb > 0 else "  -  "
        print(f"{g:28s} {len(w):3d} | {wb:8.3f} {wl:8.3f} {dw[0]:+.3f} [{dw[1]:+.3f},{dw[2]:+.3f}] {rel:>6s} | "
              f"{du[0]:+.3f} [{du[1]:+.3f},{du[2]:+.3f}] | "
              + (f"{de[0]:+.3f} [{de[1]:+.3f},{de[2]:+.3f}]" if de else "   -"))


if __name__ == "__main__":
    main()
