#!/usr/bin/env python3
"""Tabla de la F1: por identidad y variante de referencia, identidad (ECAPA), WER, UTMOS y ritmo
del clon, con IC 95 % por bootstrap, frente a la referencia y al objetivo (como habla de verdad).

  python3 f1_informe.py medidas.json [--trabajo ~/Documents/mejora-modelo/f1]
"""
import json
import random
import sys
from pathlib import Path

m = json.load(open(sys.argv[1]))
trabajo = Path(sys.argv[3] if len(sys.argv) > 3 else Path.home() / "Documents/mejora-modelo/f1")


def ic(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, None
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(3000))
    return sum(v) / len(v), b[75], b[2924]


def f(v, d=3):
    x, a, b = ic(v)
    return "   -   " if x is None else f"{x:.{d}f} [{a:.{d}f},{b:.{d}f}]"


for ident in sorted({k.split("__")[0] for k in m}):
    info = json.load(open(trabajo / ident / "variantes.json"))
    print(f"\n===== {ident}   objetivo: pausas/min {info['objetivo']['pausas_min']}, silabas/s {info['objetivo']['silabas_s']}")
    print(f"{'variante':13s} {'ref s':>5s} {'ref p/m':>7s} | {'ECAPA es':>21s} {'ECAPA en':>21s} | {'WER es':>21s} {'WER en':>7s} | {'UTMOS es':>21s} | {'pausas/min es':>21s} {'silabas/s es':>21s}")
    for var, v in info["variantes"].items():
        es = [x for k, x in m.items() if k.startswith(f"{ident}__{var}__es")]
        en = [x for k, x in m.items() if k.startswith(f"{ident}__{var}__en")]
        if not es:
            continue
        wen = ic([x["wer"] for x in en])[0]
        print(f"{var:13s} {v['segundos']:5.1f} {v['pausas_min']:7.1f} | {f([x.get('ecapa') for x in es])} {f([x.get('ecapa') for x in en])} | "
              f"{f([x['wer'] for x in es])} {wen if wen is None else round(wen, 3)!s:>7s} | {f([x['utmos'] for x in es], 2)} | "
              f"{f([x['pausas_min'] for x in es], 1)} {f([x['silabas_s'] for x in es], 2)}   n={len(es)}+{len(en)}")

# ---- pareado contra el control (mismo segmento, misma semilla): lo que decide las puertas
CONTROL = {"espontanea30": "dobla30", "neutra30": "dobla30", "rep2": "c5", "rep3": "c5", "c8": "c5", "c12": "c5",
           "c20": "c5", "dobla30": "c5"}
print("\n--- pareado (variante - control), IC 95 %")
for ident in sorted({k.split("__")[0] for k in m}):
    obj = json.load(open(trabajo / ident / "variantes.json"))["objetivo"]
    for var, ctl in CONTROL.items():
        filas = {}
        for k, x in m.items():
            i, v, frase, sem = k.split("__")
            if i != ident or v != var:
                continue
            y = m.get(f"{ident}__{ctl}__{frase}__{sem}")
            if not y:
                continue
            lengua = "en" if frase.startswith("en") else "es"
            for met in ("ecapa", "wer", "utmos"):
                if x.get(met) is not None and y.get(met) is not None:
                    filas.setdefault(f"{met}_{lengua}", []).append(x[met] - y[met])
            if lengua == "es" and x.get("pausas_min") is not None and y.get("pausas_min") is not None:
                filas.setdefault("|pausas-obj|", []).append(abs(x["pausas_min"] - obj["pausas_min"]) - abs(y["pausas_min"] - obj["pausas_min"]))
            if lengua == "es" and x.get("silabas_s") and y.get("silabas_s"):
                filas.setdefault("|silabas-obj|", []).append(abs(x["silabas_s"] - obj["silabas_s"]) - abs(y["silabas_s"] - obj["silabas_s"]))
        if filas:
            print(f"{ident:16s} {var:13s} vs {ctl:8s} " + "  ".join(
                f"{k} {ic(v)[0]:+.3f} [{ic(v)[1]:+.3f},{ic(v)[2]:+.3f}]" for k, v in filas.items() if k in ("ecapa_es", "ecapa_en", "wer_es", "utmos_es", "|pausas-obj|")))
