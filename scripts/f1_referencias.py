#!/usr/bin/env python3
"""F1 del plan de mejora: ¿que referencia transfiere mejor el ritmo, y cuantos segundos hacen falta?

Por identidad del banco de reconstruccion (con consentimiento), y SOLO con segmentos que no se
evaluan (el clon nunca oye lo que luego se le compara):

  dobla30        la regla de produccion: los mas tipicos (centroide fijo) hasta 30 s. Control.
  espontanea30   entre los tipicos (parecido >= mediana), los de MAS pausas por minuto, hasta 30 s
  neutra30       entre los tipicos, los de MENOS pausas por minuto, hasta 30 s
  c5 c8 c12 c20  la combinacion de tramos tipicos mas cercana a 5/8/12/20 s: la curva real
  rep2 rep3      los mismos clips de c5 repetidos 2 y 3 veces: ¿mas latentes del mismo audio ayudan?

Deja las referencias, el lote y variantes.json (segundos, pausas/min y silabas/s de cada
referencia, y los del audio evaluado como objetivo), y clona todo con UNA carga del modelo.

  python3 f1_referencias.py --identidad carlos-segura liliana-morales
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(Path.home() / "Documents/GitHub/dobla-fase1/pipeline"))
import perfil_vocal as PVOC  # noqa: E402
import perfil_voz as PV  # noqa: E402
import reconstruccion as R  # noqa: E402

RECON = Path.home() / "Documents/dobla-recon"
MOTOR = Path.home() / "Documents/GitHub/VibeVoiceNix"


def llenar(orden, tope):
    sel, acum = [], 0.0
    for s in orden:
        if acum >= tope:
            break
        sel.append(s)
        acum += s["fin"] - s["ini"]
    return sorted(sel, key=lambda s: s["ini"])


def ajustar(por_centro, n, k=12):
    """La combinacion de 1-3 tramos, entre los k mas tipicos, cuya duracion mas se acerca a n
    segundos (sin pasarse de n + 1); a igualdad, la mas tipica. Los tramos miden 3-18 s: llenar
    'hasta n' daba 6,4 s para 5 y lo mismo para 8 que para 12."""
    from itertools import combinations
    top = por_centro[:k]
    mejor = None
    for r in (1, 2, 3):
        for comb in combinations(top, r):
            dur = sum(s["fin"] - s["ini"] for s in comb)
            if dur > n + 1:
                continue
            clave = (abs(dur - n), -np.mean([s["tipico"] for s in comb]))
            if mejor is None or clave < mejor[0]:
                mejor = (clave, comb)
    return sorted(mejor[1], key=lambda s: s["ini"]) if mejor else []


def resumen(sel):
    dur = sum(s["fin"] - s["ini"] for s in sel)
    pesos = [s["fin"] - s["ini"] for s in sel]
    media = lambda k: float(np.average([s[k] for s in sel], weights=pesos)) if sel else None
    return {"clips": len(sel), "segundos": round(dur, 1), "pausas_min": round(media("pausas_min"), 2),
            "silabas_s": round(media("silabas_s"), 2), "tipico": round(media("tipico"), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identidad", nargs="+", default=["carlos-segura", "liliana-morales"])
    ap.add_argument("--trabajo", default=str(Path.home() / "Documents/mejora-modelo/f1"))
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    trabajo = Path(a.trabajo)
    corpus = json.loads((RECON / "corpus.json").read_text(encoding="utf-8"))
    jueces = R.Jueces(url="sin-whisper")
    lote = []
    for ident in a.identidad:
        c = next(x for x in corpus if x["identidad"] == ident)
        d = RECON / "material" / c["stem"]
        anot = json.loads((d / "anotacion.json").read_text(encoding="utf-8"))
        pista, hz = R.leer(d / "voces24k.wav")
        puros = [s for s in PV.tramos_puros(anot["segmentos"], c["hablante"], min_seg=R.EVAL_MIN_S,
                                            max_seg=R.EVAL_MAX_S) if s["entero"] and s["texto"]]
        evaluados = {(round(s["ini"], 3), round(s["fin"], 3)) for s in c["segmentos"]}
        pool = [s for s in puros if (round(s["ini"], 3), round(s["fin"], 3)) not in evaluados]
        for s in pool:
            x, _ = R.leer_tramo(pista, hz, s)
            s["huella"] = jueces.huella(x, hz)
            p = PVOC.perfil(x, hz, s["texto"])
            s["pausas_min"], s["silabas_s"] = p.get("pausas_min") or 0.0, p.get("silabas_s") or 0.0
        pool = [s for s in pool if s["huella"] is not None]
        centro = PV._unit(np.mean([s["huella"] for s in pool], 0))
        for s in pool:
            s["tipico"] = float(s["huella"] @ centro)
        mediana = float(np.median([s["tipico"] for s in pool]))
        tipicos = [s for s in pool if s["tipico"] >= mediana]
        por_centro = sorted(pool, key=lambda s: -s["tipico"])
        variantes = {
            "dobla30": llenar(por_centro, 30),
            "espontanea30": llenar(sorted(tipicos, key=lambda s: -s["pausas_min"]), 30),
            "neutra30": llenar(sorted(tipicos, key=lambda s: s["pausas_min"]), 30),
            **{f"c{n}": ajustar(por_centro, n) for n in (5, 8, 12, 20)},
        }
        variantes["rep2"] = variantes["c5"] * 2
        variantes["rep3"] = variantes["c5"] * 3
        # el objetivo: como habla de verdad en lo que se evalua
        obj = []
        for s in c["segmentos"]:
            x, h = R.leer(s["wav"])
            p = PVOC.perfil(x, h, s["texto"])
            obj.append((p.get("pausas_min") or 0.0, p.get("silabas_s") or 0.0, s["dur"]))
        w = [o[2] for o in obj]
        info = {"objetivo": {"pausas_min": round(float(np.average([o[0] for o in obj], weights=w)), 2),
                             "silabas_s": round(float(np.average([o[1] for o in obj], weights=w)), 2)},
                "pool": len(pool), "variantes": {}}
        carpeta = trabajo / ident
        for nombre, sel in variantes.items():
            refs = []
            for k, s in enumerate(sel):
                wv = carpeta / "refs" / nombre / f"ref-{k:02d}.wav"
                wv.parent.mkdir(parents=True, exist_ok=True)
                if not wv.exists():
                    sf.write(str(wv), R.leer_tramo(pista, hz, s)[0], hz, subtype="PCM_16")
                refs.append({"audio": str(wv), "transcripcion": s["texto"]})
            voz = f"f1-{ident}-{nombre}"
            info["variantes"][nombre] = {"voz": voz, **resumen(sel)}
            lote.append({"salida": str(trabajo / "clones" / f"{voz}.pt"), "semilla": 11, "refs": refs})
        (carpeta / "variantes.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(f"[f1] {ident}: {len(pool)} tramos; objetivo {info['objetivo']}", flush=True)
        for n, v in info["variantes"].items():
            print(f"    {n:13s} {v['clips']:2d} clips {v['segundos']:5.1f} s  pausas/min {v['pausas_min']:5.1f}  "
                  f"silabas/s {v['silabas_s']:.2f}  tipico {v['tipico']:.3f}", flush=True)
    (trabajo / "clones").mkdir(parents=True, exist_ok=True)
    (trabajo / "lote.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    subprocess.run([a.python, str(MOTOR / "scripts/clonar_voz.py"), "--lote", str(trabajo / "lote.json")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
