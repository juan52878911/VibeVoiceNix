#!/usr/bin/env python3
"""DPO, puerta D3 (plan de preferencias, 24-09; fijada ANTES de medir): base frente a LoRA, generando, en
lectores y textos apartados, con las mismas semillas; juez de la PUERTA (whisper large-v3), no el de la
recompensa (medium).

  python3 juez_lote.py base/lote.json base/medidas_puerta.json --identidades base/identidades --rapido
  python3 dpo_puerta.py base/ lora/ [--medidas medidas_puerta.json]

Condiciones (pareado por clip; IC 95 % por bootstrap):
  tramos con WER > 0,15      -25 % relativo o mejor
  WER medio                  <= base (diferencia media <= 0)
  catastrofes                <= base (WER > 0,5, corte, repeticion o tope)
  identidad ECAPA            >= base - 0,005
  UTMOS                      IC inferior de la diferencia >= -0,02
  duracion media             +-5 %
El juez de estilo (juzgar_estilo.py, voces con consentimiento) y la ronda cuantizada van aparte.
"""
import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from dpo_puntuar import ic, medir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("lora")
    ap.add_argument("--medidas", default="medidas_puerta.json")
    a = ap.parse_args()
    b = {m["clave"]: m for ms in medir(Path(a.base), a.medidas).values() for m in ms}
    x = {m["clave"]: m for ms in medir(Path(a.lora), a.medidas).values() for m in ms}
    comunes = sorted(set(b) & set(x))
    B, X = [b[k] for k in comunes], [x[k] for k in comunes]
    n = len(comunes)
    t015_b = sum(m["wer"] > 0.15 for m in B) / n
    t015_x = sum(m["wer"] > 0.15 for m in X) / n
    rel = (t015_x / t015_b - 1) if t015_b > 0 else float("nan")
    d_wer = ic([q["wer"] - p["wer"] for p, q in zip(B, X)])
    d_t015 = ic([float(q["wer"] > 0.15) - float(p["wer"] > 0.15) for p, q in zip(B, X)])
    d_cat = ic([float(q["catastrofe"]) - float(p["catastrofe"]) for p, q in zip(B, X)])
    pe = [(p["ecapa"], q["ecapa"]) for p, q in zip(B, X) if p["ecapa"] is not None and q["ecapa"] is not None]
    d_ec = ic([q - p for p, q in pe])
    d_ut = ic([q["utmos"] - p["utmos"] for p, q in zip(B, X)])
    dur_b, dur_x = sum(m["dur"] for m in B), sum(m["dur"] for m in X)
    cond = {
        "tramos_wer_015": {"base": round(t015_b, 4), "lora": round(t015_x, 4), "relativo": round(rel, 3),
                           "dif_ic": d_t015, "pasa": rel <= -0.25},
        "wer_medio": {"base": round(sum(m["wer"] for m in B) / n, 4), "lora": round(sum(m["wer"] for m in X) / n, 4),
                      "dif_ic": d_wer, "pasa": d_wer[0] <= 0},
        "catastrofes": {"base": sum(m["catastrofe"] for m in B), "lora": sum(m["catastrofe"] for m in X),
                        "dif_ic": d_cat, "pasa": d_cat[0] <= 0},
        "ecapa": {"dif_ic": d_ec, "pasa": d_ec[0] >= -0.005},
        "utmos": {"dif_ic": d_ut, "pasa": d_ut[1] >= -0.02},
        "duracion": {"relativa": round(dur_x / dur_b - 1, 4), "pasa": abs(dur_x / dur_b - 1) <= 0.05},
    }
    por_tipo = {}
    for tp in sorted({m["tipo"] for m in B}):
        ks = [i for i, m in enumerate(B) if m["tipo"] == tp]
        por_tipo[tp] = {"n": len(ks), "wer_015_base": round(sum(B[i]["wer"] > 0.15 for i in ks) / len(ks), 3),
                        "wer_015_lora": round(sum(X[i]["wer"] > 0.15 for i in ks) / len(ks), 3),
                        "dif_wer": ic([X[i]["wer"] - B[i]["wer"] for i in ks])}
    por_idioma = {}
    for lg in sorted({m["idioma"] for m in B}):
        ks = [i for i, m in enumerate(B) if m["idioma"] == lg]
        por_idioma[lg] = {"n": len(ks), "dif_wer": ic([X[i]["wer"] - B[i]["wer"] for i in ks]),
                          "wer_015_base": round(sum(B[i]["wer"] > 0.15 for i in ks) / len(ks), 3),
                          "wer_015_lora": round(sum(X[i]["wer"] > 0.15 for i in ks) / len(ks), 3)}
    res = {"n": n, "condiciones": cond, "por_tipo": por_tipo, "por_idioma": por_idioma,
           "pasa": all(c["pasa"] for c in cond.values())}
    Path(a.lora, "puerta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[puerta] {a.lora} frente a {a.base}: {n} clips pareados")
    for k, c in cond.items():
        print(f"  {k:16s} {'OK ' if c['pasa'] else 'NO '} {json.dumps({kk: vv for kk, vv in c.items() if kk != 'pasa'})}")
    for k, v in {**por_tipo, **por_idioma}.items():
        print(f"  {k:10s} n {v['n']:3d} · WER>0,15 {v['wer_015_base']:.3f} -> {v['wer_015_lora']:.3f} · dif WER {v['dif_wer']}")
    print("PUERTA D3 (GPU):", "PASA" if res["pasa"] else "NO PASA")


if __name__ == "__main__":
    main()
