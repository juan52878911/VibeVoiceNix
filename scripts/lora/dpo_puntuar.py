#!/usr/bin/env python3
"""DPO, paso 2 (plan de preferencias, 24-09): jueces -> recompensa normalizada por grupo -> pares con margen.

  python3 juez_lote.py d0/lote.json d0/medidas.json --identidades d0/identidades --whisper medium --rapido
  python3 dpo_puntuar.py d0/                 # lee lote.*.json, medidas.json y hablantes.json

Grupo = misma voz y mismo texto, semillas distintas. Por muestra:
  wer     wer_norm de whisper MEDIUM (la puerta usa large-v3: jueces distintos para que no se engane al juez)
  ecapa   contra los audios REALES del lector;  utmos
  dur     duracion con los silencios de los bordes recortados, frente a la esperada: silabas del texto a la
          velocidad del propio lector (silabas por segundo de sus clips reales)
  catastrofe: WER > 0,5, corte (duracion < 0,6 de la esperada) o repeticion (> 1,6, o el tope de fotogramas)
Recompensa: -1,0 wer + 1,0 ecapa + 0,5 utmos - penalizacion de duracion (lo que sale de [0,8; 1,25]),
restada la media del grupo y dividida por su desviacion. Par (ganador, perdedor) con recompensa mayor y
margen: WER >= 0,05 mejor (sin perder ECAPA >= 0,03), ECAPA >= 0,03 mejor (sin perder WER > 0,02) o el
perdedor es una catastrofe y el ganador no.

Deja pares.jsonl, muestras.json y resumen.json (las medidas de D0: grupos con par util, dispersion del WER
dentro del grupo, tasa de tramos con WER > 0,15 y de catastrofes; IC 95 % por bootstrap sobre grupos).
"""
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import soundfile as sf

VOCALES = re.compile(r"[aeiouyáéíóúàèìòùâêîôûäëïöüãõœ]+", re.I)
PESOS = {"wer": -1.0, "ecapa": 1.0, "utmos": 0.5, "dur": -1.0}
M_WER, M_ECAPA = 0.05, 0.03


def silabas(t):
    return max(1, len(VOCALES.findall(t)))


def dur_util(ruta):
    """Duracion sin los silencios del principio y del final (umbral: 35 dB bajo el pico)."""
    import librosa
    x, hz = sf.read(str(ruta), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    y, _ = librosa.effects.trim(x, top_db=35)
    return len(y) / hz


def ic(v, n=4000):
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(n))
    return round(sum(v) / len(v), 4), round(b[int(0.025 * n)], 4), round(b[int(0.975 * n)], 4)


def catastrofe(m):
    return m["wer"] > 0.5 or m["razon"] < 0.6 or m["razon"] > 1.6 or m["tope"]


def util(w, l):
    """Motivo por el que (w, l) es un par con margen, o None."""
    if catastrofe(w):
        return None
    if catastrofe(l):
        return "catastrofe"
    ew, el = w.get("ecapa"), l.get("ecapa")
    if l["wer"] - w["wer"] >= M_WER and (ew is None or el is None or ew >= el - M_ECAPA):
        return "wer"
    if ew is not None and el is not None and ew - el >= M_ECAPA and w["wer"] <= l["wer"] + 0.02:
        return "ecapa"
    return None


def main():
    carpeta = Path(sys.argv[1])
    lote = [c for f in sorted(carpeta.glob("lote.*.json")) for c in json.loads(f.read_text())]
    medidas = json.loads((carpeta / "medidas.json").read_text())
    hablantes = json.loads((carpeta / "hablantes.json").read_text())
    velocidad = {}
    for ident, h in hablantes.items():
        sil = sum(silabas(r["texto"]) for r in h["reales"])
        velocidad[ident] = sil / sum(dur_util(carpeta / r["audio"]) for r in h["reales"])
    import torch
    grupos = defaultdict(list)
    for c in lote:
        m = medidas.get(c["clave"])
        if not m or m.get("wer_norm", m.get("wer")) is None:
            continue
        esperada = silabas(c["texto"]) / velocidad[c["identidad"]]
        lat = torch.load(carpeta / "lat" / f"{c['clave']}.pt", map_location="cpu")
        mu = {"clave": c["clave"], "grupo": c["grupo"], "tipo": c["tipo"], "idioma": c["idioma"],
              "identidad": c["identidad"], "texto": c["texto"], "oido": m.get("oido"),
              "wer": m.get("wer_norm", m["wer"]), "ecapa": m.get("ecapa"), "utmos": m["utmos"],
              "dur": round(dur_util(c["audio"]), 3), "T": int(lat["lat"].shape[0]), "tope": bool(lat["tope"])}
        mu["razon"] = round(mu["dur"] / esperada, 3)
        mu["pen_dur"] = round(max(0.0, 0.8 - mu["razon"]) + max(0.0, mu["razon"] - 1.25), 3)
        mu["catastrofe"] = catastrofe(mu)
        mu["r"] = (PESOS["wer"] * mu["wer"] + PESOS["ecapa"] * (mu["ecapa"] or 0.0)
                   + PESOS["utmos"] * mu["utmos"] + PESOS["dur"] * mu["pen_dur"])
        grupos[c["grupo"]].append(mu)
    pares, muestras = [], []
    por_grupo = {}
    for g, ms in grupos.items():
        rs = [m["r"] for m in ms]
        med, dev = st.mean(rs), (st.pstdev(rs) if len(rs) > 1 else 0.0)
        for m in ms:
            m["z"] = round((m["r"] - med) / dev, 3) if dev > 1e-6 else 0.0
            m["r"] = round(m["r"], 4)
        muestras += ms
        motivos = []
        for w, l in permutations(ms, 2):
            if w["z"] <= l["z"]:
                continue
            mot = util(w, l)
            if mot:
                motivos.append(mot)
                pares.append({"grupo": g, "ganador": w["clave"], "perdedor": l["clave"], "motivo": mot,
                              "margen_z": round(w["z"] - l["z"], 3), "d_wer": round(l["wer"] - w["wer"], 4),
                              "d_ecapa": round((w["ecapa"] or 0) - (l["ecapa"] or 0), 4),
                              "d_utmos": round(w["utmos"] - l["utmos"], 3),
                              "identidad": w["identidad"], "idioma": w["idioma"], "tipo": w["tipo"]})
        wers = [m["wer"] for m in ms]
        ecs = [m["ecapa"] for m in ms if m["ecapa"] is not None]
        por_grupo[g] = {"n": len(ms), "util": bool(motivos), "motivos": sorted(set(motivos)),
                        "wer_rango": max(wers) - min(wers), "wer_std": st.pstdev(wers),
                        "ecapa_rango": (max(ecs) - min(ecs)) if ecs else None,
                        "utmos_rango": max(m["utmos"] for m in ms) - min(m["utmos"] for m in ms),
                        "tipo": ms[0]["tipo"], "idioma": ms[0]["idioma"], "wer_iguales": len(set(wers)) == 1}
    with open(carpeta / "pares.jsonl", "w") as f:
        for p in pares:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (carpeta / "muestras.json").write_text(json.dumps(muestras, ensure_ascii=False, indent=1))

    def resumen(filtro):
        gs = [v for v in por_grupo.values() if filtro(v)]
        ms = [m for m in muestras if filtro(por_grupo[m["grupo"]])]
        if not gs:
            return None
        r = {"grupos": len(gs), "muestras": len(ms),
             "grupos_con_par_util": ic([float(v["util"]) for v in gs]),
             "wer_rango_en_grupo": ic([v["wer_rango"] for v in gs]),
             "wer_std_en_grupo": ic([v["wer_std"] for v in gs]),
             "grupos_con_wer_identico": round(sum(v["wer_iguales"] for v in gs) / len(gs), 3),
             "tramos_wer_015": ic([float(m["wer"] > 0.15) for m in ms]),
             "catastrofes": ic([float(m["catastrofe"]) for m in ms]),
             "wer_medio": ic([m["wer"] for m in ms]),
             "razon_dur_media": round(st.mean(m["razon"] for m in ms), 3)}
        ers = [v["ecapa_rango"] for v in gs if v["ecapa_rango"] is not None]
        if ers:
            r["ecapa_rango_en_grupo"] = ic(ers)
        r["utmos_rango_en_grupo"] = ic([v["utmos_rango"] for v in gs])
        for mot in ("wer", "ecapa", "catastrofe"):
            r[f"grupos_con_motivo_{mot}"] = round(sum(mot in v["motivos"] for v in gs) / len(gs), 3)
        return r
    res = {"todo": resumen(lambda v: True), "pares": len(pares),
           "pares_por_motivo": {k: sum(p["motivo"] == k for p in pares) for k in ("wer", "ecapa", "catastrofe")}}
    for tp in sorted({v["tipo"] for v in por_grupo.values()}):
        res[f"tipo:{tp}"] = resumen(lambda v, tp=tp: v["tipo"] == tp)
    for lg in sorted({v["idioma"] for v in por_grupo.values()}):
        res[f"idioma:{lg}"] = resumen(lambda v, lg=lg: v["idioma"] == lg)
    frac = res["todo"]["grupos_con_par_util"]
    res["puerta_d0"] = {"condicion": ">= 30 % de grupos con par util", "valor": frac[0], "ic": frac[1:],
                        "pasa": frac[0] >= 0.30}
    (carpeta / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    t = res["todo"]
    print(f"[dpo] {t['grupos']} grupos, {t['muestras']} muestras, {len(pares)} pares {res['pares_por_motivo']}")
    for k in ("grupos_con_par_util", "wer_rango_en_grupo", "wer_std_en_grupo", "tramos_wer_015", "catastrofes",
              "wer_medio", "ecapa_rango_en_grupo", "utmos_rango_en_grupo"):
        print(f"  {k:24s} {t.get(k)}")
    print(f"  grupos con WER identico  {t['grupos_con_wer_identico']} · razon de duracion media {t['razon_dur_media']}")
    for k, v in res.items():
        if ":" in k and v:
            print(f"  {k:12s} n {v['grupos']:3d} · par util {v['grupos_con_par_util'][0]:.2f} · "
                  f"WER>0,15 {v['tramos_wer_015'][0]:.2f} · rango WER {v['wer_rango_en_grupo'][0]:.3f}")
    print(f"PUERTA D0: {'PASA' if res['puerta_d0']['pasa'] else 'NO PASA'} ({frac[0]:.2f}, IC {frac[1]:.2f}-{frac[2]:.2f})")


if __name__ == "__main__":
    main()
