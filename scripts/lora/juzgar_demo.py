#!/usr/bin/env python3
"""Mide el RESULTADO FINAL de la demo: la charla doblada a cada idioma, tal como la oiria alguien.

Por segmento del manifiesto (su texto traducido y su posicion en el video doblado): WER con whisper large-v3
en el idioma del doblaje (texto y transcripcion normalizados), identidad ECAPA contra la voz real del
hablante en el video original, y el coste y el tiempo del job segun el manifiesto.

  python3 juzgar_demo.py --original charla.mp4 --doblados demo/ [--idiomas en,fr,de,it,pt]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))


def pista(video, hz=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav") as t:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-ac", "1", "-ar", str(hz), t.name],
                       check=True)
        return sf.read(t.name, dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--anotacion", required=True)
    ap.add_argument("--doblados", required=True, help="carpeta con <nombre>.mp4 y <nombre>.mp4.manifiesto.json")
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()
    import juez_lote as JL
    j = JL.Jueces(a.dispositivo, 2)
    anot = json.loads(Path(a.anotacion).read_text())
    orig = pista(a.original)
    centros = {}
    for s in anot["segmentos"]:
        h = s.get("hablante")
        x = orig[int(s["ini"] * 16000):int(s["fin"] * 16000)]
        if len(x) > 32000:
            centros.setdefault(h, []).append(j.huella(x))
    centros = {h: (lambda v: v / np.linalg.norm(v))(np.mean(v, 0)) for h, v in centros.items()}
    filas = []
    for mp4 in sorted(Path(a.doblados).glob("*.mp4")):
        man = Path(str(mp4) + ".manifiesto.json")
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        # el manifiesto guarda los args por defecto (destino en, acento propio) aunque el job corriera con
        # DOBLA_ARGS_EXTRA: idioma y acento salen del nombre (charla_<idioma>[_propio|_nativo].mp4)
        destino = mp4.stem.split("_")[-2] if mp4.stem.split("_")[-1] in ("propio", "nativo") else mp4.stem.split("_")[-1]
        # el manifiesto guarda "propio" aunque el job corriera con --acento nativo (el log lo muestra):
        # la etiqueta sale del nombre del fichero
        acento = "propio" if "propio" in mp4.stem else "nativo"
        x = pista(mp4)
        wers, ids = [], []
        for k, s in m["segmentos"].items():
            # el doblaje se coloca dentro de la ventana del segmento original (sincronia); margen de 0,5 s
            ini, fin = max(0.0, s["ini"] - 0.3), s["fin"] + 0.5
            seg = x[int(ini * 16000):int(fin * 16000)]
            if len(seg) < 8000 or len(s["texto"]) < 15:
                continue
            segs, _ = j.whisper.transcribe(seg, language=destino, beam_size=5, condition_on_previous_text=False)
            oido = " ".join(t.text for t in segs).strip()
            wers.append(JL.wer(s["texto"], oido))
            if s.get("hablante") in centros and len(seg) > 32000:
                ids.append(float(j.huella(seg) @ centros[s["hablante"]]))
        t = m.get("tiempos") or {}
        filas.append({"video": mp4.name, "destino": destino, "acento": acento, "segmentos": len(wers),
                      "wer": round(float(np.mean(wers)), 3) if wers else None,
                      "wer_mediana": round(float(np.median(wers)), 3) if wers else None,
                      "identidad": round(float(np.mean(ids)), 3) if ids else None,
                      "minutos_job": round(sum(v for v in t.values() if isinstance(v, (int, float))) / 60, 1)})
        print(json.dumps(filas[-1], ensure_ascii=False), flush=True)
    Path(a.doblados, "juicio_demo.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
