#!/usr/bin/env python3
"""Evolucion de dobla: el MISMO video doblado en fechas distintas, juzgado con el mismo juez.

S3 tiene versionado: cada doblaje de un video deja su mp4 y su manifiesto. Por cada version y cada
tramo del manifiesto (su texto en el idioma de destino, su ventana y su hablante):
  wer        whisper large-v3 sobre el tramo doblado frente al texto que se queria decir (normalizados)
  identidad  ECAPA del tramo doblado frente a la voz ORIGINAL en la misma ventana (voces24k del video):
             no depende de las etiquetas de hablante, que cambian entre versiones
  utmos      naturalidad del tramo doblado
y del manifiesto, los minutos del job.

  python3 juzgar_historia.py lista.json salida/   (lista: {"original_voces": url, "versiones": [{nombre, fecha, mp4, manifiesto}]})
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))


def bajar(url, destino):
    if not Path(destino).exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def pista(ruta, hz=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav") as t:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ruta), "-ac", "1", "-ar", str(hz), t.name],
                       check=True)
        return sf.read(t.name, dtype="float32")[0]


def segmentos(m):
    s = m.get("segmentos") or []
    return list(s.values()) if isinstance(s, dict) else list(s)


def main():
    lista = json.loads(Path(sys.argv[1]).read_text())
    sal = Path(sys.argv[2])
    (sal / "bajado").mkdir(parents=True, exist_ok=True)
    import juez_lote as JL
    from normalizar_texto import normalizar
    j = JL.Jueces("cuda", 2)
    orig = pista(bajar(lista["original_voces"], sal / "bajado" / "original_voces.wav"))
    filas = []
    for v in lista["versiones"]:
        mp4 = bajar(v["mp4"], sal / "bajado" / f"{v['nombre']}.mp4")
        m = json.loads(Path(bajar(v["manifiesto"], sal / "bajado" / f"{v['nombre']}.json")).read_text())
        destino = (m.get("args") or {}).get("destino", "en")
        x = pista(mp4)
        wers, ids, uts, qc = [], [], [], []
        for s in segmentos(m):
            texto = (s.get("texto") or "").strip()
            if len(texto) < 15 or "ini" not in s:
                continue
            ini, fin = max(0.0, s["ini"] - 0.3), s["fin"] + 0.5
            seg = x[int(ini * 16000):int(fin * 16000)]
            if len(seg) < 16000:
                continue
            segs, _ = j.whisper.transcribe(seg, language=destino, beam_size=5, condition_on_previous_text=False)
            oido = " ".join(t.text for t in segs).strip()
            wers.append(JL.wer(normalizar(texto, destino), normalizar(oido, destino)))
            o = orig[int(s["ini"] * 16000):int(s["fin"] * 16000)]
            if len(o) > 32000 and len(seg) > 32000:
                ids.append(float(j.huella(seg) @ j.huella(o)))
            import torch
            with torch.inference_mode():
                uts.append(float(j.utmos(torch.from_numpy(seg)[None].to("cuda"), 16000).item()))
            if (s.get("qc") or {}).get("wer") is not None:
                qc.append(s["qc"]["wer"])
        t = m.get("tiempos") or {}
        fila = {"version": v["nombre"], "fecha": v["fecha"][:16], "destino": destino,
                "acento": (m.get("args") or {}).get("acento", "propio"), "tramos": len(wers),
                "wer_medio": round(float(np.mean(wers)), 3) if wers else None,
                "wer_mediana": round(float(np.median(wers)), 3) if wers else None,
                "identidad": round(float(np.mean(ids)), 3) if ids else None,
                "utmos": round(float(np.mean(uts)), 3) if uts else None,
                "qc_wer": round(float(np.mean(qc)), 3) if qc else None,
                "minutos_job": round(sum(v2 for v2 in t.values() if isinstance(v2, (int, float))) / 60, 1) if t else None}
        filas.append(fila)
        print(json.dumps(fila, ensure_ascii=False), flush=True)
    (sal / "historia.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
