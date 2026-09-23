#!/usr/bin/env python3
"""Sintetiza un lote de peticiones con voz-stream, midiendo el streaming de verdad.

Corre en la VM voz (el token no sale de ella). Por cada peticion guarda el WAV y
cuanto tardo el PRIMER trozo de audio desde que se mando el texto.

  python3 sintetizar_lote.py peticiones.json salida/
  peticiones.json: [{"clave", "texto", "voz", "semilla", "cfg"?}]
"""
import json
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path

URL = os.environ.get("VOZ_URL", "http://127.0.0.1:8082")
peticiones = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sal = Path(sys.argv[2])
sal.mkdir(parents=True, exist_ok=True)
medidas = {}
for p in peticiones:
    cuerpo = {"texto": p["texto"], "voz": p["voz"], "semilla": p["semilla"], "cfg_scale": p.get("cfg", 3.0)}
    r = urllib.request.Request(f"{URL}/tts/stream", json.dumps(cuerpo).encode(),
                               {"Content-Type": "application/json",
                                "Authorization": "Bearer " + os.environ["VOZ_TOKEN"]})
    t0 = time.monotonic()
    primero, trozos = None, []
    with urllib.request.urlopen(r, timeout=600) as resp:
        cabecera = resp.read(44)
        while True:
            b = resp.read(4800)          # 100 ms de PCM a 24 kHz
            if not b:
                break
            if primero is None:
                primero = time.monotonic() - t0
            trozos.append(b)
    total = time.monotonic() - t0
    pcm = b"".join(trozos)
    with wave.open(str(sal / f"{p['clave']}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    dur = len(pcm) / 2 / 24000
    medidas[p["clave"]] = {**p, "primer_audio_s": round(primero or 0, 3), "total_s": round(total, 2),
                           "dur_s": round(dur, 2), "rtf": round(total / dur, 3) if dur else None}
    print(f"[voz] {p['clave']:28s} primer audio {primero:5.2f} s · {dur:5.2f} s de audio en {total:5.2f} s", flush=True)
(sal / "medidas.json").write_text(json.dumps(medidas, ensure_ascii=False, indent=1))
