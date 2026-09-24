#!/usr/bin/env python3
"""Datos de UNA persona para el LoRA por voz (F8) desde un video ANOTADO A MANO (editor de dobla).

La F8 con transcripciones de whisper (datos_voz.py) subio la identidad pero tambien el WER (+0,07). Una
explicacion posible es que el LoRA aprenda de textos mal transcritos. Aqui el texto es el humano de la
anotacion, y el audio la pista de voz separada (voces24k.wav) cortada con sus tiempos. Los segmentos que
se solapan con los de evaluacion (--excluir: el corpus del banco de reconstruccion) no entran nunca.

  python3 datos_anotados.py --anotacion videoplayback.json --voz voces24k.wav --hablante 0 \\
      --excluir corpus.json --identidad carlos-segura --salida anotado_carlos/ --nombre carlos
Deja <salida>/voz.pt (formato de datos.py: referencia = otro segmento suyo).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402

MIN_S, MAX_S = 3.0, 14.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotacion", required=True)
    ap.add_argument("--voz", required=True)
    ap.add_argument("--hablante", type=int, required=True)
    ap.add_argument("--excluir", required=True, help="corpus.json del banco de reconstruccion")
    ap.add_argument("--identidad", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", default="voz")
    ap.add_argument("--idioma", default="es")
    ap.add_argument("--margen", type=float, default=1.0, help="segundos de separacion con lo evaluado")
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    d = "cuda" if torch.cuda.is_available() else "cpu"
    segs = [s for s in json.loads(Path(a.anotacion).read_text())["segmentos"] if s.get("hablante") == a.hablante]
    corpus = json.loads(Path(a.excluir).read_text())
    ev = [x for x in corpus if x["identidad"] == a.identidad][0]["segmentos"]
    prohibido = [(s["ini"] - a.margen, s["fin"] + a.margen) for s in ev]
    x, hz = sf.read(a.voz, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    utiles, fuera = [], 0
    for s in segs:
        if any(s["ini"] < b and s["fin"] > a0 for a0, b in prohibido):
            fuera += 1
            continue
        dur = s["fin"] - s["ini"]
        if not (MIN_S <= dur <= MAX_S) or len(s["texto"].strip()) < 10:
            continue
        t = x[int(s["ini"] * hz):int(s["fin"] * hz)]
        if hz != 24000:
            import librosa
            t = librosa.resample(t, orig_sr=hz, target_sr=24000)
        utiles.append((t / max(1e-4, float(np.abs(t).max())) * 0.7, s["texto"].strip(), s["ini"]))
    print(f"[anotado] {len(segs)} segmentos del hablante {a.hablante}; {fuera} fuera por solapar con la evaluacion; "
          f"{len(utiles)} utiles ({sum(len(u[0]) for u in utiles) / 24000 / 60:.1f} min)", flush=True)
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    _, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    modelo.eval()
    lats = [FZ.latentes(modelo, u[0], d, muestrear=True).half().cpu() for u in utiles]
    ejemplos = []
    for i, u in enumerate(utiles):
        j = (i + 1 + random.Random(i).randrange(len(utiles) - 1)) % len(utiles)
        # "nota" = minuto del video: la validacion de entrenar.py aparta por hablante (aqui, por minuto)
        ejemplos.append({"idioma": a.idioma, "hablante": f"{a.nombre}-n{int(u[2] // 60)}", "ref_lat": lats[j].float(),
                         "ref_txt": utiles[j][1] + "\n", "lat": lats[i].float(), "txt": u[1]})
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    torch.save(ejemplos, Path(a.salida) / "voz.pt")
    print(f"[anotado] {len(ejemplos)} ejemplos en {a.salida}/voz.pt", flush=True)


if __name__ == "__main__":
    main()
