#!/usr/bin/env python3
"""Datos de UNA persona para el LoRA de personalidad (F8), desde una carpeta de notas de voz.

Solo audio con permiso de la persona (Juan, 23-09-2026: notas de WhatsApp suyas). Sale de aqui:
  <salida>/datos/voz.pt           ejemplos de entrenamiento en el formato de datos.py (referencia = otro
                                  enunciado suyo, objetivo = lo que dice), para entrenar.py
  <salida>/datos/evaluacion.json  vacio (evaluar.py lo pide) y
  <salida>/voces.json             la evaluacion: clonado desde tramos de ENTRENAMIENTO y frases APARTADAS
                                  (su texto y su audio real, para medir identidad, ritmo y pausas)
  <salida>/tramos/*.wav           los tramos, 24 kHz

Pasos: cada nota a 24 kHz mono -> whisper large-v3 con marcas por palabra -> enunciados de 3-14 s
cortados en pausas -> se quitan los que no son la persona (ECAPA frente a su centroide) -> apartado
por NOTA (no por tramo: dos tramos de la misma nota se parecen demasiado).

  python3 datos_voz.py --notas ~/notas_whatsapp/ --salida voz_juan/ --idioma es [--apartar 0.15]
Los datos son biometricos: fuera del repo, y la carpeta se borra al terminar el experimento.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402

EXT = {".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".mp4", ".webm", ".flac"}
MIN_S, MAX_S, PAUSA_S = 3.0, 14.0, 0.45


def leer(ruta, hz=24000):
    with tempfile.NamedTemporaryFile(suffix=".wav") as t:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ruta), "-ac", "1", "-ar", str(hz), t.name],
                       check=True)
        x, _ = sf.read(t.name, dtype="float32")
    return x


def enunciados(palabras, total_s):
    """Palabras [(ini, fin, texto)] -> tramos [(ini, fin, texto)] de MIN_S-MAX_S cortados en pausas."""
    tramos, actual = [], []
    for p in palabras:
        if actual and (p[0] - actual[-1][1] > PAUSA_S or p[1] - actual[0][0] > MAX_S):
            tramos.append(actual)
            actual = []
        actual.append(p)
    if actual:
        tramos.append(actual)
    salida = []
    for t in tramos:
        ini, fin = max(0.0, t[0][0] - 0.15), min(total_s, t[-1][1] + 0.2)
        if MIN_S <= fin - ini <= MAX_S + 0.5:
            salida.append((ini, fin, "".join(w[2] for w in t).strip()))
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--idioma", default="es")
    ap.add_argument("--nombre", default="voz")
    ap.add_argument("--apartar", type=float, default=0.15, help="fraccion de NOTAS apartadas para evaluar")
    ap.add_argument("--margen", type=float, default=0.15, help="fuera si ECAPA < mediana - margen (otra voz)")
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sal = Path(a.salida)
    (sal / "tramos").mkdir(parents=True, exist_ok=True)
    (sal / "datos").mkdir(exist_ok=True)
    notas = sorted(p for p in Path(a.notas).expanduser().rglob("*") if p.suffix.lower() in EXT)
    print(f"[voz] {len(notas)} notas", flush=True)
    import juez_lote as JL
    j = JL.Jueces(d, 2)
    tramos = []                          # (nota, ruta_wav, texto, huella)
    total_audio = 0.0
    for k, nota in enumerate(notas):
        try:
            x = leer(nota)
        except subprocess.CalledProcessError:
            print(f"[voz] no se pudo leer {nota.name}", flush=True)
            continue
        total_audio += len(x) / 24000
        x16 = JL.a16(x, 24000)
        segs, _ = j.whisper.transcribe(x16, language=a.idioma, beam_size=5, word_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=True)
        palabras = [(w.start, w.end, w.word) for s in segs for w in (s.words or [])]
        for m, (ini, fin, texto) in enumerate(enunciados(palabras, len(x) / 24000)):
            if len(texto) < 10:
                continue
            trozo = x[int(ini * 24000):int(fin * 24000)]
            trozo = trozo / max(1e-4, float(np.abs(trozo).max())) * 0.7          # nivel comun, como datos.py
            ruta = sal / "tramos" / f"n{k:04d}_{m:02d}.wav"
            sf.write(str(ruta), trozo, 24000, subtype="PCM_16")
            tramos.append((k, ruta, texto, j.huella(JL.a16(trozo, 24000))))
    if not tramos:
        raise SystemExit("ningun tramo util")
    centro = np.mean([t[3] for t in tramos], 0)
    centro /= np.linalg.norm(centro)
    sims = np.array([float(t[3] @ centro) for t in tramos])
    umbral = float(np.median(sims)) - a.margen
    buenos = [t for t, s in zip(tramos, sims) if s >= umbral]
    print(f"[voz] {total_audio / 60:.1f} min de notas -> {len(tramos)} tramos; {len(buenos)} de la persona "
          f"(ECAPA >= {umbral:.3f}, mediana {np.median(sims):.3f})", flush=True)
    ids = sorted({t[0] for t in buenos})
    random.Random(0).shuffle(ids)
    n_ap = max(1, int(round(len(ids) * a.apartar)))
    apartadas = set(ids[:n_ap])
    ent = [t for t in buenos if t[0] not in apartadas]
    ev = [t for t in buenos if t[0] in apartadas]
    # referencia de clonado para la evaluacion: los tramos de entrenamiento mas tipicos hasta ~30 s (como dobla)
    tipicos = sorted(ent, key=lambda t: -float(t[3] @ centro))
    refs, acum = [], 0.0
    for t in tipicos:
        if acum >= 30:
            break
        refs.append({"audio": str(t[1]), "transcripcion": t[2]})
        acum += sf.info(str(t[1])).duration
    (sal / "voces.json").write_text(json.dumps({a.nombre: {
        "refs": refs, "reales": [str(t[1]) for t in ev],
        "frases": {a.idioma: [t[2] for t in ev]}}}, ensure_ascii=False, indent=1))
    (sal / "datos" / "evaluacion.json").write_text("{}")
    # ejemplos de entrenamiento: cada tramo es objetivo una vez, con otro tramo suyo como referencia
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    _, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    modelo.eval()
    lats = [FZ.latentes(modelo, sf.read(str(t[1]), dtype="float32")[0], d, muestrear=True).half().cpu() for t in ent]
    ejemplos = []
    for i, t in enumerate(ent):
        jj = (i + 1 + random.Random(i).randrange(len(ent) - 1)) % len(ent)
        ejemplos.append({"idioma": a.idioma, "hablante": f"{a.nombre}-n{t[0]}", "ref_lat": lats[jj].float(),
                         "ref_txt": ent[jj][2] + "\n", "lat": lats[i].float(), "txt": t[2]})
    torch.save(ejemplos, sal / "datos" / "voz.pt")
    horas = sum(e["lat"].shape[0] for e in ejemplos) / 7.5 / 60
    print(f"[voz] {len(ejemplos)} ejemplos de entrenamiento ({horas:.1f} min) de {len(ids) - n_ap} notas; "
          f"{len(ev)} frases apartadas de {n_ap} notas; referencia de clonado {acum:.0f} s", flush=True)


if __name__ == "__main__":
    main()
