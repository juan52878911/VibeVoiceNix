#!/usr/bin/env python3
"""Datos para el LoRA: audio a 24 kHz con licencia comercial, por idioma y hablante, ya codificado en
latentes y emparejado (referencia de voz + objetivo del MISMO hablante, enunciados distintos).

Fuentes (CC-BY 4.0; hay que atribuirlas si el modelo se distribuye):
  CML-TTS    (ylacombe/cml-tts)     es fr de it pt nl pl, 24 kHz, lectura de LibriVox
  LibriTTS-R (mythicinfinity/libritts_r) en, 24 kHz, restaurado
NO se usan MLS ni VoxPopuli: van a 16 kHz y el modelo aprenderia a generar audio sin agudos.

  python3 datos.py --salida datos/ --idiomas es,en,fr,de,it,pt --hablantes 80 --por-hablante 16

Deja datos/<idioma>.pt (lista de ejemplos) y datos/evaluacion.json (hablantes y frases apartados para
evaluar.py: nunca entran en el entrenamiento).
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402

CML = {"es": "spanish", "fr": "french", "de": "german", "it": "italian", "pt": "portuguese",
       "nl": "dutch", "pl": "polish"}
MIN_S, MAX_S = 3.0, 14.0


def flujo(idioma, lev_min=0.95, splits=("train",)):
    """Filas (decodificar(), texto, hablante, duracion o None). El audio se decodifica SOLO si se va a usar
    (los audiolibros vienen agrupados por lector y hay que leer muchas filas), con soundfile: datasets 4-5
    pide torchcodec para decodificar, y no hace falta. En CML-TTS se descartan las filas cuyo texto no
    coincide con lo que oye su reconocedor: el campo "levenshtein" es una SIMILITUD 0-1 (mediana 0,978
    medida en 300 filas de espanol, 23-09) y se exige >= 0,95."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset
    for split in splits:
        if idioma == "en":
            ds = load_dataset("mythicinfinity/libritts_r", "clean", split="train.clean.360", streaming=True)
            campo = "text_normalized"
        else:
            ds = load_dataset("ylacombe/cml-tts", CML[idioma], split=split, streaming=True)
            campo = "text"
        ds = ds.cast_column("audio", Audio(decode=False))
        for r in ds:
            if r.get("levenshtein") is not None and r["levenshtein"] < lev_min:
                continue
            crudo = r["audio"]["bytes"]

            def decodificar(crudo=crudo):
                x, hz = sf.read(io.BytesIO(crudo), dtype="float32")
                return {"array": x.mean(1) if x.ndim > 1 else x, "sampling_rate": hz}
            yield decodificar, r[campo], str(r["speaker_id"]), r.get("duration")


def a24(audio):
    x, hz = np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]
    if hz != 24000:
        import librosa
        x = librosa.resample(x, orig_sr=hz, target_sr=24000)
    return x / max(1e-4, float(np.abs(x).max())) * 0.7          # nivel comun, como igualar_volumen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--idiomas", default="es,en,fr,de,it,pt")
    ap.add_argument("--hablantes", type=int, default=80)
    ap.add_argument("--por-hablante", type=int, default=16)
    ap.add_argument("--apartar", type=int, default=6, help="hablantes por idioma apartados para evaluar")
    ap.add_argument("--leer-max", type=int, default=40000, help="filas del flujo como mucho por idioma")
    # el portugues de CML-TTS es pequeno y distinto (23-09): clips de 10,9-21 s (mediana 13,6), similitud
    # mediana 0,94 y un lector dominante; con 14 s y 0,95 quedaban 8 lectores. Ahi: --max-s 21 --lev-min 0,9
    # --splits train,dev,test --conservar-eval
    ap.add_argument("--max-s", type=float, default=MAX_S)
    ap.add_argument("--lev-min", type=float, default=0.95)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--conservar-eval", action="store_true",
                    help="no rehacer los hablantes apartados: se mantienen los de evaluacion.json y se excluyen")
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    d = "cuda" if torch.cuda.is_available() else "cpu"
    proc, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    modelo.eval()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    evaluacion = json.loads((sal / "evaluacion.json").read_text()) if (sal / "evaluacion.json").exists() else {}
    for idioma in a.idiomas.split(","):
        if (sal / f"{idioma}.pt").exists():
            print(f"[datos] {idioma}: ya esta", flush=True)
            continue
        por_h = defaultdict(list)
        leidas = 0
        fijos = {e["hablante"] for e in evaluacion.get(idioma, [])} if a.conservar_eval else set()
        for decodificar, texto, h, dur in flujo(idioma, a.lev_min, tuple(a.splits.split(","))):
            leidas += 1
            if leidas > a.leer_max:
                break
            if h in fijos or len(por_h.get(h, ())) >= a.por_hablante + 2 or not texto or len(texto) < 10:
                continue
            if dur is not None and not (MIN_S <= dur <= a.max_s):
                continue
            try:
                audio = decodificar()
            except Exception:
                continue
            dur = len(audio["array"]) / audio["sampling_rate"]
            if not (MIN_S <= dur <= a.max_s):
                continue
            # float16 y tope de lectores a medias: con clips de hasta 21 s (pt) el proceso llego a 9,5 GB de RAM
            # y lo mato el kernel (23-09)
            if h not in por_h and len(por_h) >= 4 * (a.hablantes + a.apartar):
                continue
            por_h[h].append((a24(audio).astype(np.float16), texto.strip()))
            # para cuando hay bastantes lectores COMPLETOS (antes exigia que todos los que tenian >= 4 llegaran
            # al cupo: con un solo lector de 10 frases en el corpus no paraba nunca y leia --leer-max filas)
            completos = sum(len(v) >= a.por_hablante for v in por_h.values())
            if completos >= a.hablantes + (0 if fijos else a.apartar):
                break
            if leidas % 2000 == 0:
                print(f"[datos] {idioma}: {leidas} filas, {completos} lectores completos", flush=True)
        n_ap = 0 if fijos else a.apartar
        hs = sorted((h for h, v in por_h.items() if len(v) >= 4), key=lambda h: -len(por_h[h]))[: a.hablantes + n_ap]
        random.Random(0).shuffle(hs)
        apartados, usados = hs[:n_ap], hs[n_ap: n_ap + a.hablantes]
        # evaluacion: por hablante apartado, 1 referencia y hasta 3 frases (audio real para la identidad)
        ev = []
        (sal / "eval_audio" / idioma).mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        for h in apartados:
            clips = por_h[h]
            ref = sal / "eval_audio" / idioma / f"{h}_ref.wav"
            sf.write(str(ref), clips[0][0].astype(np.float32), 24000)
            reales = []
            for k, (x, t) in enumerate(clips[1:4], 1):
                w = sal / "eval_audio" / idioma / f"{h}_{k}.wav"
                sf.write(str(w), x.astype(np.float32), 24000)
                reales.append({"audio": str(w), "texto": t})
            ev.append({"hablante": h, "ref": str(ref), "ref_txt": clips[0][1], "frases": reales})
        if not fijos:
            evaluacion[idioma] = ev
        # entrenamiento: cada enunciado es objetivo una vez, con OTRO del mismo hablante como referencia
        ejemplos = []
        for h in usados:
            clips = por_h[h]
            lats = [FZ.latentes(modelo, x.astype(np.float32), d, muestrear=True).half().cpu() for x, _ in clips]
            for i, (x, t) in enumerate(clips):
                j = (i + 1 + random.Random(i).randrange(len(clips) - 1)) % len(clips)
                ejemplos.append({"idioma": idioma, "hablante": h, "ref_lat": lats[j].float(),
                                 "ref_txt": clips[j][1] + "\n", "lat": lats[i].float(), "txt": t})
        torch.save(ejemplos, sal / f"{idioma}.pt")
        # se relee justo antes de escribir: con varios procesos en paralelo (uno por pareja de idiomas)
        # cada uno pisaba lo que los otros habian escrito despues de su arranque (23-09: se perdieron es y fr)
        disco = json.loads((sal / "evaluacion.json").read_text()) if (sal / "evaluacion.json").exists() else {}
        if idioma in evaluacion:
            disco[idioma] = evaluacion[idioma]
        (sal / "evaluacion.json").write_text(json.dumps(disco, ensure_ascii=False, indent=1))
        horas = sum(e["lat"].shape[0] for e in ejemplos) / 7.5 / 3600
        print(f"[datos] {idioma}: {leidas} filas leidas · {len(usados)} hablantes · {len(ejemplos)} ejemplos "
              f"({horas:.2f} h) · {len(apartados)} hablantes apartados para evaluar", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)      # datasets 5 revienta al cerrar el interprete (PyGILState_Release): los datos ya estan escritos
