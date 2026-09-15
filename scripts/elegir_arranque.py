#!/usr/bin/env python
"""Elige la semilla del RUIDO DE ARRANQUE de una voz: la que no inventa música y más se parece a la persona.

    pkgs/vibevoice/.venv/bin/python scripts/elegir_arranque.py --voz avril --url http://192.168.2.54:8082 \\
        --token "$VOZ_TOKEN" --referencia trabajo/avril/real.wav --salida trabajo/avril/arranque \\
        --ficha /var/lib/voz/voces-propias/avril.json

POR QUÉ
voz-stream fija el ruido de los primeros fotogramas de cada locución (`ruido_arranque`, bloque MÚSICA INVENTADA
de voz_stream.py) para que el modelo no ponga una sintonía de fondo en textos de intro. En torch fp32 la semilla 7
dio 0/72 clips con música y ECAPA +0,000, pero en OpenVINO (producción) 18/72: la semilla buena depende del MOTOR
y de la voz (15-09-2026, en OpenVINO con Avril: 7 -> 0/12 y ECAPA 0,519; 3 -> 0/12 y 0,565; 11 -> 5/12). Por eso se
mide contra el servidor de producción, voz a voz, y el servicio lo trae apagado por defecto.

QUÉ HACE
  1. Sintetiza con voz-stream las FRASES_INTRO (las que más música disparaban) con varias semillas de síntesis,
     una vez por candidata (`ruido_arranque`: 4-6 candidatas) y otra sin ruido de arranque (`null`, la base).
     Si el WAV ya está en --salida/wav no lo pide: se pueden generar fuera (en un servidor de laboratorio) con
     el mismo nombre, <voz>__<frase>__s<semilla>__r<candidata>.wav (r0 = null).
  2. Mide por clip: música (AudioSet AST, máximo de las etiquetas de música > 0,2), WER con faster-whisper y
     ECAPA contra la persona: el centroide de --referencia (audio real, trozos de 10 s) o, sin referencia, el
     de los clips base de la propia voz (vale para voces sin audio real; avisa).
  3. Elige con el criterio fijado antes de medir: entre las candidatas con CERO clips con música y sin
     catástrofes (ningún clip con WER > 25 %), la de mayor ECAPA media. Si ninguna pasa, no escribe nada.
  4. Con --ficha escribe "ruido_arranque" (y la tabla en "ruido_arranque_banco") en <voz>.json, que es lo que
     voz-stream lee; con --semilla-json, lo mismo en la ficha del banco de dobla (voces/<id>/semilla.json).

Datos biométricos (referencias, WAV y huellas): fuera del repo, solo identidades con consentimiento.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from evaluar_clones import wer_en  # noqa: E402

FRASES_INTRO = {
    "welcome": "Welcome to another episode of the podcast. Today we are going to talk about cloud computing.",
    "show": "Welcome to the show. In this episode we talk about cloud storage and what it costs.",
    "hey": "Hey there, welcome to my podcast about technology and business.",
}
ETQ_MUSICA = ("Music", "Musical instrument", "Background music", "Piano", "Guitar", "Synthesizer",
              "Electronic music", "Pop music", "Ambient music")
UMBRAL_MUSICA = 0.2
WER_CATASTROFE = 25.0


def nombre_wav(voz, frase, semilla, cand):
    return f"{voz}__{frase}__s{semilla}__r{cand}.wav"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz", required=True, help="nombre de la voz en voz-stream")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--url", default="http://192.168.2.54:8082")
    ap.add_argument("--token", default="")
    ap.add_argument("--candidatos", type=int, nargs="+", default=[7, 1, 3, 11, 23])
    ap.add_argument("--semillas", type=int, nargs="+", default=[30, 31, 32, 33])
    ap.add_argument("--referencia", action="append", default=[], help="audio real de la persona (repetible)")
    ap.add_argument("--ficha", help="<voz>.json de voz-stream donde escribir la elegida")
    ap.add_argument("--semilla-json", help="voces/<id>/semilla.json del banco de dobla")
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--solo-sintesis", action="store_true")
    a = ap.parse_args()
    if not 4 <= len(a.candidatos) <= 6 or 0 in a.candidatos:
        raise SystemExit("hacen falta entre 4 y 6 candidatas distintas de 0 (0 es la base, sin ruido de arranque)")

    import soundfile as sf
    sal = Path(a.salida)
    (sal / "wav").mkdir(parents=True, exist_ok=True)
    conds = [0] + list(a.candidatos)

    # ------------------------------------------------------------ síntesis
    def pedir(texto, semilla, cand):
        cuerpo = json.dumps({"texto": texto, "voz": a.voz, "cfg_scale": 3.0, "semilla": semilla, "pasos": 6,
                             "formato": "wav", "ruido_arranque": cand or None}).encode()
        pet = urllib.request.Request(f"{a.url}/tts/stream", data=cuerpo, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {a.token}"} if a.token else {})})
        with urllib.request.urlopen(pet, timeout=600) as r:
            return r.read()

    for cand in conds:
        t = time.time()
        n = 0
        for frase, texto in FRASES_INTRO.items():
            for s in a.semillas:
                ruta = sal / "wav" / nombre_wav(a.voz, frase, s, cand)
                if not ruta.exists():
                    ruta.write_bytes(pedir(texto, s, cand))
                    n += 1
        if n:
            print(f"  sintetizado r{cand}: {n} clips en {time.time() - t:.0f} s", flush=True)
    if a.solo_sintesis:
        return

    # ------------------------------------------------------------ jueces
    import librosa
    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
    idx_musica = [i for i, e in ast.config.id2label.items() if e in ETQ_MUSICA]
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"),
                                           run_opts={"device": "cpu"})
    whisper = WhisperModel(a.whisper, device="cpu", compute_type="int8")

    def huella(x16):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x16).float()[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)

    def musica(x16):
        ent = fe(x16[:160000], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(ast(**ent).logits)[0].numpy()
        return float(p[idx_musica].max())

    cache_ruta = sal / "medidas.json"
    medidas = json.load(open(cache_ruta)) if cache_ruta.exists() else {}
    for cand in conds:
        for frase, texto in FRASES_INTRO.items():
            for s in a.semillas:
                nombre = nombre_wav(a.voz, frase, s, cand)
                if nombre in medidas:
                    continue
                x, sr = sf.read(str(sal / "wav" / nombre), dtype="float32")
                x16 = librosa.resample(x, orig_sr=sr, target_sr=16000)
                segs, _ = whisper.transcribe(str(sal / "wav" / nombre), language="en", beam_size=5,
                                             vad_filter=False, condition_on_previous_text=False, temperature=0.0)
                oido = " ".join(g.text.strip() for g in segs).strip()
                medidas[nombre] = {"cand": cand, "frase": frase, "semilla": s, "oido": oido,
                                   "wer": 100 * wer_en(texto, oido), "musica": musica(x16),
                                   "huella": huella(x16).tolist()}
        json.dump(medidas, open(cache_ruta, "w"), ensure_ascii=False)
        print(f"  puntuado r{cand}", flush=True)

    if a.referencia:
        trozos = []
        for r in a.referencia:
            x, _ = librosa.load(r, sr=16000, mono=True)
            trozos += [huella(x[i:i + 160000]) for i in range(0, max(len(x) - 80000, 1), 160000)]
        centro = np.mean(trozos, 0)
        origen = f"{len(trozos)} trozos de audio real"
    else:
        centro = np.mean([v["huella"] for v in medidas.values() if v["cand"] == 0], 0)
        origen = "centroide de los clips base (sin audio real: mide parecido a la voz de siempre, no a la persona)"
        print(f"[aviso] ECAPA contra el {origen}", flush=True)
    centro /= np.linalg.norm(centro)

    # ------------------------------------------------------------ tabla y elección
    filas = []
    for cand in conds:
        m = [v for v in medidas.values() if v["cand"] == cand]
        filas.append({"ruido_arranque": cand or None, "clips": len(m),
                      "musica": int(sum(v["musica"] > UMBRAL_MUSICA for v in m)),
                      "ecapa": float(np.mean([np.dot(v["huella"], centro) for v in m])),
                      "wer": float(np.mean([v["wer"] for v in m])), "wer_max": float(max(v["wer"] for v in m))})
    validas = [f for f in filas if f["ruido_arranque"] and f["musica"] == 0 and f["wer_max"] <= WER_CATASTROFE]
    elegida = max(validas, key=lambda f: f["ecapa"]) if validas else None
    banco = {"fecha": time.strftime("%Y-%m-%d"), "criterio": "max ECAPA con 0 clips con musica AST > 0,2 y WER max <= 25 %",
             "frases": list(FRASES_INTRO), "semillas": a.semillas, "identidad": origen, "tabla": filas}
    json.dump({"elegida": elegida, **banco}, open(sal / "resumen.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n{'ruido_arranque':>14s} {'musica':>7s} {'ecapa':>6s} {'wer':>6s} {'wer_max':>7s}")
    for f in filas:
        marca = " <- elegida" if f is elegida else ("" if f in validas or not f["ruido_arranque"] else "  (descartada)")
        print(f"{str(f['ruido_arranque']):>14s} {f['musica']:3d}/{f['clips']:<3d} {f['ecapa']:6.3f} {f['wer']:6.1f} "
              f"{f['wer_max']:7.1f}{marca}")
    if elegida is None:
        raise SystemExit("ninguna candidata pasa: no se escribe nada (prueba otras candidatas o más semillas)")
    for ruta in (a.ficha, a.semilla_json):
        if not ruta:
            continue
        p = Path(ruta)
        datos = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        datos["ruido_arranque"] = elegida["ruido_arranque"]
        datos["ruido_arranque_banco"] = banco
        p.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"escrito ruido_arranque={elegida['ruido_arranque']} en {p}", flush=True)


if __name__ == "__main__":
    main()
