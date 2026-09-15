#!/usr/bin/env python
"""Evalúa candidatos de clon de UNA persona contra su audio real APARTADO, en español e inglés, con el motor de producción.

    pkgs/vibevoice/.venv/bin/python scripts/evaluar_clones.py --candidatos trabajo/sebastian/clones \
        --pista voces24k.wav --apartados trabajo/sebastian/apartados.json --salida trabajo/sebastian/eval \
        --url http://192.168.2.54:8082 --token "$VOZ_TOKEN"

POR QUÉ
Elegir la referencia y la semilla de un clon midiendo contra la PROPIA referencia (lo que hace
banco_semillas.py) premia al clon que copia ese audio concreto. Con una charla larga hay minutos de la misma
persona que el clon no ha oído: `mejor_referencia.py` los aparta, y aquí el juez es el centroide ECAPA de esos
apartados. Se sintetiza con voz-stream (OpenVINO, lo que oye dobla), no con torch.

QUÉ HACE
  1. Copia cada candidato (<voz>__<variante>__c<semilla>.pt) a la VM como voz temporal `eval-<stem>` y la
     borra al acabar (no para el servicio: la carpeta de voces se relee sola).
  2. Sintetiza FRASES en español e inglés con una semilla de síntesis fija, sin forma (la identidad no
     depende de las pausas).
  3. Mide por clip: identidad ECAPA contra el centroide de los apartados, WER con faster-whisper large-v3 en
     el idioma de la frase, UTMOS22 y el desvío de tono (semitonos) contra la mediana real.
  4. Elige, con el criterio fijado antes de medir (docs/clonado-de-voz.md, «Voces de la charla larga»):
     máxima identidad media ES+EN entre los candidatos sin catástrofes (ningún clip con WER > 25 %) y con WER
     medio a no más de 3 puntos del mejor en cada idioma.

Datos biométricos: solo para identidades con consentimiento registrado; todo fuera del repo.
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fidelidad as FI  # noqa: E402
import naturalidad as NA  # noqa: E402
import prosodia as PR  # noqa: E402

FRASES = {
    "es": ["Bueno, hoy vamos a hablar de cómo la nube cambia la manera en que una empresa maneja sus datos.",
           "Dentro de Amazon existen tres modelos de servicio, y cada uno resuelve un problema distinto.",
           "Si tengo una aplicación en mi empresa, ¿qué pasa cuando de repente llegan mil usuarios al mismo tiempo?",
           "La idea es que el equipo pueda enfocarse en el negocio y no en mantener servidores."],
    "en": ["Today we are going to talk about how the cloud changes the way a company handles its data.",
           "Inside Amazon there are three service models, and each one solves a different problem.",
           "If I have an application in my company, what happens when a thousand users arrive at the same time?",
           "The idea is that the team can focus on the business instead of maintaining servers."],
}
SEMILLA_SINTESIS = 11


def wer_en(ref, hip):
    norm = lambda t: re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()
    r, h = norm(ref), norm(hip)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidatos", required=True)
    ap.add_argument("--pista", required=True)
    ap.add_argument("--apartados", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--url", default="http://192.168.2.54:8082")
    ap.add_argument("--token", required=True)
    ap.add_argument("--vm", default="root@192.168.2.54")
    ap.add_argument("--clave", default=str(Path.home() / ".ssh/oracle_a1"))
    ap.add_argument("--voces-vm", default="/run/voz-stream/voces")
    ap.add_argument("--max-apartados", type=int, default=80)
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--ruido-arranque", type=int, default=None,
                    help="ruido_arranque de la sintesis (0 = apagado); sin darlo, el de voz-stream. La semilla por "
                         "voz la elige scripts/elegir_arranque.py una vez escogido el clon")
    ap.add_argument("--solo-sintesis", action="store_true",
                    help="genera los WAV en la VM y sale: la puntuación va después, sin ocupar voz-stream")
    a = ap.parse_args()
    import librosa
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier

    sal = Path(a.salida)
    (sal / "wav").mkdir(parents=True, exist_ok=True)
    candidatos = sorted(Path(a.candidatos).glob("*.pt"))
    if not candidatos:
        raise SystemExit("no hay candidatos .pt")
    ssh = ["ssh", "-n", "-i", a.clave, "-o", "IdentitiesOnly=yes", a.vm]

    # ------------------------------------------------------------ síntesis en la VM
    def pedir(texto, voz):
        cuerpo = json.dumps({"texto": texto, "voz": voz, "cfg_scale": 3.0, "semilla": SEMILLA_SINTESIS,
                             "pasos": 6, "formato": "wav",
                             **({"ruido_arranque": a.ruido_arranque or None} if a.ruido_arranque is not None else {})}).encode()
        pet = urllib.request.Request(f"{a.url}/tts/stream", data=cuerpo, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {a.token}"})
        with urllib.request.urlopen(pet, timeout=600) as r:
            datos = r.read()
        i = datos.find(b"data")
        return np.frombuffer(datos[i + 8:], dtype="<i2").astype(np.float32) / 32768.0

    instaladas = []
    try:
        pendientes = [c for c in candidatos
                      if not all((sal / "wav" / f"{c.stem}__{l}{k}.wav").exists()
                                 for l in FRASES for k in range(len(FRASES[l])))]
        for c in pendientes:
            destino = f"{a.voces_vm}/eval-{c.stem}.pt"
            subprocess.run(["scp", "-q", "-i", a.clave, "-o", "IdentitiesOnly=yes", str(c), f"{a.vm}:{destino}"], check=True)
            subprocess.run(ssh + [f"chown voz-stream:voz-stream {destino}"], check=True)
            instaladas.append(destino)
        for n, c in enumerate(pendientes, 1):
            for idioma, frases in FRASES.items():
                for k, texto in enumerate(frases):
                    ruta = sal / "wav" / f"{c.stem}__{idioma}{k}.wav"
                    if not ruta.exists():
                        sf.write(str(ruta), pedir(texto, f"eval-{c.stem}"), 24000, subtype="PCM_16")
            print(f"  sintetizado {n}/{len(pendientes)}: {c.stem}", flush=True)
    finally:
        if instaladas:
            subprocess.run(ssh + ["rm -f " + " ".join(instaladas)], check=False)

    if a.solo_sintesis:
        print("solo síntesis: la puntuación, en otra pasada", flush=True)
        return

    # ------------------------------------------------------------ jueces
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"),
                                           run_opts={"device": "cpu"})

    def huella(x, sr):
        x16 = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x16).float()[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)

    pista, sr = sf.read(a.pista, dtype="float32")
    apartados = json.load(open(a.apartados))["segmentos"][: a.max_apartados]
    reales = [pista[int(s["ini"] * sr):int(s["fin"] * sr)] for s in apartados]
    centro = np.mean([huella(x, sr) for x in reales], 0)
    centro /= np.linalg.norm(centro)
    f0_real = float(np.nanmedian([PR.descripcion(x, sr)["hz"] for x in reales]))
    print(f"centroide real con {len(reales)} segmentos apartados · F0 mediana {f0_real:.0f} Hz", flush=True)

    whisper = WhisperModel(a.whisper, device="cpu", compute_type="int8")
    utmos = NA.cargar_juez()
    cache_ruta = sal / "medidas.json"
    medidas = json.load(open(cache_ruta)) if cache_ruta.exists() else {}
    for c in candidatos:
        for idioma, frases in FRASES.items():
            for k, texto in enumerate(frases):
                clave = f"{c.stem}__{idioma}{k}"
                if clave in medidas:
                    continue
                x, sr_c = sf.read(str(sal / "wav" / f"{clave}.wav"), dtype="float32")
                segs, _ = whisper.transcribe(str(sal / "wav" / f"{clave}.wav"), language=idioma, beam_size=5,
                                             vad_filter=False, condition_on_previous_text=False, temperature=0.0)
                oido = " ".join(s.text.strip() for s in segs).strip()
                hz = PR.descripcion(x, sr_c)["hz"]
                medidas[clave] = {
                    "candidato": c.stem, "idioma": idioma, "oido": oido,
                    "wer": 100 * (FI.wer(texto, oido) if idioma == "es" else wer_en(texto, oido)),
                    "ecapa": float(np.dot(huella(x, sr_c), centro)),
                    "utmos": NA.utmos_de(utmos, x, sr_c),
                    "sesgo_st": float(12 * np.log2(hz / f0_real)) if hz and hz == hz else None,
                }
        json.dump(medidas, open(cache_ruta, "w"), ensure_ascii=False, indent=1)
        print(f"  puntuado {c.stem}", flush=True)

    # ------------------------------------------------------------ tabla y elección
    filas = []
    for c in candidatos:
        m = [v for v in medidas.values() if v["candidato"] == c.stem]
        por = lambda idi, k: float(np.mean([v[k] for v in m if v["idioma"] == idi and v[k] is not None]))
        filas.append({"candidato": c.stem, "ecapa_es": por("es", "ecapa"), "ecapa_en": por("en", "ecapa"),
                      "ecapa": float(np.mean([v["ecapa"] for v in m])),
                      "wer_es": por("es", "wer"), "wer_en": por("en", "wer"),
                      "wer_max": float(max(v["wer"] for v in m)),
                      "utmos": float(np.mean([v["utmos"] for v in m])),
                      "sesgo_st": float(np.nanmean([v["sesgo_st"] for v in m if v["sesgo_st"] is not None]))})
    mejor_es = min(f["wer_es"] for f in filas)
    mejor_en = min(f["wer_en"] for f in filas)
    validos = [f for f in filas if f["wer_max"] <= 25 and f["wer_es"] <= mejor_es + 3 and f["wer_en"] <= mejor_en + 3]
    elegido = max(validos, key=lambda f: f["ecapa"]) if validos else None
    json.dump({"filas": filas, "elegido": elegido, "f0_real": f0_real, "apartados": len(reales)},
              open(sal / "resumen.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n{'candidato':38s} {'ecapa':>6s} {'es':>6s} {'en':>6s} {'wer_es':>7s} {'wer_en':>7s} {'wer_max':>7s} {'utmos':>6s} {'tono':>6s}")
    for f in sorted(filas, key=lambda f: -f["ecapa"]):
        marca = " <- elegido" if elegido and f["candidato"] == elegido["candidato"] else ("" if f in validos else "  (descartado)")
        print(f"{f['candidato']:38s} {f['ecapa']:6.3f} {f['ecapa_es']:6.3f} {f['ecapa_en']:6.3f} {f['wer_es']:7.1f} "
              f"{f['wer_en']:7.1f} {f['wer_max']:7.1f} {f['utmos']:6.2f} {f['sesgo_st']:+6.2f}{marca}")


if __name__ == "__main__":
    main()
