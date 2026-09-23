#!/usr/bin/env python3
"""Todos los jueces del plan de mejora sobre un lote, en una pasada. Corre en CPU o en GPU.

Por clip:
  ecapa        parecido de huella contra la identidad (centroide de sus audios REALES de control)
  wer          whisper large-v3 con el idioma fijado (la misma normalizacion que el banco de
               reconstruccion). Medido (22-09): CPU int8 y GPU float16 dan el mismo WER en 20/20 clips
  wer_norm     el mismo WER con referencia Y transcripcion pasadas por normalizar_texto.py: whisper
               escribe "3.30" o "$40,000" donde el texto dice "3:30" o "40,000 dollars", y eso no es
               un error del modelo
  per          juez_acento.py: fonemas contra la variedad nativa mas cercana
  utmos        UTMOS22 strong (naturalidad)
  pausas_min, silabas_s   perfil_vocal.py (ritmo)
  sonidos      juez_sonidos.py (solo con --sonidos: es el mas caro)

  python3 juez_lote.py lote.json medidas.json --identidades ids/ [--dispositivo cuda] [--sonidos]
  lote.json: [{"clave", "audio", "texto", "idioma", "identidad"?}]
  ids/<identidad>/*.wav: audios reales de control de cada identidad (nunca los de su referencia)

Reanudable: lo ya medido en medidas.json no se repite.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import juez_acento as JA  # noqa: E402
import perfil_vocal as PVOC  # noqa: E402
try:
    from normalizar_texto import normalizar  # noqa: E402  (necesita num2words)
except ImportError:
    normalizar = None


def wer(ref, hip):
    def norm(t):
        t = unicodedata.normalize("NFD", t.lower())
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9' ]+", " ", t).split()
    r, h = norm(ref), norm(hip)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def a16(x, hz):
    if x.ndim > 1:
        x = x.mean(1)
    if hz == 16000:
        return x.astype(np.float32)
    import librosa
    return librosa.resample(x.astype(np.float32), orig_sr=hz, target_sr=16000)


class Jueces:
    def __init__(self, dispositivo="cpu", hilos=2, sonidos=False):
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch, self.disp = torch, dispositivo
        gpu = dispositivo.startswith("cuda")
        self.whisper = WhisperModel("large-v3", device="cuda" if gpu else "cpu",
                                    compute_type="float16" if gpu else "int8", cpu_threads=hilos)
        self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"),
                                                    run_opts={"device": dispositivo})
        self.fonemas = JA.Reconocedor(dispositivo)
        self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dispositivo).eval()
        self.sonidos = None
        if sonidos:
            import juez_sonidos as JS
            self.sonidos = JS.Juez(dispositivo)

    def huella(self, x16):
        with self.torch.inference_mode():
            e = self.ecapa.encode_batch(self.torch.from_numpy(np.ascontiguousarray(x16))[None].to(self.disp))
        e = e.squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-12)

    def medir(self, c, centros):
        x, hz = sf.read(str(c["audio"]), dtype="float32")
        x16 = a16(x, hz)
        m = {"dur": round(len(x16) / 16000, 3)}
        if c.get("identidad") in centros and len(x16) > 12800:
            m["ecapa"] = round(float(self.huella(x16) @ centros[c["identidad"]]), 4)
        if c.get("texto"):
            segs, _ = self.whisper.transcribe(x16, language=c.get("idioma", "es"), beam_size=5,
                                              condition_on_previous_text=False, temperature=0.0)
            m["oido"] = " ".join(s.text for s in segs).strip()
            m["wer"] = round(wer(c["texto"], m["oido"]), 4)
            if normalizar:
                idioma = c.get("idioma", "es")
                m["wer_norm"] = round(wer(normalizar(c["texto"], idioma), normalizar(m["oido"], idioma)), 4)
            oidos = self.fonemas.fonemas(c["audio"])
            m["per"], m["variedad"], _ = JA.per_idioma(c["texto"], c.get("idioma", "es"), oidos)
            m["per"] = round(m["per"], 4)
        with self.torch.inference_mode():
            m["utmos"] = round(float(self.utmos(self.torch.from_numpy(x16)[None].to(self.disp), 16000).item()), 3)
        p = PVOC.perfil(x, hz, c.get("texto"))
        m["pausas_min"] = p.get("pausas_min")
        m["silabas_s"] = p.get("silabas_s")
        if self.sonidos:
            m["sonidos"] = {k: v["max"] for k, v in self.sonidos.medir(x, hz).items()}
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lote")
    ap.add_argument("salida")
    ap.add_argument("--identidades", help="carpeta con <identidad>/*.wav reales de control")
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--sonidos", action="store_true")
    a = ap.parse_args()
    lote = json.loads(Path(a.lote).read_text(encoding="utf-8"))
    sal = Path(a.salida)
    medidas = json.loads(sal.read_text()) if sal.exists() else {}
    pendientes = [c for c in lote if c["clave"] not in medidas]
    print(f"[juez] {len(lote)} clips, {len(pendientes)} pendientes, en {a.dispositivo}", flush=True)
    if not pendientes:
        return 0
    j = Jueces(a.dispositivo, a.hilos, a.sonidos)
    centros = {}
    if a.identidades:
        for d in sorted(Path(a.identidades).iterdir()):
            hs = []
            for w in sorted(d.glob("*.wav")):
                x, hz = sf.read(str(w), dtype="float32")
                x16 = a16(x, hz)
                for k in range(0, max(1, len(x16) - 16000 * 3), 16000 * 8):   # trozos de 8 s
                    if len(x16[k:k + 16000 * 8]) > 16000 * 2:
                        hs.append(j.huella(x16[k:k + 16000 * 8]))
            if hs:
                v = np.mean(hs, 0)
                centros[d.name] = v / np.linalg.norm(v)
        print(f"[juez] identidades: {', '.join(sorted(centros))}", flush=True)
    for n, c in enumerate(pendientes, 1):
        medidas[c["clave"]] = {**{k: v for k, v in c.items() if k != "audio"}, **j.medir(c, centros)}
        if n % 20 == 0 or n == len(pendientes):
            sal.write_text(json.dumps(medidas, ensure_ascii=False, indent=1))
            print(f"[juez] {n}/{len(pendientes)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
