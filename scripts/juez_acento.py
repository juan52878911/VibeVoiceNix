#!/usr/bin/env python3
"""Juez de PRONUNCIACION: PER (tasa de error de fonemas) de un audio contra su texto.

Sirve para medir el acento, que hasta ahora no se podia medir: el WER dice si se entiende la
palabra, no COMO se pronuncia. Aqui se comparan los fonemas que se oyen con los que tocarian en
ese idioma:

  fonemas del audio : wav2vec2-lv-60-espeak-cv-ft (reconocedor de fonemas, multilingue)
  fonemas del texto : espeak-ng --ipa en el idioma pedido
  PER               : distancia de edicion entre las dos cadenas / longitud de la de referencia

Leer el numero: un hablante nativo del idioma da el PER mas bajo; el mismo texto dicho con acento
extranjero se aleja de la transcripcion de espeak de ESE idioma y sube. El valor absoluto depende
del reconocedor, asi que solo vale COMPARADO con los controles de la misma tanda (ver --controles).

  python3 juez_acento.py --idioma en clip1.wav clip2.wav --texto "This is a test"
  python3 juez_acento.py --lote peticiones.json          # [{"audio", "texto", "idioma", "etiqueta"?}]
  python3 juez_acento.py --lote x.json --salida per.json --dispositivo cuda
"""
import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

MODELO = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# Cada idioma se compara contra TODAS sus variedades nativas y se queda la mas cercana: un colombiano
# no tiene "acento extranjero" por sesear. MEDIDO (22-09): contra el espanol de Espana solo, el clon
# de Juan (bogotano) daba PER 0,320, peor que las voces extranjeras; con seseo, ver --calibrar.
VARIEDADES = {"es": ["es", "es-419"], "en": ["en-us", "en-gb"], "fr": ["fr-fr"], "de": ["de"],
              "pt": ["pt", "pt-br"], "it": ["it"]}
# El reconocedor no marca acento tonico ni separa silabas; espeak si. Se quitan de los dos lados.
SOBRAN = "ˈˌːˑ.,;:!?¡¿\"'()-–—…"


def fonemas_texto(texto, variedad):
    sal = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", variedad, texto],
                         capture_output=True, text=True, check=True).stdout
    return limpiar(sal)


def per_idioma(texto, idioma, oidos):
    """PER contra la variedad nativa mas cercana del idioma. Devuelve (per, variedad, fonemas)."""
    mejor = None
    for v in VARIEDADES.get(idioma, [idioma]):
        ref = fonemas_texto(texto, v)
        p = per(ref, oidos)
        if mejor is None or p < mejor[0]:
            mejor = (p, v, ref)
    return mejor


def limpiar(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if c not in SOBRAN and unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return [c for c in s if c != " "]


def per(ref, hip):
    """Distancia de edicion por fonema, normalizada por la referencia."""
    d = list(range(len(hip) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hip) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hip[j - 1]))
    return d[len(hip)] / max(1, len(ref))


class Reconocedor:
    def __init__(self, dispositivo="cpu"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODELO)
        self.modelo = Wav2Vec2ForCTC.from_pretrained(MODELO).to(dispositivo).eval()
        self.dispositivo = dispositivo

    def fonemas(self, wav):
        x, hz = sf.read(str(wav), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if hz != 16000:                      # remuestreo lineal: el reconocedor solo acepta 16 kHz
            n = int(len(x) / hz * 16000)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        e = self.proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.dispositivo)
        with self.torch.inference_mode():
            ids = self.modelo(e).logits.argmax(-1)
        return limpiar(self.proc.batch_decode(ids)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audios", nargs="*")
    ap.add_argument("--texto")
    ap.add_argument("--idioma", default="es")
    ap.add_argument("--lote", help="json con [{audio, texto, idioma, etiqueta?}]")
    ap.add_argument("--salida")
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()
    if a.lote:
        casos = json.loads(Path(a.lote).read_text(encoding="utf-8"))
    else:
        casos = [{"audio": w, "texto": a.texto, "idioma": a.idioma} for w in a.audios]
    r = Reconocedor(a.dispositivo)
    salida = []
    for c in casos:
        oidos = r.fonemas(c["audio"])
        v, variedad, ref = per_idioma(c["texto"], c["idioma"], oidos)
        salida.append({**c, "per": round(v, 4), "variedad": variedad,
                       "fonemas_oidos": "".join(oidos), "fonemas_texto": "".join(ref)})
        print(f"{c.get('etiqueta', Path(c['audio']).stem):32s} [{variedad}] PER {v:.3f}", flush=True)
    if a.salida:
        Path(a.salida).write_text(json.dumps(salida, ensure_ascii=False, indent=1))
    if len(salida) > 1:
        v = np.array([s["per"] for s in salida])
        print(f"\nPER medio {v.mean():.3f} · mediana {np.median(v):.3f} · n {len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
