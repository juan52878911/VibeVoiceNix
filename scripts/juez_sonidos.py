#!/usr/bin/env python3
"""Juez de SONIDOS no verbales y de fondo con AudioSet (AST): risa, respiracion, tos, canto,
musica, teclado, multitud. Para medir si el modelo genera lo que se le pide (y nada que no).

Una risa dura menos de un segundo y AST mira 10 s de golpe: se pasa por ventanas de 2 s con
salto de 1 s y cada clase se queda con su maximo y con los segundos en que supera el umbral.

  python3 juez_sonidos.py clip.wav [clip2.wav ...]
  python3 juez_sonidos.py --lote lote.json --salida sonidos.json     # [{"audio", "etiqueta"?}]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODELO = "MIT/ast-finetuned-audioset-10-10-0.4593"
CLASES = {
    "habla": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation"],
    "risa": ["Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh"],
    "respiracion": ["Breathing", "Sigh", "Gasp", "Sniff"],
    "tos": ["Cough", "Throat clearing"],
    "canto": ["Singing", "Male singing", "Female singing"],
    "musica": ["Music", "Musical instrument", "Background music", "Piano", "Guitar", "Synthesizer",
               "Electronic music", "Pop music", "Ambient music"],
    "teclado": ["Typing", "Computer keyboard"],
    "multitud": ["Crowd", "Chatter", "Hubbub, speech noise, speech babble"],
}
UMBRAL = 0.2          # el mismo que el QC de musica de dobla (elegir_arranque.py)
VENTANA, SALTO = 2.0, 1.0


class Juez:
    def __init__(self, dispositivo="cpu"):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(MODELO)
        self.ast = ASTForAudioClassification.from_pretrained(MODELO).to(dispositivo).eval()
        self.dispositivo = dispositivo
        nombres = {v: k for k, v in self.ast.config.id2label.items()}
        self.idx = {c: [nombres[e] for e in etq if e in nombres] for c, etq in CLASES.items()}

    def medir(self, x, hz):
        if x.ndim > 1:
            x = x.mean(1)
        if hz != 16000:
            n = int(len(x) / hz * 16000)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        v, s = int(VENTANA * 16000), int(SALTO * 16000)
        trozos = [x[k:k + v] for k in range(0, max(1, len(x) - v + s), s)] or [x]
        partes = []
        for k in range(0, len(trozos), 16):                # por lotes: 15 min son 900 ventanas
            ent = self.fe(trozos[k:k + 16], sampling_rate=16000, return_tensors="pt").to(self.dispositivo)
            with self.torch.inference_mode():
                partes.append(self.torch.sigmoid(self.ast(**ent).logits).cpu().numpy())
        p = np.concatenate(partes)                          # (ventanas, 527)
        salida = {}
        for c, idx in self.idx.items():
            por_ventana = p[:, idx].max(1)
            salida[c] = {"max": round(float(por_ventana.max()), 3),
                         "segundos": [round(k * SALTO, 1) for k in np.where(por_ventana > UMBRAL)[0]]}
        return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audios", nargs="*")
    ap.add_argument("--lote")
    ap.add_argument("--salida")
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()
    casos = json.loads(Path(a.lote).read_text()) if a.lote else [{"audio": w} for w in a.audios]
    j = Juez(a.dispositivo)
    res = []
    for c in casos:
        x, hz = sf.read(str(c["audio"]), dtype="float32")
        m = j.medir(x, hz)
        res.append({**c, "sonidos": m})
        hay = {k: v["max"] for k, v in m.items() if k != "habla" and v["max"] > UMBRAL}
        print(f"{c.get('etiqueta', Path(c['audio']).stem):32s} habla {m['habla']['max']:.2f}  "
              f"{' '.join(f'{k} {v:.2f}' for k, v in hay.items()) or '-'}", flush=True)
    if a.salida:
        Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
