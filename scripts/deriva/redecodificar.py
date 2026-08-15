#!/usr/bin/env python
"""Control: decodifica latentes YA generados en BLOQUE, sin cache de streaming.

Sirve para separar dos causas posibles de la deriva de tono en el 0.5B:
  - si al decodificar de golpe la deriva SIGUE ahi, esta en los latentes (el LM)
  - si desaparece, la culpa era del estado de la cache del decodificador causal

El 1.5B decodifica siempre en bloque, asi que esto pone a los dos modelos en el
mismo terreno.
"""
import argparse
import glob
import os
import wave

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)

RITMO = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--latentes", required=True, help="patron glob de *-latentes.npy")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--dispositivo", default="mps")
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    dtype = torch.float16 if a.dispositivo == "mps" else torch.float32
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=dtype, device_map="cpu", attn_implementation="sdpa")
    modelo = modelo.eval().to(a.dispositivo)
    tok = modelo.model.acoustic_tokenizer
    escala = modelo.model.speech_scaling_factor.to(a.dispositivo)
    sesgo = modelo.model.speech_bias_factor.to(a.dispositivo)

    for ruta in sorted(glob.glob(a.latentes)):
        L = np.load(ruta)
        t = torch.from_numpy(L).to(a.dispositivo).to(dtype).unsqueeze(0)
        with torch.no_grad():
            audio = tok.decode(t / escala - sesgo)
        x = audio[0, 0].float().cpu().numpy()
        nombre = os.path.basename(ruta).replace("-latentes.npy", "-bloque.wav")
        destino = os.path.join(a.salida, nombre)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(destino, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
            w.writeframes(pcm.tobytes())
        print(f"{nombre}  {len(L)} latentes -> {len(x) / RITMO:.1f}s")


if __name__ == "__main__":
    main()
