#!/usr/bin/env python
"""Fase 3, paso 1: la condición que da el LM cuando el audio que vuelve es el REAL de la persona.

    python scripts/fase3_condiciones.py --dataset /var/lib/taller/dataset-voces --voces /var/lib/taller/fase1/voces \
        --salida /var/lib/taller/fase3/condiciones

Para cada clip del manifiesto cuya persona tenga clon (.pt de la fase 1):

  1. audio real -> codificador acústico (encoder comunitario) -> media de los latentes, escalada como la ve
     la cabeza de difusión: (media + sesgo) · escala (generate() decodifica con latente / escala − sesgo).
  2. generate() con el prefijo del clon y la transcripción del clip, FORZADO: sample_speech_tokens devuelve
     el latente real del fotograma i en vez de muestrear y guarda la condición positiva que recibió. El
     latente real vuelve al LM por el conector, así que la condición del fotograma i+1 es la que el LM daría
     si la persona hubiese hablado así hasta ahí. El texto entra por ventanas, por delante del habla, igual
     que al generar.

Sin difusión ni decodificador (se sustituye por ceros) y con el clasificador de fin anulado: la generación
dura exactamente los fotogramas del clip.

Deja <salida>/<persona>/<clip>.npz con cond [T, 896], media [T, 64] (escalada), std (la fija del
codificador, escalada) y los tokens de texto. Se salta los que ya existen.
"""
import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--voces", required=True, help="los .pt de clonar_voz.py de la fase 1")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=os.environ.get("VIBEVOICE_MODELO"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    ap.add_argument("--hilos", type=int, default=6)
    a = ap.parse_args()
    import soundfile as sf
    import torch
    import torch.nn as nn
    from auditar_encoder import encoder_comunitario
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    torch.set_num_threads(a.hilos)

    filas = list(csv.DictReader(open(Path(a.dataset) / "manifiesto.csv", encoding="utf-8")))
    filas = [f for f in filas if (Path(a.voces) / f"{f['hablante']}.pt").exists()]
    pendientes = [f for f in filas if not (Path(a.salida) / f["hablante"] / f"{Path(f['fichero']).stem}.npz").exists()]
    print(f"{len(filas)} clips con clon, {len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return

    proc = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    m = modelo.model
    m.tts_language_model.embed_tokens = m.language_model.embed_tokens
    m.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    escala, sesgo = float(m.speech_scaling_factor), float(m.speech_bias_factor)
    std_fija = float(m.acoustic_tokenizer.fix_std) * escala
    print(f"escala {escala:.4f} · sesgo {sesgo:.4f} · std fija escalada {std_fija:.4f}", flush=True)

    class SinFin(nn.Module):
        def forward(self, h):
            return torch.full((*h.shape[:-1], 1), -30.0)

    modelo.tts_eos_classifier = SinFin()
    m.acoustic_tokenizer.decode = lambda *args, **kw: torch.zeros(1, 1, 3200)

    prefijos = {}
    t0, seg = time.time(), 0.0
    for n, f in enumerate(pendientes, 1):
        persona, clip = f["hablante"], Path(f["fichero"]).stem
        if persona not in prefijos:
            prefijos[persona] = torch.load(Path(a.voces) / f"{persona}.pt", weights_only=False, map_location="cpu")
        x, sr = sf.read(str(Path(a.dataset) / f["fichero"]), dtype="float32")
        with torch.no_grad():
            media = m.acoustic_tokenizer.encode(torch.from_numpy(x)[None, None]).mean[0]
        lat = (media + sesgo) * escala
        T = len(lat)
        estado, conds = {"i": 0}, []

        def forzado(condition, neg_condition, cfg_scale=3.0):
            i = estado["i"]
            estado["i"] += 1
            if i < T:
                conds.append(condition[0].detach().float().clone())
            return lat[min(i, T - 1)][None]

        modelo.sample_speech_tokens = forzado
        entradas = proc.process_input_with_cached_prompt(
            text=f["texto"].strip() + "\n", cached_prompt=copy.deepcopy(prefijos[persona]),
            padding=True, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            modelo.generate(**entradas, max_new_tokens=None, cfg_scale=3.0, tokenizer=proc.tokenizer,
                            generation_config={"do_sample": False}, verbose=False, return_speech=False,
                            all_prefilled_outputs=copy.deepcopy(prefijos[persona]),
                            stop_check_fn=lambda: estado["i"] >= T)
        if len(conds) != T:
            print(f"  AVISO {persona} {clip}: {len(conds)} condiciones para {T} fotogramas; se descarta", flush=True)
            continue
        destino = Path(a.salida) / persona / f"{clip}.npz"
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino, cond=torch.stack(conds).numpy(), media=lat.numpy(), std=np.float32(std_fija),
                            texto_tokens=np.int32(entradas["tts_text_ids"].shape[1]))
        seg += len(x) / sr
        if n % 10 == 0 or n == len(pendientes):
            dt = time.time() - t0
            print(f"  {n}/{len(pendientes)} · {seg / 60:.1f} min de audio en {dt / 60:.1f} min (x{dt / seg:.2f}) · "
                  f"ultimo {T} fotogramas, {int(entradas['tts_text_ids'].shape[1])} tokens", flush=True)


if __name__ == "__main__":
    main()
