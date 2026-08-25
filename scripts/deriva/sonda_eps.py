#!/usr/bin/env python
"""Sonda del lazo: que le hace la guia CFG al ruido predicho, paso a paso.

Registra por cada latente generado:
  std_cond    desviacion del eps de la rama condicional
  std_guiado  desviacion del eps despues de aplicar CFG
  inflado     std_guiado / std_cond   <- lo que el freno del proyecto corrige
  norma_lat   norma del latente resultante
  norma_cond  norma del estado oculto del LM que condiciona la difusion

Con eso se ve si la guia mete energia de mas y si esa energia se acumula.
"""
import argparse
import copy
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen05b import a_dispositivo  # noqa: E402

from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # noqa: E402
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (  # noqa: E402
    VibeVoiceStreamingProcessor,
)

REG = []


def instrumentar(modelo, freno=0.0):
    @torch.no_grad()
    def muestrear(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        cond = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        voz = torch.randn(cond.shape[0], self.config.acoustic_vae_dim).to(cond)
        sc = sg = 0.0
        for t in self.model.noise_scheduler.timesteps:
            mitad = voz[: len(voz) // 2]
            combinado = torch.cat([mitad, mitad], dim=0)
            eps = self.model.prediction_head(
                combinado, t.repeat(combinado.shape[0]).to(combinado), condition=cond)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            media = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            sc = float(cond_eps.std().item())
            sg = float(media.std().item())
            if freno > 0:
                std_c = cond_eps.std(dim=-1, keepdim=True)
                std_g = media.std(dim=-1, keepdim=True)
                media = freno * (media * (std_c / (std_g + 1e-8))) + (1.0 - freno) * media
            eps = torch.cat([media, media], dim=0)
            voz = self.model.noise_scheduler.step(eps, t, voz).prev_sample
        lat = voz[: len(voz) // 2]
        REG.append({
            "std_cond": sc, "std_guiado": sg,
            "inflado": sg / (sc + 1e-9),
            "norma_lat": float(lat.norm().item()),
            "norma_cond": float(condition.norm().item()),
            "norma_neg": float(neg_condition.norm().item()),
        })
        return lat

    modelo.sample_speech_tokens = types.MethodType(muestrear, modelo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--voces", required=True)
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--texto", required=True)
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--pasos", type=int, default=20)
    ap.add_argument("--dispositivo", default="mps")
    a = ap.parse_args()

    guion = open(a.texto).read().strip()
    dtype = torch.float16 if a.dispositivo == "mps" else torch.float32
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=dtype, device_map="cpu", attn_implementation="sdpa")
    modelo = modelo.eval().to(a.dispositivo)
    modelo.set_ddpm_inference_steps(a.pasos)
    proc = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    base = torch.load(os.path.join(a.voces, f"{a.voz}.pt"), map_location="cpu",
                      weights_only=False)
    base = a_dispositivo(base, a.dispositivo, dtype)

    for cfg, freno in [(1.3, 0.0), (3.0, 0.0), (3.0, 0.75)]:
        REG.clear()
        instrumentar(modelo, freno=freno)
        torch.manual_seed(a.semilla)
        entradas = proc.process_input_with_cached_prompt(
            text=guion, cached_prompt=copy.deepcopy(base), padding=True,
            return_tensors="pt", return_attention_mask=True)
        entradas = a_dispositivo(dict(entradas), a.dispositivo, dtype)
        with torch.no_grad():
            modelo.generate(**entradas, max_new_tokens=None, cfg_scale=cfg,
                            tokenizer=proc.tokenizer,
                            generation_config={"do_sample": False}, verbose=False,
                            show_progress_bar=False, return_speech=False,
                            all_prefilled_outputs=copy.deepcopy(base), audio_streamer=None)
        r = REG[:]
        n = len(r)
        print(f"\n=== cfg {cfg}  freno {freno}  ({n} latentes) ===")
        print(f"{'tramo':8s} {'inflado':>8s} {'std_cond':>9s} {'std_guiado':>11s} "
              f"{'norma_lat':>10s} {'norma_cond':>11s} {'norma_neg':>10s}")
        for i, t in enumerate(np.array_split(np.arange(n), 4)):
            s = [r[j] for j in t]
            print(f"cuarto {i+1} "
                  f"{np.mean([v['inflado'] for v in s]):8.3f} "
                  f"{np.mean([v['std_cond'] for v in s]):9.3f} "
                  f"{np.mean([v['std_guiado'] for v in s]):11.3f} "
                  f"{np.mean([v['norma_lat'] for v in s]):10.2f} "
                  f"{np.mean([v['norma_cond'] for v in s]):11.2f} "
                  f"{np.mean([v['norma_neg'] for v in s]):10.2f}")


if __name__ == "__main__":
    main()
