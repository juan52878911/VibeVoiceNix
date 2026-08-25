#!/usr/bin/env python
"""0.5B con ANCLAJE del latente: renormaliza cada latente a la estadistica de los
primeros N fotogramas.

Es la via que quedaba pendiente de probar. La idea: si el lazo autorregresivo se
va desviando, se le devuelve por la fuerza la energia (media y desviacion) que
tenian los latentes del arranque, cuando la voz todavia era la buena.

    z' = (z - media(z)) / std(z) * s0 + m0        con s0, m0 del arranque
    z_final = (1 - alfa) * z + alfa * z'

alfa=0 lo desactiva; alfa=1 ancla del todo.
"""
import argparse
import copy
import os
import sys
import time
import types
import wave

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

RITMO = 24000
LATENTES = []
ESTADO = {"m": [], "s": [], "m0": None, "s0": None}


def instrumentar(modelo, freno=0.0, ancla_n=40, alfa=1.0):
    @torch.no_grad()
    def muestrear(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        cond = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        voz = torch.randn(cond.shape[0], self.config.acoustic_vae_dim).to(cond)
        for t in self.model.noise_scheduler.timesteps:
            mitad = voz[: len(voz) // 2]
            combinado = torch.cat([mitad, mitad], dim=0)
            eps = self.model.prediction_head(
                combinado, t.repeat(combinado.shape[0]).to(combinado), condition=cond)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            media = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            if freno > 0:
                std_c = cond_eps.std(dim=-1, keepdim=True)
                std_g = media.std(dim=-1, keepdim=True)
                media = freno * (media * (std_c / (std_g + 1e-8))) + (1.0 - freno) * media
            eps = torch.cat([media, media], dim=0)
            voz = self.model.noise_scheduler.step(eps, t, voz).prev_sample
        lat = voz[: len(voz) // 2]

        if alfa > 0:
            m = lat.mean(dim=-1, keepdim=True)
            s = lat.std(dim=-1, keepdim=True)
            n = len(LATENTES)
            if n < ancla_n:
                ESTADO["m"].append(float(m[0].item()))
                ESTADO["s"].append(float(s[0].item()))
            else:
                if ESTADO["m0"] is None:
                    ESTADO["m0"] = float(np.mean(ESTADO["m"]))
                    ESTADO["s0"] = float(np.mean(ESTADO["s"]))
                anclado = (lat - m) / (s + 1e-6) * ESTADO["s0"] + ESTADO["m0"]
                lat = (1.0 - alfa) * lat + alfa * anclado

        LATENTES.append(lat[0].float().cpu().numpy().copy())
        return lat

    modelo.sample_speech_tokens = types.MethodType(muestrear, modelo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--voces", required=True)
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--texto", required=True)
    ap.add_argument("--semillas", default="11")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=20)
    ap.add_argument("--freno", type=float, default=0.0)
    ap.add_argument("--ancla-n", type=int, default=40)
    ap.add_argument("--alfa", type=float, default=1.0)
    ap.add_argument("--salida", default="/tmp/deriva/ancla")
    ap.add_argument("--dispositivo", default="mps")
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    guion = open(a.texto).read().strip()
    dtype = torch.float16 if a.dispositivo == "mps" else torch.float32

    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=dtype, device_map="cpu", attn_implementation="sdpa")
    modelo = modelo.eval().to(a.dispositivo)
    modelo.set_ddpm_inference_steps(a.pasos)
    proc = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    instrumentar(modelo, freno=a.freno, ancla_n=a.ancla_n, alfa=a.alfa)

    base_voz = torch.load(os.path.join(a.voces, f"{a.voz}.pt"),
                          map_location="cpu", weights_only=False)
    base_voz = a_dispositivo(base_voz, a.dispositivo, dtype)

    for s in [int(v) for v in a.semillas.split(",")]:
        print(f"\n=== semilla {s} cfg {a.cfg} freno {a.freno} ancla {a.ancla_n} alfa {a.alfa} ===",
              flush=True)
        LATENTES.clear()
        ESTADO.update({"m": [], "s": [], "m0": None, "s0": None})
        torch.manual_seed(s)
        np.random.seed(s)
        entradas = proc.process_input_with_cached_prompt(
            text=guion, cached_prompt=copy.deepcopy(base_voz),
            padding=True, return_tensors="pt", return_attention_mask=True)
        entradas = a_dispositivo(dict(entradas), a.dispositivo, dtype)
        t0 = time.time()
        with torch.no_grad():
            sal = modelo.generate(
                **entradas, max_new_tokens=None, cfg_scale=a.cfg,
                tokenizer=proc.tokenizer, generation_config={"do_sample": False},
                verbose=False, show_progress_bar=False, return_speech=True,
                all_prefilled_outputs=copy.deepcopy(base_voz), audio_streamer=None)
        x = sal.speech_outputs[0].float().cpu().numpy().reshape(-1)
        etiqueta = f"05b-ancla{a.ancla_n}-alfa{a.alfa}-cfg{a.cfg}-freno{a.freno}-s{s}"
        ruta = os.path.join(a.salida, etiqueta + ".wav")
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
            w.writeframes(pcm.tobytes())
        np.save(os.path.join(a.salida, etiqueta + "-latentes.npy"), np.stack(LATENTES))
        print(f"  {len(x) / RITMO:.1f}s, {len(LATENTES)} latentes, {time.time() - t0:.0f}s "
              f"(m0={ESTADO['m0']}, s0={ESTADO['s0']}) -> {ruta}", flush=True)


if __name__ == "__main__":
    main()
