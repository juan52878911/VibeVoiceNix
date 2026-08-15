#!/usr/bin/env python
"""Genera audio con VibeVoice-Realtime-0.5B (streaming) y guarda WAV + latentes.

Carga el modelo ORIGINAL del hub (no la copia podada del proyecto) para que la
comparacion con el 1.5B sea entre dos modelos de fabrica. Los parches del
proyecto (freno de guia, refuerzo de arranque) estan APAGADOS salvo que se pidan
con --freno, que replica frenar_guia() de pkgs/vibevoice-cli/voz_stream.py.

No reproduce audio: escribe WAV y un .npy con los latentes por paso.
"""
import argparse
import copy
import os
import time
import types
import wave

import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

RITMO = 24000
LATENTES = []


def a_dispositivo(obj, disp, dtype):
    """Mueve el prefijo de voz (.pt, bfloat16) al dispositivo y tipo del modelo.

    Ojo: BaseModelOutputWithPast ES un OrderedDict, asi que hay que tratarlo
    ANTES que a los dicts o se aplana y generate() deja de encontrar
    .past_key_values.
    """
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.to(device=disp, dtype=dtype)
        return obj.to(device=disp)
    if isinstance(obj, ModelOutput):
        for clave in list(obj.keys()):
            obj[clave] = a_dispositivo(obj[clave], disp, dtype)
        return obj
    if isinstance(obj, Cache):
        for atrib in ("key_cache", "value_cache"):
            if getattr(obj, atrib, None) is not None:
                setattr(obj, atrib, [a_dispositivo(t, disp, dtype)
                                     for t in getattr(obj, atrib)])
        for capa in getattr(obj, "layers", []) or []:
            for atrib in ("keys", "values"):
                if getattr(capa, atrib, None) is not None:
                    setattr(capa, atrib, a_dispositivo(getattr(capa, atrib), disp, dtype))
        return obj
    if isinstance(obj, dict):
        return {k: a_dispositivo(v, disp, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(a_dispositivo(v, disp, dtype) for v in obj)
    return obj


def instrumentar(modelo, freno=0.0):
    """Captura cada latente y, si freno>0, replica frenar_guia() del proyecto."""
    original = modelo.sample_speech_tokens

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
        LATENTES.append(lat[0].float().cpu().numpy().copy())
        return lat

    modelo.sample_speech_tokens = types.MethodType(muestrear, modelo)
    return original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--voces", required=True)
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--texto", required=True)
    ap.add_argument("--semillas", default="11")
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--pasos", type=int, default=20)
    ap.add_argument("--freno", type=float, default=0.0)
    ap.add_argument("--salida", default="/tmp/deriva-05b")
    ap.add_argument("--dispositivo", default="mps")
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    guion = open(a.texto).read().strip()
    dtype = torch.float16 if a.dispositivo == "mps" else torch.float32

    print(f"cargando {a.modelo}...", flush=True)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=dtype, device_map="cpu", attn_implementation="sdpa")
    modelo = modelo.eval().to(a.dispositivo)
    modelo.set_ddpm_inference_steps(a.pasos)
    proc = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    instrumentar(modelo, freno=a.freno)
    print("escala:", modelo.model.speech_scaling_factor.item(),
          " sesgo:", modelo.model.speech_bias_factor.item(),
          " pasos:", modelo.ddpm_inference_steps, " freno:", a.freno, flush=True)

    base_voz = torch.load(os.path.join(a.voces, f"{a.voz}.pt"),
                          map_location="cpu", weights_only=False)
    base_voz = a_dispositivo(base_voz, a.dispositivo, dtype)

    for s in [int(v) for v in a.semillas.split(",")]:
        print(f"\n=== semilla {s}  cfg {a.cfg}  freno {a.freno} ===", flush=True)
        LATENTES.clear()
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
        seg = time.time() - t0
        x = sal.speech_outputs[0].float().cpu().numpy().reshape(-1)
        etiqueta = f"05b-cfg{a.cfg}-freno{a.freno}-s{s}"
        ruta = os.path.join(a.salida, etiqueta + ".wav")
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
            w.writeframes(pcm.tobytes())
        np.save(os.path.join(a.salida, etiqueta + "-latentes.npy"), np.stack(LATENTES))
        print(f"  {len(x) / RITMO:.1f}s de audio, {len(LATENTES)} latentes, "
              f"{seg:.0f}s de calculo -> {ruta}", flush=True)


if __name__ == "__main__":
    main()
