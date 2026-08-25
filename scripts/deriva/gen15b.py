#!/usr/bin/env python
"""Genera audio con VibeVoice-1.5B (arquitectura NO streaming) y guarda WAV + latentes.

El paquete instalado trae la arquitectura del 1.5B (VibeVoiceForConditionalGeneration)
pero NO el bucle de inferencia. Aqui se reconstruye ese bucle a partir de las mismas
piezas que usa el 0.5B en modeling_vibevoice_streaming_inference.py:

  prefill  -> LM ve TODO el texto y el prompt de voz de una vez
  bucle    -> hidden del ultimo paso condiciona la difusion de UN latente acustico
              el latente vuelve como embedding del siguiente paso (acoustic_connector)
  final    -> los latentes se decodifican DE GOLPE (sin cache streaming)

La rama negativa del CFG replica la del 0.5B: arranca con <|image_pad|> y recibe los
mismos latentes realimentados.

No reproduce audio: escribe WAV y un .npy con los latentes por paso.
"""
import argparse
import json
import os
import time
import wave

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

RITMO = 24000


def cargar(ruta, dispositivo, dtype):
    modelo = VibeVoiceForConditionalGeneration.from_pretrained(
        ruta, dtype=dtype, attn_implementation="sdpa")
    modelo = modelo.to(dispositivo).eval()
    proc = VibeVoiceProcessor.from_pretrained(ruta)
    return modelo, proc


@torch.no_grad()
def difunde_latente(modelo, cond, neg_cond, cfg_scale, pasos):
    """Identico a sample_speech_tokens del 0.5B: CFG sobre el head de difusion."""
    modelo.model.noise_scheduler.set_timesteps(pasos)
    condicion = torch.cat([cond, neg_cond], dim=0)
    voz = torch.randn(condicion.shape[0], modelo.config.acoustic_vae_dim).to(condicion)
    for t in modelo.model.noise_scheduler.timesteps:
        mitad = voz[: len(voz) // 2]
        combinado = torch.cat([mitad, mitad], dim=0)
        eps = modelo.model.prediction_head(
            combinado, t.repeat(combinado.shape[0]).to(combinado), condition=condicion)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        media_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([media_eps, media_eps], dim=0)
        voz = modelo.model.noise_scheduler.step(eps, t, voz).prev_sample
    return voz[: len(voz) // 2]


@torch.no_grad()
def genera(modelo, proc, guion, voz_ref, cfg_scale=1.3, semilla=11,
           max_latentes=1200, pasos_ddpm=20, verbose=True):
    disp = modelo.device
    dtype = next(modelo.parameters()).dtype
    tok = proc.tokenizer

    entradas = proc(text=[guion], voice_samples=[[voz_ref]], return_tensors="pt")
    input_ids = entradas["input_ids"].to(disp)
    mascara_voz = entradas["speech_input_mask"].to(disp)
    tensores_voz = entradas["speech_tensors"].to(disp).to(dtype)
    mascaras_voz = entradas["speech_masks"].to(disp)

    torch.manual_seed(semilla)
    np.random.seed(semilla)

    # --- prompt de voz: latentes acusticos + semanticos del audio de referencia ---
    sal_ac = modelo.model.acoustic_tokenizer.encode(tensores_voz.unsqueeze(1))
    fichas_ac = sal_ac.sample(modelo.model.acoustic_tokenizer.std_dist_type)[0]
    escala = modelo.model.speech_scaling_factor.to(fichas_ac)
    sesgo = modelo.model.speech_bias_factor.to(fichas_ac)
    rasgos_ac = (fichas_ac + sesgo) * escala
    emb_ac = modelo.model.acoustic_connector(rasgos_ac)

    sal_sem = modelo.model.semantic_tokenizer.encode(tensores_voz.unsqueeze(1))
    rasgos_sem = sal_sem.mean if hasattr(sal_sem, "mean") else sal_sem
    emb_sem = modelo.model.semantic_connector(rasgos_sem)

    x = modelo.get_input_embeddings()(input_ids)
    x[mascara_voz] = (emb_ac[mascaras_voz] + emb_sem[mascaras_voz]).to(x.dtype)

    # --- prefill positivo ---
    salida = modelo.model(inputs_embeds=x, use_cache=True, return_dict=True)
    cache_pos = salida.past_key_values
    oculto = salida.last_hidden_state[:, -1, :]

    # --- prefill negativo: un solo <|image_pad|>, igual que el 0.5B ---
    id_neg = tok.convert_tokens_to_ids("<|image_pad|>")
    ids_neg = torch.full((1, 1), id_neg, dtype=torch.long, device=disp)
    salida_neg = modelo.model(input_ids=ids_neg, use_cache=True, return_dict=True)
    cache_neg = salida_neg.past_key_values
    oculto_neg = salida_neg.last_hidden_state[:, -1, :]

    latentes = []
    id_fin = tok.speech_end_id
    id_eos = tok.eos_token_id
    t0 = time.time()
    for paso in range(max_latentes):
        lat = difunde_latente(modelo, oculto, oculto_neg, cfg_scale, pasos_ddpm)
        latentes.append(lat[0].float().cpu().numpy().copy())

        emb = modelo.model.acoustic_connector(lat).unsqueeze(1)

        salida = modelo.model(inputs_embeds=emb, past_key_values=cache_pos,
                              use_cache=True, return_dict=True)
        cache_pos = salida.past_key_values
        oculto = salida.last_hidden_state[:, -1, :]

        salida_neg = modelo.model(inputs_embeds=emb, past_key_values=cache_neg,
                                  use_cache=True, return_dict=True)
        cache_neg = salida_neg.past_key_values
        oculto_neg = salida_neg.last_hidden_state[:, -1, :]

        # el LM decide cuando callar
        logits = modelo.lm_head(oculto)
        siguiente = int(logits.argmax(-1)[0].item())
        if siguiente in (id_fin, id_eos):
            if verbose:
                print(f"  fin por token {siguiente} en el paso {paso + 1}")
            break
        if verbose and (paso + 1) % 50 == 0:
            print(f"  {paso + 1} latentes  {time.time() - t0:.0f}s", flush=True)

    L = torch.from_numpy(np.stack(latentes)).to(disp).to(dtype).unsqueeze(0)
    L_esc = L / escala - sesgo
    audio = modelo.model.acoustic_tokenizer.decode(L_esc)
    x = audio[0, 0].float().cpu().numpy()
    return x, np.stack(latentes), time.time() - t0


def guarda_wav(x, ruta):
    pcm = np.clip(x, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--voz", required=True)
    ap.add_argument("--texto", required=True, help="fichero con el guion")
    ap.add_argument("--semillas", default="11")
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--salida", default="/tmp/deriva-15b")
    ap.add_argument("--dispositivo", default="mps")
    ap.add_argument("--max-latentes", type=int, default=1200)
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    guion = open(a.texto).read().strip()
    dtype = torch.float16 if a.dispositivo == "mps" else torch.float32

    print(f"cargando {a.modelo} en {a.dispositivo} ({dtype})...", flush=True)
    t0 = time.time()
    modelo, proc = cargar(a.modelo, a.dispositivo, dtype)
    print(f"cargado en {time.time() - t0:.0f}s", flush=True)
    print("escala:", modelo.model.speech_scaling_factor.item(),
          " sesgo:", modelo.model.speech_bias_factor.item(), flush=True)

    for s in [int(v) for v in a.semillas.split(",")]:
        print(f"\n=== semilla {s}  cfg {a.cfg} ===", flush=True)
        x, lat, seg = genera(modelo, proc, guion, a.voz, cfg_scale=a.cfg,
                             semilla=s, max_latentes=a.max_latentes)
        base = os.path.join(a.salida, f"15b-cfg{a.cfg}-s{s}")
        guarda_wav(x, base + ".wav")
        np.save(base + "-latentes.npy", lat)
        print(f"  {len(x) / RITMO:.1f}s de audio, {len(lat)} latentes, "
              f"{seg:.0f}s de calculo -> {base}.wav", flush=True)


if __name__ == "__main__":
    main()
