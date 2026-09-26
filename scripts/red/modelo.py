"""Cargar el 0.5B (o una copia con su forma y pesos aleatorios) y los prefijos de voz, para mirar la red por dentro.

    modelo, tok = cargar(ruta)             # el modelo real: <ruta> con config.json, model.safetensors y tokenizer
    modelo, tok = cargar(None, aleatorio=True)   # la misma arquitectura con pesos al azar: solo para probar mecanica
    base = prefijo("~/.cache/vibevoice-nix/voces/sp-Spk1_man.pt")

Con --aleatorio no hay tokenizador (vive en el repo del modelo, que no siempre esta a mano): `fichas()` devuelve
ids al azar del tamano pedido y `TokFalso` da los ids especiales que generate() necesita. Sirve para comprobar
que un bucle, una foto de la cache o un enganche hacen lo que dicen; los numeros de calidad salen solo con
los pesos reales.
"""
import copy
import os
from pathlib import Path

import torch

FORMA_05B = dict(hidden_size=896, num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
                 intermediate_size=4864, vocab_size=151936, max_position_embeddings=65536, rope_theta=1000000.0,
                 tie_word_embeddings=True)
CAPAS_TTS = 20
# Escala y sesgo con que la cabeza ve los latentes en el 0.5B (docs/clonado-de-voz.md §4.1).
ESCALA_05B, SESGO_05B = 0.2334, -0.0703
IMAGE_PAD = 151655


class TokFalso:
    """Lo minimo que generate() y el procesador leen de un tokenizador Qwen2."""
    bos_token_id = None
    eos_token_id = 151643
    pad_token_id = IMAGE_PAD
    pad_id = IMAGE_PAD
    speech_start_id = 151646
    speech_end_id = 151647
    speech_diffusion_id = 151648

    def convert_tokens_to_ids(self, t):
        return {"<|image_pad|>": IMAGE_PAD}[t]

    def encode(self, texto, add_special_tokens=False):
        g = torch.Generator().manual_seed(abs(hash(texto)) % (2 ** 31))
        n = max(3, len(texto.split()) * 2)
        return torch.randint(1000, 100000, (n,), generator=g).tolist()


def cargar(ruta=None, aleatorio=False, atencion="sdpa", dtype=torch.float32):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference as M)
    if aleatorio:
        from transformers import Qwen2Config
        from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
        cfg = VibeVoiceStreamingConfig(decoder_config=Qwen2Config(**FORMA_05B), tts_backbone_num_hidden_layers=CAPAS_TTS,
                                       diffusion_head_config=dict(hidden_size=896, head_layers=4, latent_size=64))
        cfg._attn_implementation = atencion
        torch.manual_seed(0)
        m = M(cfg).eval()
        m.model.speech_scaling_factor.fill_(ESCALA_05B)
        m.model.speech_bias_factor.fill_(SESGO_05B)
        m.set_ddpm_inference_steps(6)
        return m, TokFalso()
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    ruta = os.path.expanduser(ruta)
    proc = VibeVoiceStreamingProcessor.from_pretrained(ruta)
    m = M.from_pretrained(ruta, dtype=dtype, device_map="cpu", attn_implementation=atencion).eval()
    m.model.tts_language_model.embed_tokens = m.model.language_model.embed_tokens
    m.set_ddpm_inference_steps(6)
    return m, proc.tokenizer


def prefijo(ruta):
    """El .pt de una voz: dict lm / tts_lm / neg_lm / neg_tts_lm con BaseModelOutputWithPast (KV en bf16)."""
    return torch.load(os.path.expanduser(ruta), map_location="cpu", weights_only=False)


def copia(base):
    return copy.deepcopy(base)


def fichas(texto, tok):
    return tok.encode(texto.strip() + "\n", add_special_tokens=False)


def voces_disponibles(carpeta):
    return sorted(p.stem for p in Path(os.path.expanduser(carpeta)).glob("*.pt"))
