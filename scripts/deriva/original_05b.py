#!/usr/bin/env python
"""Genera audio con el codigo ORIGINAL de VibeVoice (clon sin tocar) y guarda WAV.

El bucle de carga y de generate() es copia literal de
demo/realtime_model_inference_from_file.py del repo oficial
(microsoft/VibeVoice, commit 94da20d). Lo unico que se anade es:

  - fijar la semilla antes de generate(), porque el demo no la fija y sin eso
    no hay forma de repetir una medida;
  - poder pedir cfg, pasos y tipo por linea de ordenes;
  - escribir el WAV a mano (24 kHz, int16) en vez de con save_audio, para que
    el fichero sea exactamente el que espera medir.py.

Y, aparte, las palancas --freno / --refuerzo / --depthwise / --sin-encoder /
--compartir-emb / --int8 / --cebado, que REPRODUCEN (no importan) cada una de
las modificaciones de pkgs/vibevoice-cli/voz_stream.py para poder encenderlas
de una en una SOBRE el codigo original y ver cual mueve la deriva.

No importa NADA de VibeVoiceNix. No reproduce audio.

    python scripts/deriva/original_05b.py --cfg 1.3 --semillas 11,42,7
"""
import argparse
import copy
import os
import sys
import time
import types
import wave

# El clon sin tocar manda sobre cualquier vibevoice que hubiera instalado.
CLON = os.environ.get("VIBEVOICE_ORIGINAL", "/Users/juanbedoya/Documents/GitHub/VibeVoice")
sys.path.insert(0, CLON)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402

import vibevoice  # noqa: E402
from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # noqa: E402
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (  # noqa: E402
    VibeVoiceStreamingProcessor,
)

RITMO = 24000


# ------------------------------------------------- copias de las palancas --
def poner_freno(modelo, freno: float):
    """Copia de frenar_guia() de voz_stream.py: reescala el eps guiado."""

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(
            self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0],
                             self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined),
                condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            std_cond = cond_eps.std(dim=-1, keepdim=True)
            std_guiado = half_eps.std(dim=-1, keepdim=True)
            half_eps = (freno * (half_eps * (std_cond / (std_guiado + 1e-8)))
                        + (1.0 - freno) * half_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]

    modelo.sample_speech_tokens = types.MethodType(sample_speech_tokens, modelo)


def poner_refuerzo(modelo, cfg_arranque=4.5, fotogramas=6):
    """Copia de reforzar_guia_arranque() de voz_stream.py."""
    contador = {"frame": 0}
    muestrear = modelo.sample_speech_tokens
    generar = modelo.generate

    def generate_reiniciado(*a, **kw):
        contador["frame"] = 0
        return generar(*a, **kw)

    def muestrear_reforzado(condition, neg_condition, cfg_scale=3.0):
        k = contador["frame"]
        contador["frame"] = k + 1
        if k < fotogramas and cfg_arranque > cfg_scale:
            peso = (fotogramas - k) / fotogramas
            cfg_scale = cfg_scale + (cfg_arranque - cfg_scale) * peso
        return muestrear(condition, neg_condition, cfg_scale)

    modelo.generate = generate_reiniciado
    modelo.sample_speech_tokens = muestrear_reforzado


class ConvDepthwiseRapida(torch.nn.Module):
    """Copia de ConvDepthwiseRapida de voz_stream.py."""

    def __init__(self, conv):
        super().__init__()
        c, k = conv.out_channels, conv.kernel_size[0]
        self.k = k
        w = conv.weight.detach().reshape(c, k).t().contiguous().view(k, 1, c, 1)
        self.register_buffer("w", w)
        self.tiene_sesgo = conv.bias is not None
        if self.tiene_sesgo:
            self.register_buffer("b", conv.bias.detach().reshape(1, c, 1).clone())

    def forward(self, x):
        largo = x.shape[2] - self.k + 1
        salida = x[:, :, :largo] * self.w[0]
        for j in range(1, self.k):
            salida = salida + x[:, :, j:j + largo] * self.w[j]
        return salida + self.b if self.tiene_sesgo else salida


def poner_depthwise(modelo) -> int:
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    dec = getattr(tok, "decoder", None)
    if dec is None:
        return 0
    cambiadas = 0
    for padre in dec.modules():
        for nombre, hijo in list(padre.named_children()):
            if (isinstance(hijo, torch.nn.Conv1d)
                    and hijo.groups > 1
                    and hijo.groups == hijo.in_channels == hijo.out_channels
                    and hijo.stride[0] == 1
                    and hijo.dilation[0] == 1
                    and hijo.padding[0] == 0):
                setattr(padre, nombre, ConvDepthwiseRapida(hijo))
                cambiadas += 1
    return cambiadas


def soltar_encoder(modelo) -> bool:
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    enc = getattr(tok, "encoder", None)
    if enc is None or isinstance(enc, torch.nn.Identity):
        return False
    tok.encoder = torch.nn.Identity()
    return True


def compartir_embeddings(modelo) -> bool:
    m = getattr(modelo, "model", None)
    lm = getattr(m, "language_model", None)
    tts = getattr(m, "tts_language_model", None)
    if lm is None or tts is None:
        return False
    buena = getattr(lm, "embed_tokens", None)
    muerta = getattr(tts, "embed_tokens", None)
    if buena is None or muerta is None or muerta is buena:
        return False
    if buena.weight.shape != muerta.weight.shape:
        return False
    tts.embed_tokens = buena
    return True


def poner_cebado(modelo) -> bool:
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    if tok is None or getattr(tok, "_cebado", False):
        return False
    decodificar = tok.decode

    def decode_cebado(latents, cache=None, sample_indices=None, use_cache=False, **kw):
        if use_cache and cache is not None and not getattr(cache, "cache", True):
            decodificar(torch.zeros_like(latents), cache=cache,
                        sample_indices=sample_indices, use_cache=True, **kw)
        return decodificar(latents, cache=cache, sample_indices=sample_indices,
                           use_cache=use_cache, **kw)

    tok.decode = decode_cebado
    tok._cebado = True
    return True


# ------------------------------------------------------------------ util --
def a_dispositivo(obj, disp, dtype):
    """Mueve el prefijo de voz al dispositivo/tipo del modelo.

    Solo se usa con --mover-voz. El demo oficial NO hace esto (carga con
    map_location y ya), asi que por defecto se replica el demo.
    """
    from transformers.utils import ModelOutput
    from transformers.cache_utils import Cache
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="microsoft/VibeVoice-Realtime-0.5B")
    ap.add_argument("--voz", default=os.path.join(CLON, "demo/voices/streaming_model/sp-Spk1_man.pt"))
    ap.add_argument("--texto", required=True)
    ap.add_argument("--semillas", default="11,42,7")
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--pasos", type=int, default=20)
    ap.add_argument("--dispositivo", default="mps")
    ap.add_argument("--tipo", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--salida", default="/tmp/deriva-original")
    ap.add_argument("--etiqueta", default="orig")
    # palancas del usuario, para encenderlas de una en una
    ap.add_argument("--freno", type=float, default=0.0)
    ap.add_argument("--refuerzo", action="store_true")
    ap.add_argument("--depthwise", action="store_true")
    ap.add_argument("--sin-encoder", action="store_true")
    ap.add_argument("--compartir-emb", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--cebado", action="store_true")
    ap.add_argument("--mover-voz", action="store_true",
                    help="convierte el prefijo de voz al tipo del modelo (lo que hace voz_stream)")
    a = ap.parse_args()

    print(f"vibevoice desde: {vibevoice.__file__}", flush=True)
    os.makedirs(a.salida, exist_ok=True)
    guion = open(a.texto).read().strip()
    guion = guion.replace("’", "'").replace("“", '"').replace("”", '"')

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[a.tipo]

    print(f"cargando {a.modelo} en {a.dispositivo}/{a.tipo}...", flush=True)
    procesador = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, torch_dtype=dtype, attn_implementation="sdpa", device_map=None)
    modelo.to(a.dispositivo)
    modelo.eval()

    # --- palancas, en el MISMO orden que cargar_modelo() de voz_stream.py ---
    if a.sin_encoder:
        print("  [mod] codificador acustico liberado:", soltar_encoder(modelo), flush=True)
    if a.depthwise:
        print("  [mod] depthwise reescritas:", poner_depthwise(modelo), flush=True)
    if a.cebado:
        print("  [mod] decoder cebado:", poner_cebado(modelo), flush=True)
    if a.compartir_emb:
        print("  [mod] embeddings compartidos:", compartir_embeddings(modelo), flush=True)
    if a.freno > 0:
        poner_freno(modelo, a.freno)
        print(f"  [mod] freno de guia {a.freno}", flush=True)
    if a.refuerzo:
        poner_refuerzo(modelo)
        print("  [mod] guia reforzada en el arranque (4.5, 6 fotogramas)", flush=True)
    if a.int8:
        disponibles = [m for m in torch.backends.quantized.supported_engines if m != "none"]
        motor = "qnnpack" if "qnnpack" in disponibles else (disponibles or [None])[0]
        if motor:
            torch.backends.quantized.engine = motor
        torch.ao.quantization.quantize_dynamic(
            modelo, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        print(f"  [mod] int8 dinamico con motor {motor}", flush=True)

    modelo.set_ddpm_inference_steps(num_steps=a.pasos)
    print("escala:", float(modelo.model.speech_scaling_factor),
          " sesgo:", float(modelo.model.speech_bias_factor),
          " pasos:", modelo.ddpm_inference_steps, flush=True)

    # El demo oficial usa weights_only=True dentro de safe_globals, pero con
    # torch 2.13 el desempaquetador seguro no sabe reconstruir
    # BaseModelOutputWithPast y aborta. weights_only=False carga el MISMO
    # tensor; el .pt viene del propio repo de Microsoft.
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefijo = torch.load(a.voz, map_location=a.dispositivo, weights_only=False)
    if a.mover_voz:
        prefijo = a_dispositivo(prefijo, a.dispositivo, dtype)

    for s in [int(v) for v in a.semillas.split(",")]:
        print(f"\n=== {a.etiqueta}  semilla {s}  cfg {a.cfg}  pasos {a.pasos}  "
              f"{a.tipo} ===", flush=True)
        torch.manual_seed(s)
        np.random.seed(s)
        entradas = procesador.process_input_with_cached_prompt(
            text=guion, cached_prompt=copy.deepcopy(prefijo),
            padding=True, return_tensors="pt", return_attention_mask=True)
        for k, v in entradas.items():
            if torch.is_tensor(v):
                entradas[k] = v.to(a.dispositivo)
        t0 = time.time()
        with torch.no_grad():
            sal = modelo.generate(
                **entradas, max_new_tokens=None, cfg_scale=a.cfg,
                tokenizer=procesador.tokenizer,
                generation_config={"do_sample": False}, verbose=False,
                show_progress_bar=False,
                all_prefilled_outputs=copy.deepcopy(prefijo))
        seg = time.time() - t0
        x = sal.speech_outputs[0].float().cpu().numpy().reshape(-1)
        nombre = f"{a.etiqueta}-cfg{a.cfg}-p{a.pasos}-{a.tipo}-s{s}.wav"
        ruta = os.path.join(a.salida, nombre)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RITMO)
            w.writeframes(pcm.tobytes())
        dur = len(x) / RITMO
        print(f"  {dur:.1f}s de audio en {seg:.0f}s (RTF {seg / max(dur, 1e-9):.2f}) "
              f"-> {ruta}", flush=True)


if __name__ == "__main__":
    main()
