#!/usr/bin/env python
"""Di una frase con cualquiera de las voces instaladas y guardala en un WAV.

    python scripts/decir.py --voz santiago --texto "Lo que quieras que diga."

Existe para experimentar sin levantar el servicio. `voz-stream-mac.sh` es el
camino de produccion —sesiones, streaming, WebSocket— pero carga el modelo cada
vez que arranca y no deja variar `cfg_scale` ni la semilla desde la linea de
ordenes, que es justo lo que hace falta cuando se esta afinando una voz clonada.

    --lista                muestra las voces disponibles y sale
    --cfg 3.0              guia del CFG. 3,0 es el defecto MEDIDO; 3,5 no abre
                           el recorrido tonal una vez se corrigen las octavas,
                           y 4,5 aplana la melodia
    --semilla 7            fija el ruido de la difusion. Sin ella cada llamada
                           suena distinta: el recorrido tonal varia varios
                           semitonos entre semillas, asi que para COMPARAR algo
                           hay que fijarla
    --pasos 10             pasos de difusion. Por encima de 8 no se gana nada
                           audible y cuesta un 26 % mas

El modelo tarda ~40 s en cargar y luego va a RTF ~1 en un M4, asi que si vas a
generar varias frases pasalas todas de una vez con --texto repetido: se carga
una sola vez.
"""
import argparse
import copy
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

RITMO = 24000


def escribir_wav(ruta, x):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz", default="sp-Spk1_man", help="nombre del .pt, sin extension")
    ap.add_argument("--texto", action="append", default=[],
                    help="frase a decir; se puede repetir para generar varias")
    ap.add_argument("--salida", default="salida", help="prefijo de los WAV de salida")
    ap.add_argument("--verificar", action="store_true",
                    help="transcribe lo generado y avisa si no dice lo que pediste")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--semilla", type=int, default=None)
    ap.add_argument("--pasos", type=int, default=10)
    ap.add_argument("--voces", default=os.environ.get(
        "VIBEVOICE_VOCES", str(Path.home() / ".cache/vibevoice-nix/voces")))
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--lista", action="store_true", help="lista las voces y sale")
    args = ap.parse_args()

    dir_voces = Path(args.voces)
    disponibles = sorted(p.stem for p in dir_voces.glob("*.pt"))
    if args.lista:
        propias = [v for v in disponibles if not v[:3] in
                   ("sp-", "en-", "de-", "fr-", "it-", "jp-", "kr-", "nl-", "pl-", "pt-", "in-")]
        print(f"propias ({len(propias)}): {', '.join(propias) or '(ninguna)'}")
        print(f"de fabrica ({len(disponibles)-len(propias)}): "
              f"{', '.join(v for v in disponibles if v not in propias)}")
        return 0
    if args.voz not in disponibles:
        print(f"no existe la voz '{args.voz}'. Disponibles: {', '.join(disponibles)}",
              file=sys.stderr)
        return 2
    if not args.texto:
        print("hace falta al menos un --texto", file=sys.stderr)
        return 2

    disp = args.dispositivo
    if disp == "auto":
        disp = "mps" if torch.backends.mps.is_available() else "cpu"

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    print(f"cargando el modelo en {disp}...", flush=True)
    proc = VibeVoiceStreamingProcessor.from_pretrained(args.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.modelo, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    # La tabla de embeddings del tts_lm nunca se usa y el modelo preparado la
    # trae podada; se apunta a la del otro LM, como en voz_stream.py.
    modelo.model.tts_language_model.embed_tokens = modelo.model.language_model.embed_tokens
    modelo.set_ddpm_inference_steps(args.pasos)
    modelo.to(disp)

    prefijo = torch.load(dir_voces / f"{args.voz}.pt", weights_only=False, map_location=disp)
    escritos = []

    for i, texto in enumerate(args.texto, 1):
        if not texto.endswith("\n"):
            texto += "\n"
        if args.semilla is not None:
            torch.manual_seed(args.semilla)
        entradas = proc.process_input_with_cached_prompt(
            text=texto, cached_prompt=copy.deepcopy(prefijo),
            padding=True, return_tensors="pt", return_attention_mask=True)
        entradas = {k: (v.to(disp) if torch.is_tensor(v) else v) for k, v in entradas.items()}
        ini = time.perf_counter()
        with torch.no_grad():
            salida = modelo.generate(
                **entradas, max_new_tokens=None, cfg_scale=args.cfg,
                tokenizer=proc.tokenizer, generation_config={"do_sample": False},
                verbose=False, return_speech=True,
                all_prefilled_outputs=copy.deepcopy(prefijo))
        x = salida.speech_outputs[0].detach().float().cpu().numpy().reshape(-1)
        proc_s = time.perf_counter() - ini
        # SIEMPRE con sufijo, aunque solo haya un texto. Antes con un solo
        # --texto salia "salida.wav" y con varios "salida-1.wav": si reutilizabas
        # el mismo --salida con distinto numero de frases, quedaban ficheros de
        # la tanda ANTERIOR con nombres que parecian de esta. Se confunde
        # facilisimo cual es cual, y el sintoma es abrir un audio que no dice lo
        # que pediste.
        ruta = Path(f"{args.salida}-{i}.wav").resolve()
        escribir_wav(ruta, x)
        escritos.append((ruta, texto.strip()))
        print(f"  {ruta}", flush=True)
        print(f"    {len(x)/RITMO:5.2f} s  RTF {proc_s/(len(x)/RITMO):.2f}  \"{texto.strip()[:60]}\"",
              flush=True)

    if args.verificar:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("\n[aviso] --verificar necesita faster-whisper; no se comprueba nada")
            return 0
        print("\ncomprobando que dicen lo que pediste...")
        w = WhisperModel("base", device="cpu", compute_type="int8")
        for ruta, pedido in escritos:
            seg, _ = w.transcribe(str(ruta), language="es", beam_size=5)
            dicho = " ".join(t.text for t in seg).strip()
            # comparacion tosca a proposito: solo queremos cazar "esto no es ni
            # parecido", no medir WER. Para eso esta scripts/banco_clonado.py.
            a = set(pedido.lower().split()); b = set(dicho.lower().split())
            solapa = len(a & b) / max(1, len(a))
            marca = "OK " if solapa > 0.4 else "OJO"
            print(f"  {marca} {ruta.name}: \"{dicho[:70]}\"")
            if solapa <= 0.4:
                print(f"      pedido: \"{pedido[:70]}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
