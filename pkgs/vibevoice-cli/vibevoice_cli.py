"""CLI de inferencia de VibeVoice, con cuantización y pasos de difusión.

Sustituye al `demo/realtime_model_inference_from_file.py` del repo original,
que se venía parcheando con `substitute` para tres cosas distintas (ruta de
voces, weights_only, y ningún control de cuantización). Un fichero propio es
menos frágil que tres sustituciones de texto sobre código ajeno.

Los valores por defecto salen de medir en un i7-8700T (6 núcleos, DDR4 de un
solo canal, 17,2 GB/s reales). La carga está limitada por ancho de banda de
memoria, no por cómputo, y eso decide todo lo demás:

  fp32, 20 pasos    RTF 5,39     <- configuración original
  int8, 20 pasos    RTF 2,75
  int8,  6 pasos    RTF 2,18     <- por defecto aquí
  int8,  4 pasos    RTF 2,11     <- solo un 3% mejor: no compensa el riesgo

La cuantización int8 gana porque divide por cuatro los BYTES de peso que hay
que traer de RAM. Sorprende en una CPU sin VNNI, donde la multiplicación int8
va emulada, pero el ahorro de ancho de banda se impone sobre esa penalización.

int4 daría otro factor dos sobre el papel, pero en torchao 0.18 no hay ningún
camino de int4 para CPU: dos formatos exigen la librería `mslk`, otro dice
"does not support device 'cpu' yet" y el cuarto acaba pidiendo CUDA.
"""

import argparse
import copy
import os
import time
import wave
from pathlib import Path

import numpy as np
import torch


def cargar(ruta_modelo: str, cuantizar: bool):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    procesador = VibeVoiceStreamingProcessor.from_pretrained(ruta_modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        ruta_modelo,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    modelo.eval()

    if cuantizar:
        # inplace=True es OBLIGATORIO: sin él se crea una copia completa del
        # modelo y el pico pasa de 4,7 GB, que en una VM de 5 GB acaba en OOM.
        torch.ao.quantization.quantize_dynamic(
            modelo, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )

    return procesador, modelo


def generar(procesador, modelo, texto, ruta_voz, pasos, cfg_scale, destino: Path):
    modelo.set_ddpm_inference_steps(pasos)

    # weights_only=False: el .pt guarda un BaseModelOutputWithPast, subclase de
    # OrderedDict, y el desempaquetador seguro de torch >=2.6 solo admite
    # dict/OrderedDict exactos. El fichero viene del repo oficial de Microsoft.
    prefijo = torch.load(ruta_voz, map_location="cpu", weights_only=False)

    entradas = procesador.process_input_with_cached_prompt(
        text=texto,
        cached_prompt=prefijo,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    ini = time.perf_counter()
    with torch.no_grad():
        salida = modelo.generate(
            **entradas,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=procesador.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            # deepcopy porque generate() MUTA el prefijo: sin la copia, una
            # segunda llamada en el mismo proceso produce audio distinto.
            all_prefilled_outputs=copy.deepcopy(prefijo),
        )
    computo = time.perf_counter() - ini

    audio = salida.speech_outputs[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio).reshape(-1)

    ritmo = 24000
    destino.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ritmo)
        w.writeframes(pcm.tobytes())

    return len(audio) / ritmo, computo


def main():
    p = argparse.ArgumentParser(
        description="Genera voz con VibeVoice-Realtime-0.5B en CPU."
    )
    p.add_argument("--texto", help="Fichero con el texto. Si falta, se lee de la entrada estándar.")
    p.add_argument("--salida", default="salida.wav", help="WAV de destino.")
    p.add_argument("--voz", help="Nombre del hablante, p. ej. sp-Spk1_man.")
    p.add_argument("--pasos", type=int, help="Pasos de difusión.")
    p.add_argument("--cfg-scale", type=float, default=1.5,
                   help="Afecta a la expresividad, NO a la velocidad (medido).")
    p.add_argument("--sin-cuantizar", action="store_true",
                   help="Usa fp32. Más lento y sin mejora audible.")
    p.add_argument("--listar-voces", action="store_true")
    args = p.parse_args()

    modelo_dir = os.environ["VIBEVOICE_MODELO"]
    voces_dir = Path(os.environ["VIBEVOICE_VOCES"])

    if args.listar_voces:
        for v in sorted(p.stem for p in voces_dir.glob("*.pt")):
            print(v)
        return

    voz = args.voz or os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man")
    ruta_voz = voces_dir / f"{voz}.pt"
    if not ruta_voz.exists():
        disponibles = ", ".join(sorted(p.stem for p in voces_dir.glob("*.pt")))
        raise SystemExit(f"voz '{voz}' no encontrada.\nDisponibles: {disponibles}")

    pasos = args.pasos or int(os.environ.get("VIBEVOICE_PASOS", "6"))
    cuantizar = not args.sin_cuantizar

    texto = (
        Path(args.texto).read_text(encoding="utf-8").strip()
        if args.texto
        else __import__("sys").stdin.read().strip()
    )
    if not texto:
        raise SystemExit("no hay texto que sintetizar")

    procesador, modelo = cargar(modelo_dir, cuantizar)
    dur, computo = generar(
        procesador, modelo, texto, ruta_voz, pasos, args.cfg_scale, Path(args.salida)
    )

    print(f"voz         {voz}")
    print(f"precision   {'int8' if cuantizar else 'fp32'}  ({pasos} pasos)")
    print(f"audio       {dur:.2f} s")
    print(f"computo     {computo:.2f} s")
    print(f"RTF         {computo / dur:.2f}" if dur else "RTF         n/d")
    print(f"salida      {args.salida}")


if __name__ == "__main__":
    main()
