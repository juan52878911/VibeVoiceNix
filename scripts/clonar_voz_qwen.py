#!/usr/bin/env python
"""Fabrica una voz para Qwen3-TTS a partir de una grabacion.

    python scripts/clonar_voz_qwen.py --audio nota.opus \
        --transcripcion "lo que dice la nota" --nombre juan --salida voces/

Deja en --salida tres ficheros con el mismo nombre:

    juan.wav      la referencia limpia a 24 kHz mono (recortada a --max s)
    juan.txt      la transcripcion literal
    juan.qvoice   el perfil del motor C (x-vector + injerto ICL, ~25 MB)
    juan.bin      solo el x-vector (8 KB): la voz "limpia", sin la sala de
                  la grabacion; es lo que el servidor carga por defecto

El wav + txt son la voz para el PyTorch de referencia (modo en contexto, que
clona mejor que el x-vector solo); el .qvoice/.bin son la voz para el motor C.
Se guardan los dos formatos porque el banco y el lote usan PyTorch y el
servidor de la VM usa el motor C.

No hace falta torch ni el modelo de 5 GB de VibeVoice: cabe en la VM. Los
avisos de calidad (recorte, silencio, 16 kHz reescalado) y la homogeneidad
entre clips son los de clonar_voz.py, que ya se validaron con numeros.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clonar_voz import RITMO, avisar_calidad, avisar_heterogeneos, igualar_volumen, leer_audio  # noqa: E402
from banco_motores import escribir_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", nargs="+", required=True, help="una o varias grabaciones")
    ap.add_argument("--transcripcion", required=True,
                    help="texto literal de la grabacion (de todas, en orden, si son varias)")
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--salida", default=os.environ.get("QWEN3TTS_VOCES", "voces"))
    ap.add_argument("--max", type=float, default=30.0,
                    help="segundos de referencia; el motor C usa hasta 30 por defecto")
    ap.add_argument("--idioma", default="Spanish")
    ap.add_argument("--bin", default=os.environ.get("QWEN3TTS_BIN", "qwen_tts"))
    ap.add_argument("--modelo", default=os.environ.get("QWEN3TTS_MODELO"))
    ap.add_argument("--sin-motor-c", action="store_true",
                    help="solo wav + txt (no hay motor C a mano)")
    args = ap.parse_args()

    clips = []
    for ruta in args.audio:
        print(f"leyendo {ruta}")
        x = leer_audio(ruta)
        avisar_calidad(x, sangria="  ")
        clips.append(x)
    avisar_heterogeneos(clips, args.audio)
    clips = igualar_volumen(clips)
    hueco = np.zeros(int(0.2 * RITMO), np.float32)
    ref = np.concatenate([c for par in zip(clips, [hueco] * len(clips)) for c in par])
    ref = ref[: int(args.max * RITMO)]

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    wav = salida / f"{args.nombre}.wav"
    escribir_wav(wav, ref)
    (salida / f"{args.nombre}.txt").write_text(args.transcripcion.strip() + "\n")
    print(f"referencia: {wav} ({len(ref) / RITMO:.1f} s) + {args.nombre}.txt")

    if args.sin_motor_c:
        return
    if not args.modelo or not shutil.which(args.bin):
        raise SystemExit("hace falta --bin (qwen_tts) y --modelo (directorio del 0.6B-Base); "
                         "o --sin-motor-c para dejar solo wav + txt")
    for extra, fichero in (([], f"{args.nombre}.qvoice"),
                           (["--xvector-only"], f"{args.nombre}.bin")):
        orden = [args.bin, "-d", args.modelo, "--ref-audio", str(wav), "-l", args.idioma,
                 "--voice-name", args.nombre, "--save-voice", str(salida / fichero), *extra,
                 "--text", "Hola.", "-o", os.devnull]
        r = subprocess.run(orden, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:])
            raise SystemExit(f"el motor C fallo creando {fichero}")
        print(f"voz: {salida / fichero} ({(salida / fichero).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
