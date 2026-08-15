#!/usr/bin/env python
"""Baja UNA vez el audio de la sesion con respiro y lo guarda, para poder
comparar contra el sin volver a ocupar el modelo en cada iteracion."""
import argparse
import asyncio
import hashlib
import pathlib
import sys
import wave

AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent.parent))
from generar_base import FRASES  # noqa: E402
from ws_fidelidad import ws_sesion  # noqa: E402

RITMO = 24_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.2.54:8082")
    ap.add_argument("--token", required=True)
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--salida", default="servidor_respiro.wav")
    a = ap.parse_args()
    pcm, _, _ = asyncio.run(ws_sesion(a.url, a.token, FRASES, a.voz, 3.0,
                                      a.semilla, 6, respiro=True))
    destino = AQUI / a.salida
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes(pcm)
    print(f"{destino.name}: {len(pcm)/2/RITMO:.2f} s  "
          f"md5 {hashlib.md5(pcm).hexdigest()}")


if __name__ == "__main__":
    main()
