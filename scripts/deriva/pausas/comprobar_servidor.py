#!/usr/bin/env python
"""Ata la simulacion al servidor de verdad, por md5.

Sin esto todo lo demas es un ejercicio de numpy: se comprueba que
  1. la sesion con respiro=False da EXACTAMENTE los fotogramas de
     base_espacio.wav (o sea que las variantes salen del audio real), y
  2. la variante 01_actual_espejo es EXACTAMENTE lo que el servidor emite hoy
     con respiro=True (o sea que el defecto que se mide es el que se oye).

Uso:
  .venv/bin/python comprobar_servidor.py --token XXX [--variante 04_suelo_antes]
"""
import argparse
import asyncio
import hashlib
import pathlib
import sys
import wave

import numpy as np

AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent.parent))
from generar_base import FRASES  # noqa: E402
from ws_fidelidad import ws_sesion  # noqa: E402  (el cliente ya probado)

RITMO = 24_000


def sesion(url, token, frases, voz, semilla, respiro):
    pcm, _, _ = asyncio.run(ws_sesion(url, token, frases, voz, 3.0, semilla,
                                      6, respiro=respiro))
    return pcm


def pcm_de_wav(p):
    with wave.open(str(p), "rb") as w:
        return w.readframes(w.getnframes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.2.54:8082")
    ap.add_argument("--token", required=True)
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--variante", default="01_actual_espejo",
                    help="WAV local contra el que comparar el audio CON respiro")
    a = ap.parse_args()

    base = pcm_de_wav(AQUI / "base_espacio.wav")
    sin = sesion(a.url, a.token, FRASES, a.voz, a.semilla, False)
    print(f"sesion respiro=False  {len(sin)/2/RITMO:6.2f} s  "
          f"md5 {hashlib.md5(sin).hexdigest()}")
    print(f"/tts/stream ' '.join  {len(base)/2/RITMO:6.2f} s  "
          f"md5 {hashlib.md5(base).hexdigest()}")
    print("  ->", "IGUAL: las variantes salen del audio real"
          if hashlib.md5(sin).hexdigest() == hashlib.md5(base).hexdigest()
          else "DISTINTO: la base no vale, revisa voz/semilla")

    con = sesion(a.url, a.token, FRASES, a.voz, a.semilla, True)
    esperado = pcm_de_wav(AQUI / f"{a.variante}.wav")
    print(f"\nsesion respiro=True   {len(con)/2/RITMO:6.2f} s  "
          f"md5 {hashlib.md5(con).hexdigest()}")
    print(f"{a.variante:<21} {len(esperado)/2/RITMO:6.2f} s  "
          f"md5 {hashlib.md5(esperado).hexdigest()}")
    if hashlib.md5(con).hexdigest() == hashlib.md5(esperado).hexdigest():
        print("  -> IGUAL: el servidor hace exactamente esta variante")
        return 0
    x, y = np.frombuffer(con, "<i2"), np.frombuffer(esperado, "<i2")
    n = min(len(x), len(y))
    i = int(np.argmax(x[:n] != y[:n])) if (x[:n] != y[:n]).any() else n
    print(f"  -> DISTINTO: primera muestra distinta en {i} "
          f"({i/RITMO:.3f} s), largos {len(x)} vs {len(y)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
