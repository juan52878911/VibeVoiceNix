#!/usr/bin/env python
"""Genera una nota de voz lista para mandar por WhatsApp (o donde sea) con el servicio de la VM.

    python scripts/nota_voz.py --voz juan "Llego en diez minutos, id pidiendo."
    python scripts/nota_voz.py --voz isis --formato mp3 --salida saludo.mp3 "Feliz cumple."

  ogg  (defecto) Opus a 32 kbit/s en Ogg: es el formato de las NOTAS DE VOZ de WhatsApp,
       asi que el fichero se puede reenviar como nota. ~12 veces menos que el WAV.
  mp3  64 kbit/s: llega como FICHERO de audio adjunto, no como nota, pero lo abre todo.
  wav  el original, sin perdidas.

Solo usa la biblioteca estandar: pide POST /tts/stream con el campo `formato` y guarda
lo que llega. El token se lee de VOZ_TOKEN (o --token) y la URL de VOZ_STREAM_URL
(por defecto la VM, http://192.168.2.54:8082).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("texto", nargs="+", help="lo que tiene que decir")
    ap.add_argument("--voz", default="sp-Spk1_man")
    ap.add_argument("--formato", choices=["ogg", "mp3", "wav"], default="ogg")
    ap.add_argument("--salida", default=None, help="fichero de salida (por defecto nota-<voz>.<formato>)")
    ap.add_argument("--semilla", type=int, default=None, help="la misma semilla da el mismo audio")
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL", "http://192.168.2.54:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    a = ap.parse_args()

    texto = " ".join(a.texto)
    salida = a.salida or f"nota-{a.voz}.{a.formato}"
    cuerpo = {"texto": texto, "voz": a.voz, "cfg_scale": a.cfg, "formato": a.formato}
    if a.semilla is not None:
        cuerpo["semilla"] = a.semilla
    pet = urllib.request.Request(
        a.url.rstrip("/") + "/tts/stream", method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {a.token}"} if a.token else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(pet, timeout=900) as r, open(salida, "wb") as f:
            while True:
                bloque = r.read1(65536)
                if not bloque:
                    break
                f.write(bloque)
    except urllib.error.HTTPError as e:
        sys.exit(f"el servicio respondio {e.code}: {e.read().decode(errors='replace')[:300]}")
    tam = os.path.getsize(salida)
    print(f"{salida}: {tam / 1024:.0f} KB en {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
