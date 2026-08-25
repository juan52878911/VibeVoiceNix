#!/usr/bin/env python
"""Baja de la VM el material crudo para estudiar la pausa entre frases.

Dos ficheros, misma voz, misma semilla, mismo texto:

  base_espacio.wav   frases cosidas con ESPACIO. Es EXACTAMENTE la cadena de
                     fotogramas que ve el respiro (una sesion con respiro=False
                     da esto bit a bit), asi que a partir de aqui se pueden
                     reproducir TODAS las variantes de pausa sin volver a
                     generar: el habla no cambia entre variantes por
                     construccion.
  ref_saltolinea.wav frases cosidas con "\\n\\n". Aqui el modelo GENERA la pausa
                     el mismo. Es la referencia de "como suena una pausa suya"
                     para comparar espectro y envolvente.

No reproduce nada: solo escribe WAV.
"""
import argparse
import hashlib
import json
import pathlib
import time
import urllib.request

FRASES = [
    "El tren llega a las siete de la tarde.",
    "Manana por la mañana vamos al parque, si no llueve.",
    "No olvides comprar pan, leche y un par de huevos.",
    "La reunion se ha movido al jueves.",
    "Te dejo la llave debajo del felpudo, como siempre.",
    "Avisame cuando llegues a casa.",
]


def pedir(url, token, cuerpo, destino):
    pet = urllib.request.Request(
        url, method="POST", data=json.dumps(cuerpo).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    t0 = time.monotonic()
    with urllib.request.urlopen(pet, timeout=900) as r:
        datos = r.read()
    pared = time.monotonic() - t0
    destino.write_bytes(datos)
    pcm = len(datos) - 44
    print(f"{destino.name}: {pcm/2/24000:.2f} s de audio, {pared:.1f} s de "
          f"pared, RTF {pared/(pcm/2/24000):.3f}, "
          f"md5 {hashlib.md5(datos[44:]).hexdigest()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.168.2.54:8082")
    p.add_argument("--token", required=True)
    p.add_argument("--voz", default="sp-Spk1_man")
    p.add_argument("--semilla", type=int, default=11)
    p.add_argument("--salida", default=str(pathlib.Path(__file__).parent))
    a = p.parse_args()
    salida = pathlib.Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    comun = {"voz": a.voz, "semilla": a.semilla, "cfg_scale": 3.0}
    pedir(f"{a.url}/tts/stream", a.token,
          dict(comun, texto=" ".join(FRASES)), salida / "base_espacio.wav")
    pedir(f"{a.url}/tts/stream", a.token,
          dict(comun, texto="\n\n".join(FRASES)), salida / "ref_saltolinea.wav")


if __name__ == "__main__":
    main()
