#!/usr/bin/env python
"""Sondeo de los limites del sintetizador: que sabe hacer y que se inventa.

    python scripts/sondeo_voz.py
    python scripts/sondeo_voz.py --caso largo --guardar /tmp/clips

No mide velocidad -- para eso esta scripts/fidelidad.py. Aqui se mira QUE
sale cuando se le pide algo para lo que no fue pensado:

  varios      dos hablantes en un mismo texto (formato `Speaker 0:`)
  idiomas     espanol e ingles mezclados en la misma frase
  largo       frases muy largas, para ver si el volumen se dispara al final
  sonidos     onomatopeyas y ruidos, no palabras
  canto       peticion explicita de cantar

LO QUE SE MIDE DE CADA CLIP
El volumen se parte en diez tramos y se comparan: si el ultimo tramo suena
mucho mas fuerte que el cuerpo, es que el modelo se esta desbocando al final.
La cola es lo que suena DESPUES de la ultima palabra -- si es larga, el modelo
sigue generando audio cuando ya no queda texto, que es de donde salen los
ruidos de fondo. Y la transcripcion dice si lo que suena es lo que se pidio.
"""
import argparse
import json
import os
import struct
import sys
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelidad import transcribir  # noqa: E402

RITMO = 24000

CASOS = {
    "varios": [
        ("dos hablantes",
         "Speaker 0: Buenos dias, gracias por venir.\n"
         "Speaker 1: Un placer estar aqui contigo."),
        ("dos hablantes sin marca",
         "-- ¿Como va el despliegue? -- Va bien, ya esta en produccion."),
    ],
    "idiomas": [
        ("mezcla en una frase",
         "El deployment esta running en el server, pero el latency sigue high."),
        ("frase entera en ingles",
         "The deployment is running fine and the latency looks good today."),
        ("cambio a mitad",
         "Todo listo por aqui. Now let me switch to English for a moment."),
    ],
    "largo": [
        ("corta", "El servicio ya esta arriba."),
        ("media",
         "El servicio ya esta arriba y funcionando con normalidad desde "
         "primera hora de la manana."),
        ("larga",
         "El servicio ya esta arriba y funcionando con normalidad desde "
         "primera hora de la manana, con todos los motores respondiendo "
         "dentro de los margenes previstos y sin ninguna incidencia "
         "reseñable en los registros del sistema."),
        ("muy larga",
         "El servicio ya esta arriba y funcionando con normalidad desde "
         "primera hora de la manana, con todos los motores respondiendo "
         "dentro de los margenes previstos y sin ninguna incidencia "
         "reseñable en los registros del sistema, lo cual confirma que el "
         "despliegue de anoche salio exactamente como estaba planificado y "
         "que no hace falta revertir ninguno de los cambios que se "
         "introdujeron en la ultima version del servicio de voz."),
    ],
    "sonidos": [
        ("risa", "Ja, ja, ja. Que bueno."),
        ("onomatopeya", "El reloj hacia tic, tac, tic, tac toda la noche."),
        ("no verbal", "Mmm... ehhh... a ver, dejame pensar."),
    ],
    "canto": [
        ("peticion de cantar", "Voy a cantar: la, la, la, la, la, la, la."),
        ("cumpleanos", "Cumpleanos feliz, cumpleanos feliz, te deseamos todos."),
    ],
}


def sintetizar_a_wav(texto, ruta, url, token, voz, cfg):
    """Pide el clip entero y lo deja en un WAV. Devuelve las muestras."""
    cab = {"content-type": "application/json"}
    if token:
        cab["authorization"] = f"Bearer {token}"
    pet = urllib.request.Request(
        f"{url}/tts/stream", method="POST", headers=cab,
        data=json.dumps({"texto": texto, "voz": voz, "cfg_scale": cfg}).encode())
    crudo = urllib.request.urlopen(pet, timeout=600).read()
    # La respuesta trae cabecera RIFF de longitud desconocida; se busca el
    # trozo 'data' en vez de asumir 44 bytes, que no siempre es cierto.
    i = crudo.find(b"data")
    pcm = crudo[i + 8:] if i >= 0 else crudo
    pcm = pcm[:len(pcm) // 2 * 2]
    muestras = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        w.writeframes(pcm)
    return muestras


def medir(x):
    """Volumen por tramos, cola de silencio y recorte."""
    if x.size == 0:
        return {"seg": 0.0, "tramos": [], "final_vs_cuerpo": 0.0,
                "cola_s": 0.0, "recorte": 0.0}
    tramos = [float(np.sqrt(np.mean(t ** 2))) for t in np.array_split(x, 10) if t.size]
    cuerpo = float(np.median(tramos[:-1])) if len(tramos) > 1 else tramos[0]
    # Umbral relativo al pico: lo que baja de ahi se considera silencio.
    umbral = float(np.abs(x).max()) * 0.02
    sonoro = np.flatnonzero(np.abs(x) > umbral)
    cola = (x.size - 1 - sonoro[-1]) / RITMO if sonoro.size else x.size / RITMO
    return {
        "seg": x.size / RITMO,
        "tramos": tramos,
        # Cuanto mas fuerte suena el ultimo decimo que el cuerpo del clip.
        "final_vs_cuerpo": (tramos[-1] / cuerpo) if cuerpo > 0 else 0.0,
        "cola_s": float(cola),
        "recorte": float(np.mean(np.abs(x) >= 0.999)),
    }


def tono(x, ritmo=RITMO):
    """Tono fundamental por ventanas, con autocorrelacion.

    Sirve para responder a "¿canta?" sin oirlo: al hablar, el tono se mueve de
    forma continua y en un margen estrecho; al cantar salta entre notas y
    recorre bastante mas de una octava. Se devuelve el recorrido en semitonos,
    que es la unidad en la que esa diferencia se ve.
    """
    ventana, salto = 1024, 256
    minimo, maximo = ritmo // 400, ritmo // 60      # 60-400 Hz, voz humana
    f0 = []
    for i in range(0, len(x) - ventana, salto):
        t = x[i:i + ventana]
        if np.sqrt(np.mean(t ** 2)) < 0.01:         # silencio: no hay tono
            continue
        t = t - t.mean()
        r = np.correlate(t, t, mode="full")[ventana - 1:]
        if r[0] <= 0:
            continue
        pico = int(np.argmax(r[minimo:maximo])) + minimo
        # Sin un pico claro no es sonido tonal: se descarta en vez de inventar.
        if r[pico] / r[0] > 0.3:
            f0.append(ritmo / pico)
    if len(f0) < 5:
        return {"hz": 0.0, "semitonos": 0.0, "tonal": 0.0}
    f0 = np.array(f0)
    bajo, alto = np.percentile(f0, [5, 95])
    return {"hz": float(np.median(f0)),
            "semitonos": float(12 * np.log2(alto / bajo)) if bajo > 0 else 0.0,
            "tonal": len(f0) * salto / len(x)}


def barra(tramos):
    """Diez bloques, altura proporcional al volumen de cada tramo."""
    if not tramos:
        return ""
    alto = max(tramos) or 1.0
    escala = " ▁▂▃▄▅▆▇█"
    return "".join(escala[min(8, int(t / alto * 8))] for t in tramos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caso", choices=sorted(CASOS) + ["todos"], default="todos")
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL", "http://127.0.0.1:8080"),
                    help="para transcribir con whisper; vacio para no transcribir")
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    # El sintetizador puede estar en local y whisper en la VM, y cada uno
    # tiene su propio token. Por defecto se reutiliza el mismo.
    ap.add_argument("--token-api", default=os.environ.get("VOZ_API_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--guardar", default="/tmp/sondeo-voz")
    a = ap.parse_args()

    os.makedirs(a.guardar, exist_ok=True)
    grupos = sorted(CASOS) if a.caso == "todos" else [a.caso]

    for grupo in grupos:
        print(f"\n=== {grupo} ===")
        for nombre, texto in CASOS[grupo]:
            ruta = os.path.join(a.guardar, f"{grupo}-{nombre.replace(' ', '_')}.wav")
            try:
                x = sintetizar_a_wav(texto, ruta, a.url, a.token, a.voz, a.cfg)
            except Exception as e:
                print(f"  {nombre:22s} FALLO {type(e).__name__}: {e}")
                continue
            m = medir(x)
            tn = tono(x)
            dicho = ""
            if a.api_url:
                try:
                    dicho = transcribir(ruta, a.api_url, a.token_api or a.token).strip()
                except Exception as e:
                    dicho = f"<sin transcribir: {type(e).__name__}>"
            print(f"  {nombre:22s} {m['seg']:5.1f}s  {barra(m['tramos'])}  "
                  f"final x{m['final_vs_cuerpo']:.2f}  cola {m['cola_s']:.2f}s"
                  f"{'  RECORTA' if m['recorte'] > 0.001 else ''}\n"
                  f"  {'':22s} tono {tn['hz']:5.0f} Hz  recorrido {tn['semitonos']:4.1f} semitonos")
            if dicho:
                print(f"  {'':22s} oido: {dicho[:96]}")
    print(f"\nclips en {a.guardar}")


if __name__ == "__main__":
    main()
