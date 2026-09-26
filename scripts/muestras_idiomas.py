#!/usr/bin/env python
"""La voz del asistente presentandose en cada idioma: muestras para escuchar y comparar.

    python scripts/muestras_idiomas.py                       # es en fr de it pt, voz del perfil general
    python scripts/muestras_idiomas.py --idiomas es,en,fr    # solo esos
    python scripts/muestras_idiomas.py --stt http://voz:8080 # y transcribe cada clip con whisper

Pide cada frase a voz-stream (POST /tts/stream) con los ajustes del perfil `general` de
perfiles_asistente.json (sp-Spk1_man, semilla 17, cfg 3,5), asi que suena como el asistente.
Deja en --salida (por defecto muestras-idiomas/):

  <idioma>.wav        cada presentacion, 24 kHz
  todas.wav           las seis seguidas con 0,7 s de silencio entre medias
  index.html          reproductores, texto, primer sonido, RTF y, con --stt, lo que entiende whisper

Es una prueba de oido, no una puerta: el prefijo de la voz es espanol, y con prefijo espanol
el frances salia con un 43 % de WER y el aleman con un 22 % (docs/plan-mejora-modelo.md). Lo
que se oiga aqui es el punto de partida del video del post 3; el acento nativo lo pone la
conversion de dobla (--acento nativo), no esta prueba.

Solo biblioteca estandar. Token en VOZ_TOKEN (o --token) y URL en VOZ_STREAM_URL (por defecto
la VM, http://192.168.2.54:8082). Los textos no llevan cifras a proposito: el normalizador del
servidor solo sabe es y en.
"""
import argparse
import html
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

FRASES = {
    "es": ("Español", "Hola, soy la voz del asistente de casa. No soy una persona: "
                      "me genera un procesador pequeño, sin tarjeta gráfica."),
    "en": ("English", "Hi, I'm the voice of the home assistant. I'm not a person: "
                      "a small processor generates me, with no graphics card."),
    "fr": ("Français", "Bonjour, je suis la voix de l'assistant de la maison. Je ne suis pas "
                       "une personne : un petit processeur me génère, sans carte graphique."),
    "de": ("Deutsch", "Hallo, ich bin die Stimme des Hausassistenten. Ich bin kein Mensch: "
                      "Ein kleiner Prozessor erzeugt mich, ganz ohne Grafikkarte."),
    "it": ("Italiano", "Ciao, sono la voce dell'assistente di casa. Non sono una persona: "
                       "mi genera un piccolo processore, senza scheda grafica."),
    "pt": ("Português", "Olá, eu sou a voz do assistente de casa. Não sou uma pessoa: "
                        "um pequeno processador me gera, sem placa de vídeo."),
}
RITMO = 24000
SILENCIO_S = 0.7


def perfil(nombre):
    """Voz, semilla y cfg del perfil, leidos del JSON del repo para no duplicarlos."""
    ruta = Path(__file__).resolve().parent.parent / "perfiles_asistente.json"
    try:
        v = json.loads(ruta.read_text())["perfiles"][nombre]["voz"]
        return v["voz"], v.get("semilla"), v.get("cfg_scale", 3.5)
    except (OSError, KeyError, ValueError):
        return "sp-Spk1_man", 17, 3.5


def pedir(url, token, cuerpo):
    pet = urllib.request.Request(
        url.rstrip("/") + "/tts/stream", method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    t0 = time.perf_counter()
    primero = None
    trozos = []
    with urllib.request.urlopen(pet, timeout=900) as r:
        while True:
            b = r.read1(65536)
            if not b:
                break
            if primero is None:
                primero = time.perf_counter() - t0
            trozos.append(b)
    return b"".join(trozos), primero or 0.0, time.perf_counter() - t0


def pcm_de_wav(datos):
    """PCM de un WAV que puede venir en streaming (tamanos de cabecera sin rellenar)."""
    i = datos.find(b"data")
    if i < 0:
        raise ValueError("la respuesta no es un WAV")
    return datos[i + 8:]


def escribir_wav(ruta, pcm):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes(pcm)


def transcribir(url, token, ruta, idioma):
    """POST /stt de voz-api (multipart a mano, sin dependencias)."""
    frontera = uuid.uuid4().hex
    cuerpo = io.BytesIO()
    for nombre, valor in (("idioma", idioma),):
        cuerpo.write(f"--{frontera}\r\nContent-Disposition: form-data; name=\"{nombre}\"\r\n\r\n"
                     f"{valor}\r\n".encode())
    cuerpo.write(f"--{frontera}\r\nContent-Disposition: form-data; name=\"archivo\"; "
                 f"filename=\"{ruta.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    cuerpo.write(ruta.read_bytes())
    cuerpo.write(f"\r\n--{frontera}--\r\n".encode())
    pet = urllib.request.Request(
        url.rstrip("/") + "/stt", method="POST", data=cuerpo.getvalue(),
        headers={"content-type": f"multipart/form-data; boundary={frontera}",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(pet, timeout=300) as r:
        return json.loads(r.read()).get("texto", "").strip()


def pagina(filas, voz, semilla, cfg):
    cuerpo = "".join(f"""
<section>
  <h2>{html.escape(f['nombre'])} <small>{f['idioma']}</small></h2>
  <audio controls preload="none" src="{f['idioma']}.wav"></audio>
  <p class="texto">{html.escape(f['texto'])}</p>
  {f"<p class='stt'>whisper oye: «{html.escape(f['stt'])}»</p>" if f.get('stt') else ''}
  <p class="datos">{f['dur']:.1f} s de audio · primer sonido {f['primero']:.2f} s · RTF {f['rtf']:.2f}</p>
</section>""" for f in filas)
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Idiomas del asistente</title>
<style>
:root {{ --bg:#0c0218; --tx:#f4ecff; --suave:#b79bd6; --acento:#ff4f9a; --frio:#7fd4ff; --caja:#170830; }}
body {{ margin:0; background:var(--bg); color:var(--tx); font:16px/1.5 system-ui,sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} .sub {{ color:var(--suave); margin:0 0 24px; }}
section {{ background:var(--caja); border-radius:12px; padding:16px 18px; margin:14px 0; }}
h2 {{ margin:0 0 10px; font-size:20px; }} h2 small {{ color:var(--acento); font:600 13px monospace; margin-left:6px; }}
audio {{ width:100%; }} .texto {{ margin:10px 0 4px; }}
.stt {{ color:var(--frio); margin:4px 0; font-style:italic; }}
.datos {{ color:var(--suave); font:13px monospace; margin:6px 0 0; }}
</style></head><body><main>
<h1>La voz del asistente, idioma a idioma</h1>
<p class="sub">{html.escape(voz)} · semilla {semilla} · cfg {cfg} · generado con voz-stream.
<a style="color:var(--frio)" href="todas.wav">Todas seguidas</a></p>
{cuerpo}
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--idiomas", default=",".join(FRASES), help="lista separada por comas")
    ap.add_argument("--perfil", default="general", help="perfil de perfiles_asistente.json")
    ap.add_argument("--voz", default=None, help="manda sobre la voz del perfil")
    ap.add_argument("--salida", default="muestras-idiomas")
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL", "http://192.168.2.54:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--stt", default=None, help="URL de voz-api para transcribir cada clip (opcional)")
    a = ap.parse_args()

    voz, semilla, cfg = perfil(a.perfil)
    voz = a.voz or voz
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    silencio = b"\x00\x00" * int(RITMO * SILENCIO_S)
    filas, todas = [], []

    for idioma in [x.strip() for x in a.idiomas.split(",") if x.strip()]:
        if idioma not in FRASES:
            sys.exit(f"idioma sin frase: {idioma} (hay {', '.join(FRASES)})")
        nombre, texto = FRASES[idioma]
        cuerpo = {"texto": texto, "voz": voz, "cfg_scale": cfg, "formato": "wav",
                  # el normalizador solo sabe es/en; en el resto, el texto tal cual
                  "normalizar": idioma if idioma in ("es", "en") else "no"}
        if semilla is not None:
            cuerpo["semilla"] = semilla
        try:
            datos, primero, total = pedir(a.url, a.token, cuerpo)
        except urllib.error.HTTPError as e:
            sys.exit(f"{idioma}: el servicio respondio {e.code}: "
                     f"{e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as e:
            sys.exit(f"{idioma}: no llego a {a.url} ({e.reason}). ¿VOZ_STREAM_URL?")
        pcm = pcm_de_wav(datos)
        ruta = salida / f"{idioma}.wav"
        escribir_wav(ruta, pcm)
        dur = len(pcm) / 2 / RITMO
        fila = {"idioma": idioma, "nombre": nombre, "texto": texto, "dur": dur,
                "primero": primero, "rtf": total / dur if dur else 0.0}
        if a.stt:
            try:
                fila["stt"] = transcribir(a.stt, a.token, ruta, idioma)
            except (urllib.error.URLError, ValueError) as e:
                print(f"  {idioma}: sin transcripcion ({e})", file=sys.stderr)
        filas.append(fila)
        todas += [pcm, silencio]
        print(f"{idioma}  {dur:5.1f} s  primer sonido {primero:.2f} s  RTF {fila['rtf']:.2f}"
              + (f"  «{fila['stt']}»" if fila.get("stt") else ""))

    escribir_wav(salida / "todas.wav", b"".join(todas[:-1]))
    (salida / "index.html").write_text(pagina(filas, voz, semilla, cfg), encoding="utf-8")
    print(f"\nlisto: abre {salida / 'index.html'}")


if __name__ == "__main__":
    main()
