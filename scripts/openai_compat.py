#!/usr/bin/env python
"""Comprueba que `/v1/audio/speech` es de verdad compatible con OpenAI.

    python scripts/openai_compat.py --base http://voz:8080 --token "$VOZ_TOKEN"

QUE COMPRUEBA, Y POR QUE ESO
"Compatible" no es que devuelva 200: es que un cliente escrito contra OpenAI
funcione sin tocar una linea. Las cuatro cosas que rompen eso en la practica y
que aqui se miran una por una:

  formatos   los seis que define OpenAI, y no basta con el codigo de estado:
             se comprueba la FIRMA de cada fichero. Un mp3 que empieza por
             'RIFF' es un 200 que suena a nada.
  wav sano   ffmpeg escribe en una tuberia y deja los tamanos del RIFF sin
             rellenar (0xFFFFFFFF). Se comprueba abriendolo con `wave`, que es
             lo que hara la mitad de los clientes.
  velocidad  `speed` de OpenAI es RAPIDEZ y `length_scale` de Piper es
             DURACION: van invertidos. Si alguien "arregla" esa inversion, aqui
             se ve porque speed=2 dejaria de durar la mitad.
  errores    el SDK construye sus excepciones leyendo `error.message`. Con el
             `{"detail": ...}` de FastAPI el usuario ve "unknown error".

Con el paquete `openai` instalado hace ademas la prueba de fuego -- el SDK
oficial contra este servidor. Sin el, se la salta y lo dice.

Solo stdlib: esto tiene que poder correrse desde la propia VM sin instalar nada.
"""
import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave

FIRMAS = {
    "mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    "opus": (b"OggS",),
    "aac": (b"\xff\xf1", b"\xff\xf9"),
    "flac": (b"fLaC",),
    "wav": (b"RIFF",),
}

TEXTO = "El backup de anoche termino sin errores."

fallos = []


def ok(cond, que, detalle=""):
    print(f"  {'OK   ' if cond else 'FALLA'}  {que}" + (f"  [{detalle}]" if detalle else ""))
    if not cond:
        fallos.append(que)
    return cond


def pedir(base, token, ruta, cuerpo=None):
    """Devuelve (codigo, cabeceras, bytes). No lanza en los 4xx/5xx.

    Las cabeceras se devuelven tal cual (`email.message.Message`) y NO como
    dict: uvicorn las manda en minusculas y un dict haria que `cab["X-Motor"]`
    fuese None. El Message busca sin distinguir mayusculas, que es lo que dice
    el RFC.
    """
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(base + ruta, data=datos, method="POST" if datos else "GET")
    if datos:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def duracion(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as wf:
        return wf.getnframes() / wf.getframerate()


def habla(base, token, **campos):
    cuerpo = {"model": "tts-1", "input": TEXTO}
    cuerpo.update(campos)
    return pedir(base, token, "/v1/audio/speech", cuerpo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("VOZ_API_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"\n== catalogo ({base}) ==")
    cod, _, cuerpo = pedir(base, args.token, "/v1/models")
    if not ok(cod == 200, "GET /v1/models responde", f"codigo {cod}"):
        print("\nNo hay servidor donde mirar. Levanta voz-api y repite.")
        return 1
    modelos = [m["id"] for m in json.loads(cuerpo)["data"]]
    ok("tts-1" in modelos, "anuncia tts-1", ", ".join(modelos))
    hay_vibe = "vibevoice" in modelos

    print("\n== los seis formatos ==")
    for fmt, firmas in FIRMAS.items():
        cod, cab, audio = habla(base, args.token, response_format=fmt)
        ok(cod == 200 and audio.startswith(firmas),
           f"{fmt}: firma correcta",
           f"{cod}, {len(audio)} B, {cab.get('Content-Type')}, cabeza={audio[:4]!r}")
    cod, _, crudo = habla(base, args.token, response_format="pcm")
    ok(cod == 200 and len(crudo) > 1000 and not crudo.startswith(b"RIFF"),
       "pcm: crudo y sin cabecera", f"{len(crudo)} B")

    print("\n== el wav se puede abrir de verdad ==")
    cod, _, audio = habla(base, args.token, response_format="wav")
    try:
        with wave.open(io.BytesIO(audio)) as wf:
            n, hz = wf.getnframes(), wf.getframerate()
        ok(0 < n < 10 ** 8, "el modulo `wave` lee un numero de fotogramas sensato",
           f"{n} fotogramas a {hz} Hz = {n / hz:.2f} s")
        ok(int.from_bytes(audio[4:8], "little") == len(audio) - 8,
           "el tamano del RIFF es el real, no 0xFFFFFFFF")
    except wave.Error as e:
        ok(False, "el modulo `wave` abre la respuesta", str(e))

    print("\n== speed: OpenAI es rapidez, no duracion ==")
    ds = {}
    for v in (0.5, 1.0, 2.0):
        cod, _, audio = habla(base, args.token, response_format="wav", speed=v)
        ds[v] = duracion(audio) if cod == 200 else 0.0
    ok(ds[2.0] < ds[1.0] < ds[0.5], "a mas speed, menos audio",
       " / ".join(f"speed {v} -> {d:.2f} s" for v, d in ds.items()))
    if ds[1.0]:
        ok(abs(ds[2.0] / ds[1.0] - 0.5) < 0.15, "speed 2.0 dura la mitad",
           f"razon {ds[2.0] / ds[1.0]:.2f}")

    print("\n== voces ==")
    cod, cab, _ = habla(base, args.token, voice="alloy")
    ok(cod == 200, "una voz canonica de OpenAI no rompe la peticion", f"codigo {cod}")
    ok(cab.get("X-Voz") is not None, "dice que voz uso",
       f"{cab.get('X-Voz')} {cab.get('X-Voz-Sustituida', '')}")
    cod, _, cuerpo = habla(base, args.token, voice="voz-que-no-existe-jamas")
    err = json.loads(cuerpo).get("error", {}) if cod != 200 else {}
    ok(cod == 400 and err.get("param") == "voice",
       "una voz inventada da 400 con param=voice", f"{cod} {err.get('code')}")

    print("\n== errores con forma de OpenAI ==")
    cod, _, cuerpo = habla(base, args.token, model="gpt-5-tts-inexistente")
    err = json.loads(cuerpo).get("error", {})
    ok(cod == 404 and err.get("code") == "model_not_found",
       "modelo desconocido -> 404 model_not_found", f"{cod} {err.get('code')}")
    ok(bool(err.get("message")), "el error trae `message`, que es lo que lee el SDK",
       str(err.get("message"))[:80])
    cod, _, cuerpo = habla(base, args.token, response_format="wma")
    ok(cod == 400, "formato desconocido -> 400", f"codigo {cod}")
    cod, _, _ = pedir(base, "", "/v1/audio/speech", {"model": "tts-1", "input": TEXTO})
    ok(cod in (200, 401), "sin token: 401 si hay token configurado, 200 si la API esta abierta",
       f"codigo {cod}")

    if hay_vibe:
        print("\n== VibeVoice por la misma puerta ==")
        cod, cab, audio = pedir(base, args.token, "/v1/audio/speech",
                                {"model": "vibevoice", "input": TEXTO,
                                 "response_format": "wav", "speed": 3.0})
        ok(cod == 200 and audio.startswith(b"RIFF"), "sintetiza", f"{cod}, {len(audio)} B")
        if cod == 200:
            ok(cab.get("X-Motor") == "vibevoice", "el motor es el que se pidio", cab.get("X-Motor"))
            ok("aplicada 1.20" in cab.get("X-Velocidad", ""),
               "speed 3.0 se recorta a 1,20 y se avisa", cab.get("X-Velocidad", "(sin cabecera)"))
            print(f"         RTF {cab.get('X-RTF')} sobre {cab.get('X-Duracion-S')} s de audio")
    else:
        print("\n== VibeVoice ==\n  (no anunciado en /v1/models: voz-stream no responde; se omite)")

    print("\n== el SDK oficial ==")
    try:
        from openai import NotFoundError, OpenAI
    except ImportError:
        print("  (paquete `openai` no instalado; se omite la prueba de fuego)")
    else:
        cli = OpenAI(base_url=f"{base}/v1", api_key=args.token or "sin-token")
        r = cli.audio.speech.create(model="tts-1", voice="nova", input=TEXTO)
        ok(r.content.startswith(b"ID3") or r.content[:1] == b"\xff",
           "cli.audio.speech.create() devuelve un mp3", f"{len(r.content)} B")
        with cli.audio.speech.with_streaming_response.create(
                model="tts-1", voice="echo", input=TEXTO, response_format="opus") as flujo:
            n = sum(len(t) for t in flujo.iter_bytes())
        ok(n > 0, "with_streaming_response tambien", f"{n} B")
        try:
            cli.audio.speech.create(model="gpt-5-tts-inexistente", voice="alloy", input="x")
            ok(False, "un modelo inexistente levanta NotFoundError")
        except NotFoundError as e:
            ok("no existe en este servidor" in str(e.message),
               "un modelo inexistente levanta NotFoundError con NUESTRO mensaje",
               str(e.message)[:60])

    print(f"\n{'TODO BIEN' if not fallos else str(len(fallos)) + ' FALLOS: ' + '; '.join(fallos)}\n")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
