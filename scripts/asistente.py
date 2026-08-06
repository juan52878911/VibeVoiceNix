#!/usr/bin/env python
"""Asistente de voz: pregunta -> LLM -> voz, narrando segun escribe.

    python scripts/asistente.py "¿que tal va el despliegue?"
    python scripts/asistente.py --modelo qwen3:4b --salida respuesta.wav "..."

El LLM y el sintetizador corren A LA VEZ: en cuanto hay una frase escrita se
empieza a decir, mientras el modelo sigue redactando el resto. Por eso la
espera percibida no es "lo que tarda el LLM" mas "lo que tarda la voz", sino
solo lo que tarda la PRIMERA frase.

DE DONDE SALE LA LATENCIA (medido, no estimado -- lo imprime al terminar)

    pregunta -> primer token del LLM      : depende del modelo
    -> primera frase completa             : depende de lo rapido que escriba
    -> primer sonido                       : ~355 ms del sintetizador

El tercer sumando es fijo y ya esta optimizado. Los otros dos mandan, y por eso
la palanca que de verdad se nota es CUANTO TEXTO se espera antes de hablar.
Con --arranque se ajusta: mas bajo habla antes, pero la primera frase sale peor
entonada porque el modelo tiene menos contexto para decidir la melodia.

LOS MODELOS PENSANTES ESCRIBEN COSAS QUE NO HAY QUE DECIR
qwen3 y compañia emiten un bloque <think>...</think> con su razonamiento antes
de responder. Narrarlo seria absurdo -- son parrafos enteros de deliberacion --
asi que se filtra. Tambien se quitan los asteriscos y almohadillas del markdown,
que al leerlos en voz alta suenan a ruido.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrador import RITMO, sintetizar, trocear  # noqa: E402

# Los modelos pensantes abren el bloque y lo cierran mucho despues; hay que
# poder descartar texto ENTRE medias, no solo cuando el bloque esta completo.
ABRE_PENSAMIENTO = re.compile(r"<(think|thinking|reasoning)>", re.I)
CIERRA_PENSAMIENTO = re.compile(r"</(think|thinking|reasoning)>", re.I)
# Markdown que al leerse en voz alta suena a ruido.
RUIDO_MARKDOWN = re.compile(r"[*_`#]+")


def limpiar(texto: str) -> str:
    return RUIDO_MARKDOWN.sub("", texto)


def preguntar_a_ollama(pregunta, modelo, url, sistema=None):
    """Devuelve los trozos de respuesta segun los escribe el modelo."""
    cuerpo = {"model": modelo, "prompt": pregunta, "stream": True}
    if sistema:
        cuerpo["system"] = sistema
    pet = urllib.request.Request(f"{url}/api/generate", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(pet, timeout=300)
    for linea in r:
        if not linea.strip():
            continue
        d = json.loads(linea)
        if d.get("response"):
            yield d["response"]
        if d.get("done"):
            break


# MiniMax sirve un endpoint compatible con la API de Anthropic. Se reutiliza
# la credencial que opencode ya tiene guardada: asi no hay una segunda copia
# del secreto rondando por el repo ni por el entorno.
MINIMAX_API = "https://api.minimax.io/anthropic/v1/messages"
AUTH_OPENCODE = os.path.expanduser("~/.local/share/opencode/auth.json")


def clave_minimax():
    """La clave, del entorno o de opencode. Nunca se imprime."""
    k = os.environ.get("MINIMAX_API_KEY")
    if k:
        return k
    try:
        with open(AUTH_OPENCODE) as f:
            return json.load(f)["minimax-coding-plan"]["key"]
    except (OSError, KeyError, ValueError):
        raise RuntimeError(
            "sin credencial de MiniMax: exporta MINIMAX_API_KEY o autentica "
            "el proveedor 'minimax-coding-plan' en opencode")


def preguntar_a_minimax(pregunta, modelo, sistema=None, maximo=1024):
    """Igual que la de Ollama, pero contra MiniMax. Devuelve trozos de texto.

    El razonamiento aqui NO hay que filtrarlo con expresiones regulares: la API
    lo manda en bloques `thinking_delta` aparte del texto, asi que basta con
    ignorarlos. Es mas fiable que buscar <think> dentro del flujo.
    """
    cuerpo = {"model": modelo, "max_tokens": maximo, "stream": True,
              "messages": [{"role": "user", "content": pregunta}]}
    if sistema:
        cuerpo["system"] = sistema
    pet = urllib.request.Request(MINIMAX_API, method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json",
                                          "anthropic-version": "2023-06-01",
                                          "x-api-key": clave_minimax()})
    r = urllib.request.urlopen(pet, timeout=300)
    for linea in r:
        linea = linea.strip()
        if not linea.startswith(b"data:"):
            continue
        try:
            d = json.loads(linea[5:])
        except ValueError:
            continue
        if d.get("type") == "content_block_delta":
            delta = d.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield delta["text"]
        elif d.get("type") == "message_stop":
            break


def preguntar(pregunta, modelo, url, sistema=None):
    """Elige el proveedor por el nombre del modelo. Los de MiniMax empiezan
    por 'MiniMax-'; cualquier otra cosa se le pide a Ollama."""
    if modelo.lower().startswith("minimax"):
        return preguntar_a_minimax(pregunta, modelo, sistema)
    return preguntar_a_ollama(pregunta, modelo, url, sistema)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pregunta", nargs="+")
    ap.add_argument("--modelo", default=os.environ.get("ASISTENTE_MODELO", "MiniMax-M3"),
                    help="MiniMax-M3 (por defecto) o cualquier modelo de Ollama")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--cfg", type=float, default=4.5,
                    help="guia CFG; 3.0 es el minimo fiable, 4.5 el mas marcado")
    ap.add_argument("--arranque", type=int, default=25,
                    help="caracteres minimos de la PRIMERA frase antes de hablar "
                         "(menos = responde antes, peor entonada)")
    ap.add_argument("--bufer", type=float, default=1.0,
                    help="segundos de audio acumulados antes de sonar")
    ap.add_argument("--salida", help="escribir a un WAV en vez de reproducir")
    ap.add_argument("--sistema", default="Responde en español, breve y natural, "
                                         "en frases cortas. Sin listas ni markdown.")
    a = ap.parse_args()
    pregunta = " ".join(a.pregunta)

    audio = queue.Queue(maxsize=64)
    hitos = {}
    t0 = time.time()

    def productor():
        pendiente, dentro_pensamiento, n_frases = "", False, 0
        for trozo in preguntar(pregunta, a.modelo, a.ollama, a.sistema):
            hitos.setdefault("primer_token", time.time() - t0)
            # El bloque de razonamiento puede abrirse y cerrarse a mitad de un
            # trozo, asi que se procesa caracter a caracter y no de golpe.
            texto = ""
            for parte in re.split(r"(<[^>]{0,20}>)", trozo):
                if ABRE_PENSAMIENTO.fullmatch(parte or ""):
                    dentro_pensamiento = True
                elif CIERRA_PENSAMIENTO.fullmatch(parte or ""):
                    dentro_pensamiento = False
                elif not dentro_pensamiento:
                    texto += parte or ""
            if not texto:
                continue
            pendiente += limpiar(texto)
            frases, pendiente = trocear(pendiente, primera=n_frases == 0,
                                        minimo_primera=a.arranque)
            for f in frases:
                hitos.setdefault("primera_frase", time.time() - t0)
                n_frases += 1
                for t in sintetizar(f, a.url, a.token, a.voz, a.cfg):
                    audio.put(t)
        frases, _ = trocear(pendiente, forzar_final=True, primera=n_frases == 0,
                            minimo_primera=a.arranque)
        for f in frases:
            hitos.setdefault("primera_frase", time.time() - t0)
            n_frases += 1
            for t in sintetizar(f, a.url, a.token, a.voz, a.cfg):
                audio.put(t)
        hitos["frases"] = n_frases
        audio.put(None)

    threading.Thread(target=productor, daemon=True).start()

    if a.salida:
        w = wave.open(a.salida, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        escribir, cerrar = w.writeframes, w.close
    else:
        pr = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                               "-f", "s16le", "-ar", str(RITMO), "-ac", "1", "-"],
                              stdin=subprocess.PIPE)
        escribir = pr.stdin.write
        def cerrar(): pr.stdin.close(); pr.wait()

    reloj, muestras, cortes = None, 0, 0
    while True:
        trozo = audio.get()
        if trozo is None:
            break
        llegada = time.time() - t0
        if reloj is None:
            hitos["primer_sonido"] = llegada + a.bufer
            reloj = hitos["primer_sonido"]
        elif llegada > reloj + 0.02:
            cortes += 1
            reloj = llegada
        reloj += (len(trozo) // 2) / RITMO
        muestras += len(trozo) // 2
        escribir(trozo)
    cerrar()

    seg = muestras / RITMO
    print(f"\n--- de la pregunta a la voz ---", file=sys.stderr)
    print(f"  primer token del LLM : {hitos.get('primer_token', 0):5.2f}s", file=sys.stderr)
    print(f"  primera frase lista  : {hitos.get('primera_frase', 0):5.2f}s", file=sys.stderr)
    print(f"  PRIMER SONIDO        : {hitos.get('primer_sonido', 0):5.2f}s", file=sys.stderr)
    print(f"  {hitos.get('frases', 0)} frases · {seg:.1f}s de audio · "
          f"{'sin cortes' if not cortes else f'{cortes} cortes'}", file=sys.stderr)


if __name__ == "__main__":
    main()
