#!/usr/bin/env python
"""Mide si la voz generada dice REALMENTE lo que se le pidio.

    python scripts/fidelidad.py                 # frases de prueba, 3 repeticiones
    python scripts/fidelidad.py --repeticiones 5
    python scripts/fidelidad.py --llm           # las frases las escribe un LLM
    python scripts/fidelidad.py --informe informe.md

COMO FUNCIONA
Cierra el circuito: texto -> VibeVoice -> whisper -> texto, y compara. Si lo
que vuelve no coincide con lo que se pidio, o el sintetizador se ha comido
algo, o lo ha pronunciado de forma que ni un reconocedor lo entiende. En
ambos casos un humano lo notaria.

Se genera CADA frase varias veces a proposito: la difusion parte de ruido
aleatorio, asi que dos generaciones del mismo texto no son identicas. Si una
sale bien y otra mal, el problema no es la frase sino la ESTABILIDAD.

QUE SE MIDE

  coincidencia exacta  tras normalizar (minusculas, sin puntuacion). Es la
                       vara mas dura y la que de verdad importa.
  WER                  proporcion de palabras mal, por distancia de edicion.
                       Es el estandar en reconocimiento de voz.
  estabilidad          si las N repeticiones de una misma frase coinciden
                       entre si. Separa "esta frase se le da mal" de "esto
                       es una loteria".

QUE NO MIDE
Si la voz suena natural o robotica. Eso no lo captura un reconocedor: whisper
entiende perfectamente una voz horrible. Esto detecta errores de CONTENIDO
-- palabras comidas, cambiadas o inventadas -- no de calidad percibida.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.request
import uuid
import wave

FRASES = [
    "El backup de anoche terminó sin errores.",
    "Los tres servicios responden con normalidad.",
    "El uso de memoria bajó un veinticuatro por ciento.",
    "Tienes tres cosas pendientes para hoy.",
    "La reunión de las once se confirmó esta mañana.",
    "No hay incidencias que reportar en las últimas horas.",
]


# whisper escribe los numeros en cifras aunque se hayan dicho con letra, y el
# simbolo % donde se dijo "por ciento". Eso NO es un fallo del sintetizador --
# la voz dijo lo correcto -- asi que contarlo como error inflaba el WER un 30 %
# en frases con cifras. Se unifican los dos lados a palabras antes de comparar.
NUMEROS = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco",
    "6": "seis", "7": "siete", "8": "ocho", "9": "nueve", "10": "diez",
    "11": "once", "12": "doce", "13": "trece", "14": "catorce", "15": "quince",
    "16": "dieciseis", "17": "diecisiete", "18": "dieciocho", "19": "diecinueve",
    "20": "veinte", "21": "veintiuno", "22": "veintidos", "23": "veintitres",
    "24": "veinticuatro", "25": "veinticinco", "30": "treinta", "40": "cuarenta",
    "50": "cincuenta", "60": "sesenta", "100": "cien",
}


def _cifras_a_palabras(t: str) -> str:
    t = re.sub(r"%", " por ciento ", t)
    return re.sub(r"\b\d+\b", lambda m: NUMEROS.get(m.group(), m.group()), t)


def normalizar(t: str) -> str:
    """Deja el texto comparable: sin puntuacion, sin mayusculas, sin dobles espacios.

    NO se quitan los acentos: confundir "termino" con "terminó" es un error
    real de pronunciacion y queremos verlo.
    """
    t = unicodedata.normalize("NFC", t.lower())
    t = _cifras_a_palabras(t)
    t = re.sub(r"[.,;:!?¿¡…\"'“”‘’()\[\]—–-]", " ", t)
    return " ".join(t.split())


def distancia(a: list, b: list) -> int:
    """Levenshtein sobre palabras."""
    if not a:
        return len(b)
    previa = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        actual = [i]
        for j, y in enumerate(b, 1):
            actual.append(min(previa[j] + 1, actual[j - 1] + 1,
                              previa[j - 1] + (x != y)))
        previa = actual
    return previa[-1]


def wer(referencia: str, hipotesis: str) -> float:
    r = normalizar(referencia).split()
    h = normalizar(hipotesis).split()
    return distancia(r, h) / len(r) if r else (0.0 if not h else 1.0)


def diferencias(referencia: str, hipotesis: str) -> str:
    """Resume que cambio, en lenguaje llano."""
    r, h = normalizar(referencia).split(), normalizar(hipotesis).split()
    if r == h:
        return ""
    faltan = [p for p in r if p not in h]
    sobran = [p for p in h if p not in r]
    partes = []
    if faltan:
        partes.append("se comió: " + ", ".join(faltan[:5]))
    if sobran:
        partes.append("añadió: " + ", ".join(sobran[:5]))
    return " · ".join(partes) or "mismo vocabulario, distinto orden"


def sintetizar_wav(texto, url, token, voz, ruta, cfg=3.0):
    pet = urllib.request.Request(
        f"{url}/tts/stream", method="POST",
        data=json.dumps({"texto": texto, "voz": voz, "cfg_scale": cfg}).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    t0 = time.time()
    r = urllib.request.urlopen(pet, timeout=600)
    datos = r.read()
    pcm = datos[44:]
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm)
    return len(pcm) / 2 / 24000, time.time() - t0


def transcribir(ruta, url, token):
    lim = "----" + uuid.uuid4().hex
    with open(ruta, "rb") as f:
        audio = f.read()
    cuerpo = (f'--{lim}\r\nContent-Disposition: form-data; name="idioma"\r\n\r\nes\r\n'
              f'--{lim}\r\nContent-Disposition: form-data; name="archivo"; '
              f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    cuerpo += audio + f"\r\n--{lim}--\r\n".encode()
    pet = urllib.request.Request(
        f"{url}/stt", method="POST", data=cuerpo,
        headers={"content-type": f"multipart/form-data; boundary={lim}",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    return json.load(urllib.request.urlopen(pet, timeout=600)).get("texto", "")


def frases_del_llm(n, modelo, ollama):
    pet = urllib.request.Request(
        f"{ollama}/api/generate", method="POST",
        data=json.dumps({"model": modelo, "stream": False,
                         "system": "Responde en español. Solo frases sueltas, una por línea, "
                                   "sin numerar, sin markdown. /no_think",
                         "prompt": f"Escribe {n} frases distintas que diría un asistente "
                                   f"de voz informando del estado de unos servidores."}).encode(),
        headers={"content-type": "application/json"})
    txt = json.load(urllib.request.urlopen(pet, timeout=300)).get("response", "")
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    lineas = [re.sub(r"^[\d.\-*)\s]+", "", l).strip() for l in txt.splitlines()]
    return [l for l in lineas if len(l) > 20][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz-url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="que las frases las escriba un LLM")
    ap.add_argument("--modelo", default="qwen3:1.7b")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--cfg", type=float, default=3.0,
                    help="guia CFG: cuanto se ciñe la difusion a la condicion")
    ap.add_argument("--informe", help="escribir un informe markdown")
    ap.add_argument("--audios", default="/tmp/fidelidad", help="donde dejar los WAV")
    a = ap.parse_args()

    frases = frases_del_llm(6, a.modelo, a.ollama) if a.llm else FRASES
    if not frases:
        print("no consegui frases del LLM; uso las de prueba", file=sys.stderr)
        frases = FRASES
    os.makedirs(a.audios, exist_ok=True)

    print(f"{len(frases)} frases x {a.repeticiones} repeticiones = "
          f"{len(frases)*a.repeticiones} generaciones\n")
    resultados = []
    for i, f in enumerate(frases):
        print(f"[{i+1}/{len(frases)}] {f}")
        pases = []
        for r in range(a.repeticiones):
            ruta = os.path.join(a.audios, f"f{i}_r{r}.wav")
            try:
                dur, gen = sintetizar_wav(f, a.voz_url, a.token, a.voz, ruta, a.cfg)
                oido = transcribir(ruta, a.api_url, a.token)
            except Exception as e:
                print(f"    r{r}: FALLO {type(e).__name__}: {e}")
                continue
            e_wer = wer(f, oido)
            exacto = normalizar(f) == normalizar(oido)
            pases.append({"oido": oido, "wer": e_wer, "exacto": exacto,
                          "dur": dur, "gen": gen})
            marca = "OK " if exacto else f"WER {e_wer:.0%}"
            print(f"    r{r}: {marca:9s} {oido!r}")
            if not exacto:
                d = diferencias(f, oido)
                if d:
                    print(f"              {d}")
        if pases:
            distintos = len({normalizar(p["oido"]) for p in pases})
            resultados.append({"texto": f, "pases": pases, "variantes": distintos})
            if distintos > 1:
                print(f"    >>> INESTABLE: {distintos} transcripciones distintas de {len(pases)}")
        print()

    # ---- resumen ----
    todos = [p for r in resultados for p in r["pases"]]
    if not todos:
        print("sin resultados"); return
    exactos = sum(1 for p in todos if p["exacto"])
    wers = [p["wer"] for p in todos]
    inestables = [r for r in resultados if r["variantes"] > 1]
    print("=" * 62)
    print(f"coincidencia exacta : {exactos}/{len(todos)}  ({exactos/len(todos):.0%})")
    print(f"WER medio           : {statistics.mean(wers):.1%}")
    print(f"WER peor            : {max(wers):.1%}")
    print(f"frases inestables   : {len(inestables)}/{len(resultados)}"
          f"  (dan distinto entre repeticiones)")

    if a.informe:
        with open(a.informe, "w", encoding="utf-8") as w:
            w.write(f"# Fidelidad del circuito voz\n\n")
            w.write(f"{len(resultados)} frases x {a.repeticiones} repeticiones. "
                    f"Texto -> VibeVoice -> whisper -> texto.\n\n")
            w.write(f"| medida | valor |\n|---|---|\n")
            w.write(f"| coincidencia exacta | {exactos}/{len(todos)} ({exactos/len(todos):.0%}) |\n")
            w.write(f"| WER medio | {statistics.mean(wers):.1%} |\n")
            w.write(f"| WER peor | {max(wers):.1%} |\n")
            w.write(f"| frases inestables | {len(inestables)}/{len(resultados)} |\n\n")
            fallos = [(r, p) for r in resultados for p in r["pases"] if not p["exacto"]]
            if fallos:
                w.write("## Dónde falla\n\n")
                for r, p in fallos:
                    w.write(f"- **pedido:** {r['texto']}\n")
                    w.write(f"  **oído:** {p['oido']}\n")
                    d = diferencias(r["texto"], p["oido"])
                    if d:
                        w.write(f"  *{d}* · WER {p['wer']:.0%}\n")
                    w.write("\n")
            else:
                w.write("## Sin fallos\n\nTodas las repeticiones coincidieron.\n")
        print(f"\ninforme en {a.informe}")


if __name__ == "__main__":
    main()
