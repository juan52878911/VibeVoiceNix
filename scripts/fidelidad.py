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

  coincidencia exacta  tras normalizar (minusculas, sin puntuacion, y las
                       grafias que en castellano no se pueden oir: h muda y
                       b/v -- ver comparable()). Es la vara mas dura y la que
                       de verdad importa.
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
#
# LA TABLA A MANO SE QUEDABA CORTA, Y SE VIO MIDIENDO. Llegaba a 25 y luego
# saltaba de decena en decena, asi que una narracion real del asistente -- "el
# disco va por el cuarenta y dos por ciento", "caduca en treinta y un dias" --
# daba WER 5,7 % con el contenido PERFECTO: los unicos fallos eran 42 y 31, que
# no estaban en la tabla. Ahora se generan, con las tildes que toca, porque
# normalizar() no quita acentos a proposito ("dieciseis" no casaria con
# "dieciséis").
UNIDADES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
            "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
            "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
            "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
            "veinticinco", "veintiséis", "veintisiete", "veintiocho",
            "veintinueve"]
DECENAS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta",
           7: "setenta", 8: "ochenta", 9: "noventa"}
CENTENAS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos",
            5: "quinientos", 6: "seiscientos", 7: "setecientos",
            8: "ochocientos", 9: "novecientos"}


def _numero_a_palabras(n: int) -> str:
    """0-999 en castellano. Fuera de rango se deja la cifra: mejor un fallo
    visible que una traduccion inventada."""
    if n < 30:
        return UNIDADES[n]
    if n < 100:
        d, u = divmod(n, 10)
        return DECENAS[d] + (f" y {UNIDADES[u]}" if u else "")
    if n == 100:
        return "cien"
    if n < 1000:
        c, r = divmod(n, 100)
        return CENTENAS[c] + (f" {_numero_a_palabras(r)}" if r else "")
    return str(n)


def _cifras_a_palabras(t: str) -> str:
    t = re.sub(r"%", " por ciento ", t)
    return re.sub(r"\b\d+\b",
                  lambda m: _numero_a_palabras(int(m.group()))
                  if len(m.group()) <= 3 else m.group(), t)


def normalizar(t: str) -> str:
    """Deja el texto comparable: sin puntuacion, sin mayusculas, sin dobles espacios.

    NO se quitan los acentos: confundir "termino" con "terminó" es un error
    real de pronunciacion y queremos verlo.
    """
    t = unicodedata.normalize("NFC", t.lower())
    t = _cifras_a_palabras(t)
    t = re.sub(r"[.,;:!?¿¡…\"'“”‘’()\[\]—–-]", " ", t)
    return " ".join(t.split())


def comparable(t: str) -> str:
    """normalizar() mas las dos grafias que en castellano NO se pueden oir.

    LA MISMA TRAMPA QUE LA DE LAS CIFRAS, PERO EN ORTOGRAFIA. La 'h' no suena
    y 'b' y 'v' son el MISMO fonema, asi que hay pares que el sintetizador no
    puede pronunciar distinto por mucho que se le pida: "hecho"/"echo" y
    "borrada"/"vorrada" suenan igual. Cual de las dos grafias escribe whisper
    lo decide su modelo de lenguaje, y sobre una palabra suelta de medio
    segundo no tiene contexto con el que decidirlo.

    MEDIDO en el banco de rellenos del asistente (450 clips, 1,1 s de media):
    "Hecho." sale como "¡Echo!" con 5 de 18 semillas y "Borrada." como
    "¡Vorrada!" con 6 de 18. Son 11 clips de 450 contados como error sin
    serlo: WER 28,1 % frente al 25,5 % real, y 300 exactos frente a 311.

    En frases largas no cambia NADA -- los tres bancos de pasos (36 clips cada
    uno) y el de 18 semillas (72) dan exactamente el mismo WER y los mismos
    exactos con esta funcion y sin ella --, porque ahi whisper tiene contexto
    de sobra para elegir la grafia. Solo aparece en los rellenos de una
    palabra, que es justo donde se usa para decidir.

    LA 'CH' SE PROTEGE antes de quitar las haches. Sin eso "echo" y "eco" se
    fundirian en la misma cadena, y esos dos SI suenan distinto.

    Y LA APOCOPE DEL UNO, por lo mismo que las cifras: se dice "treinta y UN
    dias" pero el numero 31 se lee "treinta y uno" en abstracto, asi que los
    dos lados se llevan a la misma forma. Es una diferencia de gramatica, no de
    pronunciacion.
    """
    t = normalizar(t).replace("ch", "\x01").replace("h", "").replace("\x01", "ch")
    t = t.replace("v", "b")
    return re.sub(r"\b(beintiun|un)\b", lambda m: m.group() + "o", t)


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
    r = comparable(referencia).split()
    h = comparable(hipotesis).split()
    return distancia(r, h) / len(r) if r else (0.0 if not h else 1.0)


def diferencias(referencia: str, hipotesis: str) -> str:
    """Resume que cambio, en lenguaje llano.

    Se decide con comparable() -- si no, se listarian como diferencia las
    grafias que suenan igual -- pero se ENSEÑA la palabra tal cual se escribio,
    que es lo que el que lee espera ver.
    """
    r_vis, h_vis = normalizar(referencia).split(), normalizar(hipotesis).split()
    r, h = comparable(referencia).split(), comparable(hipotesis).split()
    if r == h:
        return ""
    faltan = [v for p, v in zip(r, r_vis) if p not in h]
    sobran = [v for p, v in zip(h, h_vis) if p not in r]
    partes = []
    if faltan:
        partes.append("se comió: " + ", ".join(faltan[:5]))
    if sobran:
        partes.append("añadió: " + ", ".join(sobran[:5]))
    return " · ".join(partes) or "mismo vocabulario, distinto orden"


def sintetizar_wav(texto, url, token, voz, ruta, cfg=3.0, semilla=None, pasos=None):
    """Un clip por /tts/stream. semilla y pasos solo viajan si vienen: sin
    ellos el servicio hace lo de siempre (sortea; sus pasos por defecto)."""
    cuerpo = {"texto": texto, "voz": voz, "cfg_scale": cfg}
    if semilla is not None:
        cuerpo["semilla"] = semilla
    if pasos is not None:
        cuerpo["pasos"] = pasos
    pet = urllib.request.Request(
        f"{url}/tts/stream", method="POST",
        data=json.dumps(cuerpo).encode(),
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


def leer_crono(url, token):
    """GET /crono del servicio: reparto del tiempo desde la ultima lectura, y
    lo pone a cero. Se lee ANTES del banco (para descartar lo anterior) y
    DESPUES (para quedarse solo con lo del banco). None si no responde."""
    try:
        pet = urllib.request.Request(
            f"{url}/crono", headers={**({"authorization": f"Bearer {token}"}
                                        if token else {})})
        return json.load(urllib.request.urlopen(pet, timeout=30))
    except Exception:
        return None


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
    ap.add_argument("--frases", default=None,
                    help="fichero con una frase por linea (las lineas vacias y las que "
                         "empiezan por # se saltan); sustituye a las de prueba. Sirve "
                         "para medir con las frases REALES de un uso -- los rellenos "
                         "de un perfil del asistente, por ejemplo")
    ap.add_argument("--modelo", default="qwen3:1.7b")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--cfg", type=float, default=3.0,
                    help="guia CFG: cuanto se ciñe la difusion a la condicion")
    ap.add_argument("--informe", help="escribir un informe markdown")
    ap.add_argument("--audios", default="/tmp/fidelidad", help="donde dejar los WAV")
    # EL A/B EMPAREJADO. Sin semilla cada repeticion parte de otro ruido, y eso
    # es lo que mide la estabilidad; pero para comparar DOS variantes del
    # servicio (pasos, freno, cfg) hace falta que las dos partan del MISMO
    # ruido, o la diferencia que se mide es la del sorteo y no la del cambio.
    # Con --semillas, cada frase se genera una vez por semilla (y no
    # --repeticiones veces), y el fichero lleva la semilla en el nombre para
    # que naturalidad.py empareje clips entre carpetas.
    ap.add_argument("--semillas", default="",
                    help="lista separada por comas; una generacion por semilla "
                         "en vez de --repeticiones (ej. 11,7,3,23,42,101)")
    ap.add_argument("--pasos", type=int, default=None,
                    help="pasos de difusion por peticion; sin el, los del servicio")
    ap.add_argument("--csv", default=None,
                    help="una fila por clip (por defecto {audios}/clips.csv)")
    ap.add_argument("--crono", action="store_true",
                    help="leer GET /crono del servicio antes y despues, y "
                         "guardar el reparto del banco en {audios}/crono.json")
    a = ap.parse_args()

    if a.frases:
        with open(a.frases, encoding="utf-8") as fh:
            frases = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    else:
        frases = frases_del_llm(6, a.modelo, a.ollama) if a.llm else FRASES
    if not frases:
        print("no consegui frases del LLM; uso las de prueba", file=sys.stderr)
        frases = FRASES
    os.makedirs(a.audios, exist_ok=True)
    semillas = [int(s) for s in a.semillas.split(",") if s.strip()]
    # (etiqueta del fichero, semilla): con semillas una por semilla; sin ellas,
    # las repeticiones de siempre, sin semilla (sortea el servicio).
    variantes = ([(f"s{s}", s) for s in semillas] if semillas
                 else [(f"r{r}", None) for r in range(a.repeticiones)])
    ruta_csv = a.csv or os.path.join(a.audios, "clips.csv")
    if a.crono:
        leer_crono(a.voz_url, a.token)      # a cero: lo anterior no cuenta

    print(f"{len(frases)} frases x {len(variantes)} "
          f"{'semillas' if semillas else 'repeticiones'} = "
          f"{len(frases)*len(variantes)} generaciones"
          + (f" · pasos {a.pasos}" if a.pasos is not None else "") + "\n")
    resultados = []
    filas = []
    campos = ["fichero", "frase", "texto", "semilla", "pasos", "cfg", "dur_s",
              "gen_s", "rtf", "wer", "exacto", "oido"]

    def volcar_csv():
        # Se reescribe tras cada frase: si el banco muere a medias, lo hecho
        # hasta ahi queda en disco y no hay que repetirlo.
        import csv
        with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)

    for i, f in enumerate(frases):
        print(f"[{i+1}/{len(frases)}] {f}")
        pases = []
        for etiqueta, semilla in variantes:
            fichero = f"f{i}_{etiqueta}.wav"
            ruta = os.path.join(a.audios, fichero)
            try:
                dur, gen = sintetizar_wav(f, a.voz_url, a.token, a.voz, ruta,
                                          a.cfg, semilla, a.pasos)
                oido = transcribir(ruta, a.api_url, a.token)
            except Exception as e:
                print(f"    {etiqueta}: FALLO {type(e).__name__}: {e}")
                continue
            e_wer = wer(f, oido)
            exacto = comparable(f) == comparable(oido)
            pases.append({"oido": oido, "wer": e_wer, "exacto": exacto,
                          "dur": dur, "gen": gen})
            filas.append({"fichero": fichero, "frase": i, "texto": f,
                          "semilla": "" if semilla is None else semilla,
                          "pasos": "" if a.pasos is None else a.pasos,
                          "cfg": a.cfg, "dur_s": round(dur, 3),
                          "gen_s": round(gen, 3),
                          "rtf": round(gen / dur, 3) if dur else "",
                          "wer": round(e_wer, 4), "exacto": int(exacto),
                          "oido": oido})
            marca = "OK " if exacto else f"WER {e_wer:.0%}"
            print(f"    {etiqueta}: {marca:9s} {oido!r}  (rtf {gen/dur:.2f})"
                  if dur else f"    {etiqueta}: {marca:9s} {oido!r}")
            if not exacto:
                d = diferencias(f, oido)
                if d:
                    print(f"              {d}")
        if pases:
            distintos = len({comparable(p["oido"]) for p in pases})
            resultados.append({"texto": f, "pases": pases, "variantes": distintos})
            if distintos > 1:
                print(f"    >>> INESTABLE: {distintos} transcripciones distintas de {len(pases)}")
        volcar_csv()
        print()

    if a.crono:
        reparto = leer_crono(a.voz_url, a.token)
        if reparto is not None:
            with open(os.path.join(a.audios, "crono.json"), "w") as fh:
                json.dump(reparto, fh, indent=1)
            r = reparto.get("reparto_ms_por_fotograma") or {}
            if r:
                print(f"crono: {r.get('fotogramas')} fotogramas · generate "
                      f"{r.get('generate', 0):.1f} ms/fot · cabeza "
                      f"{r.get('cabeza', 0):.1f} · tts_lm {r.get('tts_lm', 0):.1f}"
                      f" · acustico {r.get('acustico', 0):.1f}\n")
    print(f"clips en {ruta_csv}")

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
            w.write(f"{len(resultados)} frases x {len(variantes)} "
                    f"{'semillas (' + a.semillas + ')' if semillas else 'repeticiones'}"
                    f"{' · ' + str(a.pasos) + ' pasos' if a.pasos is not None else ''}. "
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
