#!/usr/bin/env python
"""Mide las piezas de la escucha continua, sin VM y sin altavoces.

    pkgs/vibevoice/.venv/bin/python scripts/escucha_fidelidad.py

Todo corre contra lo LOCAL: la compuerta contra Ollama, whisper y Piper
contra el docker del repo (cd docker && docker compose up -d whisper voz-api,
token en docker/.env), y las huellas contra los WAV de ejemplos-voces/. No
reproduce ni un segundo de audio: el "hablar" de las pruebas son WAV que
genera Piper o que ya estaban en el repo.

LO QUE ESTO NO MIDE, Y HAY QUE MEDIR EN EL NAVEGADOR: el tiempo desde que
empiezas a hablar hasta que el asistente se calla. Ahi entran el VAD y el
reloj de la pagina, y no valen las piezas por separado. Se mide con la pagina
real, muda, inyectando audio por los ganchos de prueba:

    window.__escucha.mudo = true;          // sale por el nodo de ganancia a 0
    window.__escucha.armar();
    lanzarPregunta("cuéntame algo largo");
    // cuando este hablando:
    await window.__escucha.inyectarEnVivo(<pcm s16 en base64>, 16000);
    window.__escucha.estado().callarMs

QUE MIDE
  1. Compuerta de destinatario: 24 frases con verdad de terreno (12 dirigidas
     al asistente, con y sin conversacion previa; 12 que claramente no) por
     cada modelo de --modelos. Acierto y latencia.
  2. Whisper: latencia Y tasa de error de palabra, comparando el de Docker
     con uno nativo si se le pasa --whisper-url (scripts/whisper-mac.sh).
     Es el sumando gordo de la latencia y donde mas se gana.
  3. Huellas de voz (scripts/oido.py): separacion entre locutores con las 6
     voces españolas de VibeVoice y las 3 de Piper, barrido del umbral, coste
     por huella, y el descarte de la voz del propio asistente aprendida de
     audio GENERADO (los sesion-*.wav del repo).
  4. La barrera de interrupcion: cuanto audio hace falta para saber que quien
     habla NO es el asistente, y si su propia voz da falsos positivos.
  5. Las ordenes de interrupcion (scripts/interrupcion.py): 46 frases con
     verdad de terreno, incluidas las que DEBEN caer al LLM.

RESULTADOS de la pasada de referencia (2026-08-06, Mac M-series, todo local):
  compuerta qwen3:4b    24/24 aciertos · 0,70 s de media, 0,83 el peor
  compuerta qwen3:1.7b  17/24 (7 falsos positivos) · 0,31 s. NO lo uses.
  compuerta SIN Ollama  Qwen3-4B-Instruct-2507 GGUF Q4_K_M en llama-server
                        (--ollama http://127.0.0.1:PUERTO, decidir() detecta
                        el servidor sola): 24/24 · 0,47 s de media, 0,64 el
                        peor. Menos modelo NO llega, medido con esta bateria:
                        Qwen2.5-3B 21/24 · 1.5B 19/24 · 0.5B 15/24 ·
                        Llama-3.2-1B 14/24.
  whisper /stt          2,1-2,9 s para locuciones de 1,9-10,3 s. ES EL
                        SUMANDO GORDO de la latencia: whisper.cpp (modelo
                        small) corre en Docker, donde no hay Metal, y el
                        coste apenas crece con el largo del audio -- es
                        arranque, no proceso.
  huellas               mismo locutor >= 0,623 · distinto <= 0,449; todo
                        umbral entre 0,45 y 0,60 acierta 153/153 pares;
                        ~26 ms por huella tras ~1-6 s de carga unica
  asistente             6/6 locuciones generadas descartadas (coseno
                        0,60-1,00 contra el perfil aprendido de OTRAS
                        generadas); una voz humana matriculada pasa aunque
                        el asistente este hablando (margen 0,10) y una
                        desconocida en esa situacion se descarta

Y DE LA PASADA DEL 2026-08-07, con la interrupcion ya montada (mismo Mac, M4):
  whisper small Docker  2177 ms de media · WER 15,0 % (9 errores de 60)
  whisper small nativo   277 ms de media · WER 15,0 % (LOS MISMOS 9 errores)
  whisper base  nativo   101 ms de media · WER 28,3 %. No compensa: los
                        errores caen justo en las frases de una palabra que
                        son las de interrumpir («Espera» -> «Espira»).
  barrera               0 falsos positivos en 41 trozos de la propia voz del
                        asistente, de 0,35 a 1,0 s. Con 0,5-0,7 s de voz
                        humana acierta; con 0,35 s acierta a veces. ~13 ms
                        por consulta, ida y vuelta incluidas.
  intenciones           46/46 de la bateria de scripts/interrupcion.py
  cierre del VAD        450 ms es el minimo que no parte ninguna de 12
                        locuciones; de 450 a 600 no cambia nada. Se deja 500.
  umbral del asistente  SEPARADO del de las personas (0,45 frente a 0,25) por
                        un fallo que salio aqui: con uno solo, una voz que el
                        asistente no habia oido nunca daba 0,266 contra su
                        perfil y se archivaba como SUYA -- un invitado no
                        podia ni hablarle ni cortarle. Su propia voz da
                        0,602-1,00, asi que el hueco da de sobra.
"""
import argparse
import io
import json
import os
import re
import struct
import sys
import time
import urllib.request
import uuid
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conversacion import decidir  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCES = os.path.join(RAIZ, "ejemplos-voces")

# ---- la bateria de la compuerta ----------------------------------------
# Cada caso: (texto, dirigida_de_verdad, historial, hablante). El historial
# es el formato de la pagina: {rol, texto, quien}.
H_CONSUMO = [
    {"rol": "usuario", "texto": "¿cuánto consume el servidor?", "quien": "Juan"},
    {"rol": "asistente", "texto": "Unos cuarenta vatios de media."},
]
H_REINICIO = [
    {"rol": "usuario", "texto": "¿puedes reiniciar el contenedor de whisper?",
     "quien": "Juan"},
    {"rol": "asistente", "texto": "Puedo reiniciarlo ahora mismo si quieres."},
]
H_RESUMEN = [
    {"rol": "asistente",
     "texto": "¿Quieres que te lea el resumen entero o solo los titulares?"},
]
H_VISITAS = [
    {"rol": "usuario", "texto": "¿cuánta gente vino ayer?", "quien": "Juan"},
    {"rol": "asistente", "texto": "Ayer hubo doce visitas."},
]
BATERIA = [
    # dirigidas al asistente, sin contexto
    ("Oye, ¿cuánto consume mi servidor?", True, None),
    ("¿Qué hora es?", True, None),
    ("Apaga las luces del salón.", True, None),
    ("¿Me repites lo último que has dicho?", True, None),
    ("Pon un temporizador de diez minutos.", True, None),
    ("¿Qué tal va el despliegue de anoche?", True, None),
    ("Oye, ¿me miras cuánta memoria le queda a la máquina virtual?", True, None),
    # dirigidas: continuaciones que solo se entienden con el historial
    ("¿Y al mes cuánto me cuesta?", True, H_CONSUMO),
    ("Vale, hazlo.", True, H_REINICIO),
    ("Solo los titulares.", True, H_RESUMEN),
    ("No, me refería a esta semana.", True, H_VISITAS),
    ("Gracias, muy útil.", True, H_CONSUMO),
    # NO dirigidas: gente hablando entre si, telefono, hablar solo
    ("Ana, pásame el pan.", False, None),
    ("Que sí, que ya voy.", False, None),
    ("¿Quieres que pidamos pizza esta noche?", False, None),
    ("Juan, ¿has visto mis llaves?", False, None),
    ("Y entonces le dije que ni hablar, que eso no era lo acordado.", False, None),
    ("Mañana tengo que llamar al dentista sin falta.", False, None),
    ("Vale cariño, hasta luego, un beso.", False, None),
    ("El niño no se ha comido la merienda otra vez.", False, None),
    # la trampa buena: hablar DEL asistente con otra persona, con contexto
    ("Ana, este trasto dice que el servidor gasta cuarenta vatios.", False, H_CONSUMO),
    ("A ver si esta semana llueve de una vez.", False, None),
    ("Uy, se me ha caído el café encima del teclado.", False, None),
    ("Sí, sí, claro, claro.", False, None),
]


def probar_compuerta(modelos, ollama):
    print("== compuerta de destinatario ==")
    for modelo in modelos:
        aciertos, tiempos, fallos = 0, [], []
        # primera llamada fuera de la cuenta: carga el modelo en Ollama
        decidir("hola", None, None, modelo, ollama)
        for texto, verdad, hist in BATERIA:
            dirigida, s, crudo = decidir(texto, hist, "Juan", modelo, ollama,
                                         segundos_desde_respuesta=8 if hist else None)
            tiempos.append(s)
            if dirigida == verdad:
                aciertos += 1
            else:
                fallos.append((texto, verdad, crudo))
        print(f"  {modelo}: {aciertos}/{len(BATERIA)} aciertos · "
              f"media {sum(tiempos)/len(tiempos):.2f}s · "
              f"peor {max(tiempos):.2f}s")
        for texto, verdad, crudo in fallos:
            print(f"    FALLO {'(era para el)' if verdad else '(no era)'}: "
                  f"«{texto}» -> {crudo!r}")


# ---- whisper local ------------------------------------------------------

def _piper_wav(api, token, texto, voz="es_ES-davefx-medium"):
    pet = urllib.request.Request(
        f"{api}/tts", method="POST",
        data=json.dumps({"texto": texto, "voz": voz, "formato": "wav"}).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    return urllib.request.urlopen(pet, timeout=60).read()


def _stt(api, token, wav):
    lim = "----" + uuid.uuid4().hex
    cuerpo = (f'--{lim}\r\nContent-Disposition: form-data; name="idioma"'
              f'\r\n\r\nes\r\n'
              f'--{lim}\r\nContent-Disposition: form-data; name="archivo"; '
              f'filename="voz.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    cuerpo += wav + f"\r\n--{lim}--\r\n".encode()
    pet = urllib.request.Request(
        f"{api}/stt", method="POST", data=cuerpo,
        headers={"content-type": f"multipart/form-data; boundary={lim}",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(pet, timeout=120))
    return time.perf_counter() - t0, d.get("texto", "")


# Las frases con las que se compara whisper. Cortas A PROPOSITO: las de una
# palabra son las de interrumpir, y son las que peor lleva un modelo pequeño.
FRASES_STT = [
    "Espera.",
    "¿Cómo?",
    "Para, para un momento.",
    "No, espera, ¿qué has dicho?",
    "Vale, sigue.",
    "Cállate un segundo.",
    "Oye, ¿cuánto consume mi servidor?",
    "¿Qué tal ha ido el despliegue de anoche?",
    "Detalla eso último que has dicho.",
    "Vuelve atrás, a lo del contenedor de whisper.",
    "Ponme un temporizador de diez minutos.",
    "¿Me puedes decir cuánta memoria le queda a la máquina virtual?",
]


def _normalizar(t):
    import unicodedata
    t = unicodedata.normalize("NFD", (t or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]", " ", t).split()


def _wer(ref, hip):
    """(errores, palabras). Levenshtein sobre palabras, sin numpy."""
    r, h = _normalizar(ref), _normalizar(hip)
    fila = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        nueva = [i]
        for j in range(1, len(h) + 1):
            nueva.append(min(fila[j] + 1, nueva[j - 1] + 1,
                             fila[j - 1] + (r[i - 1] != h[j - 1])))
        fila = nueva
    return fila[-1], len(r)


def _stt_nativo(url, wav, prompt):
    lim = "----" + uuid.uuid4().hex
    campos = {"language": "es", "temperature": "0.0",
              "response_format": "json", "prompt": prompt}
    cuerpo = b""
    for k, v in campos.items():
        cuerpo += (f'--{lim}\r\nContent-Disposition: form-data; name="{k}"'
                   f'\r\n\r\n{v}\r\n').encode()
    cuerpo += (f'--{lim}\r\nContent-Disposition: form-data; name="file"; '
               f'filename="v.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    cuerpo += wav + f"\r\n--{lim}--\r\n".encode()
    pet = urllib.request.Request(
        f"{url}/inference", method="POST", data=cuerpo,
        headers={"content-type": f"multipart/form-data; boundary={lim}"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(pet, timeout=120))
    return time.perf_counter() - t0, (d.get("text") or "").strip()


PROMPT_STT = ("Vocabulario tecnico: homelab, Proxmox, WireGuard, Docker, "
              "contenedor, Caddy, systemd, OpenClaw, Piper, whisper, backup, "
              "deploy, NixOS.")


def probar_stt(api, token, nativo=""):
    """Latencia Y precision. Sin la segunda columna, la primera engaña: un
    modelo mas pequeño siempre es mas rapido, la gracia es saber que cuesta."""
    print("== whisper: latencia y tasa de error ==")
    audios = []
    for f in FRASES_STT:
        wav = _piper_wav(api, token, f)
        with wave.open(io.BytesIO(wav)) as w:
            audios.append((f, wav, w.getnframes() / w.getframerate()))
    largo = sum(d for _, _, d in audios)
    print(f"  {len(audios)} locuciones, {largo:.1f}s de audio en total")
    motores = [("voz-api (Docker)", lambda w: _stt(api, token, w))]
    if nativo:
        motores.append((f"nativo {nativo}",
                        lambda w: _stt_nativo(nativo, w, PROMPT_STT)))
    for nombre, fn in motores:
        try:
            fn(audios[0][1])                       # calentar
        except Exception as e:
            print(f"  {nombre}: no responde ({type(e).__name__}: {e})")
            continue
        ts, errs, tot, malas = [], 0, 0, []
        for ref, wav, _ in audios:
            s, texto = fn(wav)
            ts.append(s)
            e, n = _wer(ref, texto)
            errs += e
            tot += n
            if e:
                malas.append(f"«{ref}» -> «{texto}»")
        print(f"  {nombre:20s} {sum(ts)/len(ts)*1000:5.0f} ms de media · "
              f"{max(ts)*1000:5.0f} el peor · WER {errs/tot*100:4.1f}% "
              f"({errs}/{tot} palabras)")
        for m in malas:
            print(f"      {m}")


# ---- huellas ------------------------------------------------------------

def _wav_bytes(ruta, desde=0.0, hasta=None):
    """Un trozo de un WAV del repo, como bytes WAV (lo que ve /escuchar).

    Un `desde` pasado del final devuelve un WAV vacio en vez de reventar: los
    barridos piden trozos a intervalos fijos sobre locuciones de largos
    distintos y el que se pase se descarta solo por corto."""
    with wave.open(ruta) as w:
        ritmo, total = w.getframerate(), w.getnframes()
        ini = min(int(desde * ritmo), total)
        w.setpos(ini)
        fin = min(int(hasta * ritmo), total) if hasta else total
        datos = w.readframes(max(0, fin - ini))
    salida = io.BytesIO()
    with wave.open(salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ritmo)
        w.writeframes(datos)
    return salida.getvalue()


def _pcm(ruta):
    with wave.open(ruta) as w:
        return w.readframes(w.getnframes()), w.getframerate()


def probar_huellas(api, token, ruta_tmp):
    print("== huellas de voz (oido.py) ==")
    from oido import Oido, _decodificar_wav
    oido = Oido(ruta_tmp)
    t0 = time.perf_counter()
    if oido._cargar_modelo() is None:
        print(f"  SIN MODELO: {oido.estado()['error']}")
        return
    print(f"  carga del modelo: {time.perf_counter() - t0:.1f}s (una vez)")

    # separacion: 6 voces de VibeVoice partidas en dos + 3 de Piper x2 frases
    import torch
    muestras = {}      # locutor -> [huellas]
    tiempos = []
    for v in ["sp-Spk0_woman", "sp-Spk1_man", "sp-Spk2_woman",
              "sp-Spk3_man", "sp-Spk4_woman", "sp-Spk5_man"]:
        ruta = os.path.join(VOCES, f"{v}.wav")
        with wave.open(ruta) as w:
            mitad = w.getnframes() / w.getframerate() / 2
        for a, b in ((0, mitad), (mitad, None)):
            t = _decodificar_wav(_wav_bytes(ruta, a, b))
            t1 = time.perf_counter()
            h = oido._huella(t)
            tiempos.append(time.perf_counter() - t1)
            muestras.setdefault(v, []).append(h)
    for voz in ["es_ES-davefx-medium", "es_MX-ald-medium", "es_MX-claude-high"]:
        for frase in ["Hola, soy una voz de prueba y digo una frase cualquiera.",
                      "El despliegue de anoche terminó sin errores conocidos."]:
            t = _decodificar_wav(_piper_wav(api, token, frase, voz))
            t1 = time.perf_counter()
            h = oido._huella(t)
            tiempos.append(time.perf_counter() - t1)
            muestras.setdefault(voz, []).append(h)
    intra, inter = [], []
    claves = [(l, i) for l, hs in muestras.items() for i in range(len(hs))]
    for i, (la, ia) in enumerate(claves):
        for lb, ib in claves[i + 1:]:
            c = float(muestras[la][ia] @ muestras[lb][ib])
            (intra if la == lb else inter).append(c)
    print(f"  huella: media {sum(tiempos)/len(tiempos)*1000:.0f} ms · "
          f"peor {max(tiempos)*1000:.0f} ms")
    print(f"  mismo locutor: min {min(intra):.3f} media "
          f"{sum(intra)/len(intra):.3f} · distinto: max {max(inter):.3f} "
          f"media {sum(inter)/len(inter):.3f}")
    total = len(intra) + len(inter)
    for umbral in [0.40, 0.45, 0.50, 0.55, 0.60]:
        bien = sum(1 for c in intra if c >= umbral) + \
            sum(1 for c in inter if c < umbral)
        print(f"  umbral {umbral:.2f}: {bien}/{total} pares bien "
              f"({sum(1 for c in intra if c < umbral)} mismos separados, "
              f"{sum(1 for c in inter if c >= umbral)} distintos confundidos)")

    # el asistente: perfil aprendido de audio GENERADO, descarte del resto
    print("  -- descarte de la propia voz --")
    for f in ["sesion-encadenada-0", "sesion-suelta-1"]:
        pcm, ritmo = _pcm(os.path.join(VOCES, f"{f}.wav"))
        oido.aprender_asistente(pcm, ritmo)
    generadas = ["sesion-encadenada-1", "sesion-suelta-0", "sesion-suelta-2",
                 "velocidad-1.00", "expresividad-1.5", "expresividad-4.5"]
    descartadas = 0
    for f in generadas:
        r = oido.identificar(_wav_bytes(os.path.join(VOCES, f"{f}.wav")))
        descartadas += bool(r["descartada"])
        print(f"    {f:22s}: {'DESCARTADA' if r['descartada'] else 'PASA'} "
              f"(cos {r.get('cos')}, {r['s']*1000:.0f} ms)")
    print(f"  {descartadas}/{len(generadas)} locuciones del asistente descartadas")
    # y una persona no se descarta ni hablando el asistente, si esta dada de alta
    persona = _wav_bytes(os.path.join(VOCES, "sp-Spk0_woman.wav"), 0, 3.5)
    r = oido.identificar(persona)          # la conoce (perfil nuevo)
    persona2 = _wav_bytes(os.path.join(VOCES, "sp-Spk0_woman.wav"), 3.5, 6.9)
    r2 = oido.identificar(persona2, hablando=True)
    print(f"  interrupcion: voz humana conocida con el asistente hablando -> "
          f"{'DESCARTADA (mal)' if r2['descartada'] else 'pasa (bien)'}"
          f" (cos {r2.get('cos')})")
    desconocida = _piper_wav(api, token, "Hola, ¿me oyes bien desde ahí?",
                             "es_MX-ald-medium")
    r3 = oido.identificar(desconocida, hablando=True)
    print(f"  interrupcion: voz DESCONOCIDA con el asistente hablando -> "
          f"{'descartada (bien: podria ser una mezcla)' if r3['descartada'] else 'PASA (mal)'}")
    # El otro lado del mismo umbral: una voz que el asistente NO ha oido nunca
    # no puede archivarse como suya. Con un solo umbral para las dos cosas
    # (0,25) pasaba: sp-Spk0_woman da 0,266 contra el perfil del asistente y
    # se ignoraba entera, asi que un invitado no podia ni hablarle.
    r4 = oido.identificar(_piper_wav(api, token,
                                     "Buenas, ¿me puedes decir la hora?",
                                     "es_MX-claude-high"))
    print(f"  voz nueva con el asistente CALLADO -> "
          f"{'ARCHIVADA COMO SUYA (mal)' if r4.get('perfil') == 'asistente' else 'perfil propio (bien)'}"
          f" (cos {r4.get('cos')})")


# ---- la barrera de interrupcion ----------------------------------------

def probar_barrera(api, token, ruta_tmp):
    """Cuanto audio hace falta para callar, y si se calla cuando no debe.

    Las DOS preguntas son igual de importantes y tiran en sentidos opuestos:
    con menos audio se calla antes, pero con menos audio tambien es mas facil
    confundir su propia voz con la de otro. La segunda tabla -- falsos
    positivos sobre su propia voz -- es la que pone el suelo.
    """
    print("== la barrera de interrupcion (oido.barrera) ==")
    from oido import Oido
    oido = Oido(ruta_tmp)
    if oido._cargar_modelo() is None:
        print(f"  SIN MODELO: {oido.estado()['error']}")
        return
    # El asistente, aprendido de audio GENERADO, que es como funciona en vivo.
    for f in ["sesion-encadenada-0", "sesion-suelta-1", "velocidad-1.00"]:
        pcm, ritmo = _pcm(os.path.join(VOCES, f"{f}.wav"))
        oido.aprender_asistente(pcm, ritmo)
    # Una persona matriculada con la PRIMERA mitad de su grabacion; se prueba
    # con la segunda, que el perfil no ha visto.
    ruta = os.path.join(VOCES, "sp-Spk0_woman.wav")
    with wave.open(ruta) as w:
        mitad = w.getnframes() / w.getframerate() / 2
    oido.matricular("Persona", _wav_bytes(ruta, 0, mitad))

    largos = [0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"  {'voz':>5}  persona (debe interrumpir)   asistente (no debe)")
    for L in largos:
        ok = tot = 0
        for off in [mitad + x for x in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)]:
            w = _wav_bytes(ruta, off, off + L)
            if len(w) < 44 + int(L * 24000 * 2) * 0.9:
                continue
            tot += 1
            ok += bool(oido.barrera(w)["humano"])
        malos = suyos = 0
        for f in ["sesion-encadenada-1", "sesion-suelta-0", "velocidad-1.00"]:
            for off in (0.3, 1.0, 1.8, 2.6, 3.4):
                w = _wav_bytes(os.path.join(VOCES, f"{f}.wav"), off, off + L)
                if len(w) < 44 + int(L * 24000 * 2) * 0.9:
                    continue
                suyos += 1
                malos += bool(oido.barrera(w)["humano"])
        print(f"  {L:5.2f}s      {ok:2d}/{tot:2d}                     "
              f"{malos:2d} falsos de {suyos:2d}")
    t0 = time.perf_counter()
    n = 20
    w = _wav_bytes(ruta, mitad, mitad + 0.5)
    for _ in range(n):
        oido.barrera(w)
    print(f"  coste: {(time.perf_counter()-t0)/n*1000:.0f} ms por consulta")


# ---- las ordenes de interrupcion ----------------------------------------

def probar_intenciones():
    """La tabla de frases hechas contra su verdad de terreno.

    Los fallos importan en dos direcciones distintas: dar por orden algo que
    era una pregunta PIERDE la pregunta; no reconocer una orden solo la manda
    al LLM y cuesta un par de segundos. La segunda es la barata."""
    from interrupcion import BATERIA, clasificar
    print("== ordenes de interrupcion (interrupcion.py) ==")
    fallos = [(t, clasificar(t), v) for t, v in BATERIA if clasificar(t) != v]
    print(f"  {len(BATERIA)-len(fallos)}/{len(BATERIA)} bien")
    for t, r, v in fallos:
        grave = " (GRAVE: se come una pregunta)" if v is None else ""
        print(f"    FALLO «{t}» -> {r}, esperado {v}{grave}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL",
                                                       "http://localhost:11434"))
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL",
                                                        "http://127.0.0.1:8080"))
    ap.add_argument("--token-api", default=os.environ.get("VOZ_API_TOKEN", ""))
    ap.add_argument("--modelos", default="qwen3:4b,qwen3:1.7b",
                    help="modelos de compuerta a comparar, separados por comas")
    ap.add_argument("--whisper-url", default=os.environ.get(
        "WHISPER_NATIVO_URL", ""),
        help="un whisper.cpp nativo con el que comparar el de Docker "
             "(scripts/whisper-mac.sh levanta uno en el 8083)")
    ap.add_argument("--solo", choices=["compuerta", "stt", "huellas",
                                       "barrera", "intenciones"],
                    help="una sola seccion en vez de todas")
    ap.add_argument("--perfiles-tmp", default="/tmp/perfiles_prueba.json",
                    help="fichero de perfiles DE PRUEBA (se pisa)")
    a = ap.parse_args()
    if os.path.exists(a.perfiles_tmp):
        os.unlink(a.perfiles_tmp)
    if a.solo in (None, "intenciones"):
        probar_intenciones()        # es instantaneo: va primero
    if a.solo in (None, "compuerta"):
        probar_compuerta([m.strip() for m in a.modelos.split(",") if m.strip()],
                         a.ollama)
    if a.solo in (None, "stt"):
        probar_stt(a.api_url, a.token_api, a.whisper_url.rstrip("/"))
    if a.solo in (None, "huellas"):
        probar_huellas(a.api_url, a.token_api, a.perfiles_tmp)
    if a.solo in (None, "barrera"):
        tmp = a.perfiles_tmp + ".barrera"
        if os.path.exists(tmp):
            os.unlink(tmp)
        probar_barrera(a.api_url, a.token_api, tmp)


if __name__ == "__main__":
    main()
