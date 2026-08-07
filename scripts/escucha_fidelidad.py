#!/usr/bin/env python
"""Mide las tres piezas de la escucha continua, sin VM y sin altavoces.

    pkgs/vibevoice/.venv/bin/python scripts/escucha_fidelidad.py

Todo corre contra lo LOCAL: la compuerta contra Ollama, whisper y Piper
contra el docker del repo (cd docker && docker compose up -d whisper voz-api,
token en docker/.env), y las huellas contra los WAV de ejemplos-voces/. No
reproduce ni un segundo de audio: el "hablar" de las pruebas son WAV que
genera Piper o que ya estaban en el repo.

QUE MIDE
  1. Compuerta de destinatario: 24 frases con verdad de terreno (12 dirigidas
     al asistente, con y sin conversacion previa; 12 que claramente no) por
     cada modelo de --modelos. Acierto y latencia.
  2. Whisper local: latencia de /stt con locuciones de Piper de varios largos.
  3. Huellas de voz (scripts/oido.py): separacion entre locutores con las 6
     voces españolas de VibeVoice y las 3 de Piper, barrido del umbral, coste
     por huella, y el descarte de la voz del propio asistente aprendida de
     audio GENERADO (los sesion-*.wav del repo).

RESULTADOS de la pasada de referencia (2026-08-06, Mac M-series, todo local):
  compuerta qwen3:4b    24/24 aciertos · 0,70 s de media, 0,83 el peor
  compuerta qwen3:1.7b  17/24 (7 falsos positivos) · 0,31 s. NO lo uses.
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
"""
import argparse
import io
import json
import os
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


def probar_stt(api, token):
    print("== whisper local (/stt) ==")
    frases = [
        "Oye, ¿cuánto consume mi servidor?",
        "¿Me puedes decir qué tal ha ido el despliegue de esta noche y si "
        "queda algo pendiente?",
        "Necesito que me hagas un resumen de los contenedores que están "
        "corriendo ahora mismo, cuánta memoria usa cada uno, y que me avises "
        "si alguno se ha reiniciado más de dos veces esta semana.",
    ]
    for f in frases:
        wav = _piper_wav(api, token, f)
        with wave.open(io.BytesIO(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        s, texto = _stt(api, token, wav)
        print(f"  locucion de {dur:4.1f}s -> {s:.2f}s de whisper · «{texto[:60]}»")


# ---- huellas ------------------------------------------------------------

def _wav_bytes(ruta, desde=0.0, hasta=None):
    """Un trozo de un WAV del repo, como bytes WAV (lo que ve /escuchar)."""
    with wave.open(ruta) as w:
        ritmo = w.getframerate()
        w.setpos(int(desde * ritmo))
        fin = int(hasta * ritmo) if hasta else w.getnframes()
        datos = w.readframes(fin - int(desde * ritmo))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL",
                                                       "http://localhost:11434"))
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL",
                                                        "http://127.0.0.1:8080"))
    ap.add_argument("--token-api", default=os.environ.get("VOZ_API_TOKEN", ""))
    ap.add_argument("--modelos", default="qwen3:4b,qwen3:1.7b",
                    help="modelos de compuerta a comparar, separados por comas")
    ap.add_argument("--solo", choices=["compuerta", "stt", "huellas"],
                    help="una sola seccion en vez de las tres")
    ap.add_argument("--perfiles-tmp", default="/tmp/perfiles_prueba.json",
                    help="fichero de perfiles DE PRUEBA (se pisa)")
    a = ap.parse_args()
    if os.path.exists(a.perfiles_tmp):
        os.unlink(a.perfiles_tmp)
    if a.solo in (None, "compuerta"):
        probar_compuerta([m.strip() for m in a.modelos.split(",") if m.strip()],
                         a.ollama)
    if a.solo in (None, "stt"):
        probar_stt(a.api_url, a.token_api)
    if a.solo in (None, "huellas"):
        probar_huellas(a.api_url, a.token_api, a.perfiles_tmp)


if __name__ == "__main__":
    main()
