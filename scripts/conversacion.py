#!/usr/bin/env python
"""Conversación continua: la compuerta de destinatario y el historial.

Dos piezas que la escucha continua necesita y el asistente de una pregunta
no tenía:

  - decidir(): ¿esta frase va dirigida al asistente, o es conversación de
    fondo? Una llamada barata por intervención oída.
  - preguntar_con_historial(): la respuesta de verdad, con la conversación
    entera detrás, no preguntas sueltas.

POR QUE LA COMPUERTA VA A OLLAMA Y NO A MINIMAX, MEDIDO
El plan era MiniMax-M2.5-highspeed, pero por el endpoint compatible con
Anthropic TODOS los MiniMax razonan antes de contestar y eso manda:

    M2.5-highspeed, razonamiento libre        5,8 s   (934 letras de pensar)
    M2.7-highspeed, razonamiento libre        5,9 s
    M2.5-highspeed, thinking:{type:disabled}  3,3 s   (piensa igual, menos)
    M2.5-highspeed, prefill del asistente     sigue pensando; no lo esquiva

    qwen3:4b  en Ollama local, think:false    ~0,7 s
    qwen3:1.7b en Ollama local, think:false   ~0,3 s

Seis segundos de compuerta por intervención mata la fluidez que se pide;
0,3-0,7 s no se notan. Ollama sí tiene un interruptor DE VERDAD para el
razonamiento ("think": false, distinto del sufijo /no_think, que qwen3 ya no
respeta). Y la respuesta se fuerza con SALIDA ESTRUCTURADA (format con un
esquema JSON {dirigida: bool}): sin ella, qwen3:4b con think:false se pone a
razonar EN PROSA ("Okay, let's see...") y el veredicto no llega nunca en
pocos tokens; la decodificación restringida le quita la opción. Medido con
la batería de 24 frases de scripts/escucha_fidelidad.py: qwen3:4b acierta
más que qwen3:1.7b y es el defecto; el pequeño queda para máquinas justas.
Si la compuerta se configura con un modelo MiniMax-*, funciona igualmente
(thinking desactivado, SI/NO en texto), asumiendo la espera.

SIN OLLAMA: LO MISMO CONTRA llama-server (llama.cpp), MEDIDO 2026-08-06
Para el empaquetado final (Nix, sin servicios externos) la compuerta corre
igual contra un llama-server con Qwen3-4B-Instruct-2507 en GGUF Q4_K_M
fijado por hash. Con la bateria de escucha_fidelidad.py, en el mismo Mac:

    qwen3:4b via Ollama (referencia del dia)      24/24 · 0,65 s de media
    2507 via llama-server, Metal                  24/24 · 0,47 s · RSS 3,1 GB
    2507 via llama-server, solo CPU (8 hilos M4)  24/24 · 1,31 s

Menos modelo NO llega, medido con la misma bateria y el mismo esquema:
Qwen2.5-3B 21/24 · qwen3:1.7b 17/24 · Qwen2.5-1.5B 19/24 · Qwen2.5-0.5B
15/24 · Llama-3.2-1B 14/24. La compuerta necesita un ~4B y por tanto GPU
(en CPU rompe el segundo) y ~3 GB: vive donde el puente, no en la VM.
decidir() distingue el servidor sola (GET /props, que Ollama no tiene), asi
que la misma opcion --ollama vale para los dos y quitarlo no toca nada.

CUANTO HISTORIAL Y POR QUE
  - La compuerta ve los últimos 6 apuntes (3 turnos). Le bastan para pillar
    continuaciones («¿y en euros?», «vale, hazlo») y más solo la encarece:
    se paga en CADA intervención oída, dirigida o no.
  - La respuesta ve los últimos 16 apuntes (8 turnos). Es el cerebro: debe
    recordar de qué se habla. Más allá de 8 turnos el coste en tokens crece
    y lo que suele importar ya no está tan atrás; si un día hace falta más,
    es una constante.
El historial vive en el NAVEGADOR (viaja con cada petición), igual que la
instrucción de sistema: el puente no guarda estado y recargar la página
empieza de cero, que es lo esperable.

LAS RESPUESTAS CORTADAS SE GUARDAN PARTIDAS EN DOS
Cuando alguien interrumpe, el apunte del asistente no es «lo que dijo»: es lo
que la otra persona LLEGÓ A OÍR más lo que se quedó dentro. Las dos cosas
significan cosas distintas -- «detalla eso último» apunta a lo oído y «sigue»
a lo que faltaba -- y las dos se le enseñan al modelo. Sin ellas cree que
terminó la frase y repite el final, o peor, arranca de cero.

PERO NO DENTRO DE SU PROPIO TURNO. El apunte del asistente lleva SOLO lo que
se oyó; el aviso del corte va en la INSTRUCCIÓN DE SISTEMA
(interrupcion.nota_de_corte). Metido dentro del mensaje del asistente, el
modelo lo lee como algo que escribió él y lo continúa: medido con qwen3:4b,
su respuesta empezó literalmente por «(te quedaba por decir: «…asignada será
la del…»» y eso salió por el altavoz. En el sistema es una regla sobre la
conversación, no un trozo de ella, y no se repite.
"""
import json
import re
import time
import urllib.error
import urllib.request

from asistente import MINIMAX_API, clave_minimax
from interrupcion import nota_de_corte, texto_para_la_compuerta

RECUERDO_COMPUERTA = 6
RECUERDO_RESPUESTA = 16

# El prompt esta AFINADO contra la bateria de escucha_fidelidad.py, y dos
# reglas salieron de fallos reales, no de la imaginacion:
#   - "en la duda, NO" a secas hacia al modelo descartar las ordenes lanzadas
#     al aire («¿que hora es?»), que son EL caso de uso. La regla principal
#     ahora es la contraria: orden o pregunta sin destinatario humano -> SI.
#   - «oye» sin nombre detras apunta al asistente («oye, ¿cuanto consume mi
#     servidor?» es el ejemplo canonico del encargo); con nombre detras
#     («oye Ana...») deja de apuntar. El modelo lo distingue si se le dice.
COMPUERTA_SISTEMA = """\
Eres la compuerta de un asistente de voz doméstico. Su micrófono está siempre \
abierto y oye TODO: órdenes al asistente, conversaciones entre personas, \
teléfono, tele, gente hablando sola. Decide si lo oído va dirigido AL \
ASISTENTE.

Regla principal: una pregunta o una orden lanzada al aire, sin nombrar a otra \
persona, va dirigida al asistente: «¿qué hora es?», «apaga las luces», «¿qué \
tal va el despliegue?». Un «oye» sin nombre detrás también apunta al \
asistente: «oye, ¿cuánto consume mi servidor?», «oye, ¿me miras una cosa?». \
Pedirle repetir o hablar de nuevo también: «¿me repites lo último que has \
dicho?». Y las respuestas a algo que el asistente acaba de decir o \
preguntar, incluidas las correcciones justo después de una respuesta suya: \
«¿y en euros?», «vale, hazlo», «solo los titulares», «gracias», «no, me \
refería a esta semana».

NO va dirigido a él cuando hay señales de otro destinatario o de charla \
humana: un vocativo con nombre («Ana, pásame el pan», «oye Ana, ¿me pasas el \
pan?»), planes o charla entre personas («¿quieres que pidamos pizza?», «y \
entonces le dije que ni hablar»), hablar de uno mismo en voz alta («mañana \
tengo que llamar al dentista»), muletillas de una conversación ajena («sí, \
sí, claro»), o hablar DEL asistente con otra persona («Ana, este trasto dice \
que gasta cuarenta vatios»)."""


def _guion_compuerta(texto, historial, hablante, segundos_desde_respuesta):
    """OJO CON CADA PALABRA DE ESTE GUION, que esta afinado a base de fallos:

      - "Se oye: «...»" y NO "Se oye a Juan: «...»". Medido con qwen3:4b:
        el nombre delante hace creer al modelo que Juan le habla a OTRA
        persona y «¿Qué hora es?» pasaba de SI a NO solo por nombrarlo. El
        nombre del hablante importa para RESPONDER (viaja aparte), no aqui.
      - Nada de "aun no ha habido conversacion": esa frase tambien inclinaba
        al modelo hacia el NO. Si no hay historial, no se dice nada.
      - Sin "SI o NO" al final: la salida estructurada ya fuerza el formato,
        y la coletilla confundia. (El aviso de SI/NO de MiniMax va en su
        propio sistema.)
    """
    lineas = []
    recorte = (historial or [])[-RECUERDO_COMPUERTA:]
    if recorte:
        lineas.append("Conversación reciente:")
        for h in recorte:
            asistente = h.get("rol") == "asistente"
            quien = "asistente" if asistente else (h.get("quien") or "alguien")
            # El corte se le enseña también a la compuerta: sin él, un «sigue»
            # detrás de una respuesta a medias parece una frase suelta sin
            # destinatario. Con él es obvio a quién se le habla.
            dicho = (texto_para_la_compuerta(h) if asistente
                     else h.get("texto", ""))
            lineas.append(f"  {quien}: {dicho}")
        if segundos_desde_respuesta is not None:
            lineas.append("(el asistente terminó de hablar hace "
                          f"{segundos_desde_respuesta:.0f} s)")
    lineas.append(f"Se oye: «{texto}»")
    lineas.append("¿Va dirigida al asistente?")
    return "\n".join(lineas)


def decidir(texto, historial, hablante, modelo, url_ollama,
            segundos_desde_respuesta=None):
    """(dirigida, segundos, crudo). En caso de error, la compuerta se abre:
    mejor una respuesta de más que un asistente sordo por una avería."""
    guion = _guion_compuerta(texto, historial, hablante,
                             segundos_desde_respuesta)
    t0 = time.perf_counter()
    try:
        if modelo.lower().startswith("minimax"):
            crudo = _decidir_minimax(guion, modelo)
        elif _es_llamacpp(url_ollama):
            crudo = _decidir_llamacpp(guion, url_ollama)
        else:
            crudo = _decidir_ollama(guion, modelo, url_ollama)
    except Exception as e:
        return True, time.perf_counter() - t0, f"error ({type(e).__name__}: {e})"
    crudo = (crudo or "").strip()
    # Ollama con salida estructurada devuelve {"dirigida": bool}; MiniMax,
    # "SI"/"NO" en texto. Y si un modelo se enrolla, la primera palabra.
    try:
        return bool(json.loads(crudo).get("dirigida")), \
            time.perf_counter() - t0, crudo
    except (ValueError, AttributeError):
        pass
    palabra = re.sub(r"[^a-zA-ZíÍ]", " ", crudo).split()
    dirigida = bool(palabra) and palabra[0].lower() in ("si", "sí")
    return dirigida, time.perf_counter() - t0, crudo


def _decidir_ollama(guion, modelo, url):
    # think:false apaga el razonamiento DE BLOQUE, pero qwen3:4b se pone
    # entonces a razonar en prosa ("Okay, let's see...") y el veredicto no
    # llega. La salida estructurada (format con esquema) se lo impide: la
    # decodificacion restringida solo puede emitir el JSON pedido.
    cuerpo = {"model": modelo, "stream": False, "think": False,
              "system": COMPUERTA_SISTEMA, "prompt": guion,
              "format": {"type": "object",
                         "properties": {"dirigida": {"type": "boolean"}},
                         "required": ["dirigida"]},
              # temperatura 0: un clasificador no echa monedas al aire.
              "options": {"num_predict": 20, "temperature": 0}}
    pet = urllib.request.Request(f"{url}/api/generate", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(pet, timeout=60))
    return d.get("response", "")


_TIPO_URL = {}


def _es_llamacpp(url):
    """True si en la URL escucha un llama-server (llama.cpp), no Ollama.

    Se distingue UNA vez por proceso con GET /props: llama-server lo contesta
    y Ollama devuelve 404. Un fallo de conexion no se cachea -- lo recoge el
    try de decidir(), que ante averia abre la compuerta."""
    if url not in _TIPO_URL:
        try:
            with urllib.request.urlopen(f"{url}/props", timeout=5) as r:
                _TIPO_URL[url] = r.status == 200
        except urllib.error.HTTPError:
            _TIPO_URL[url] = False
    return _TIPO_URL[url]


def _decidir_llamacpp(guion, url):
    # El mismo esquema que con Ollama, por el endpoint OpenAI de llama-server.
    # max_tokens corto A PROPOSITO: la gramatica del esquema permite blancos
    # tras cerrar el JSON y el 2507 los emite en vez del EOS justo en los
    # casos NO (medido: sin tope se va a ~2 s por decision; con tope, 0,5 s).
    cuerpo = {"model": "compuerta", "stream": False,
              "messages": [{"role": "system", "content": COMPUERTA_SISTEMA},
                           {"role": "user", "content": guion}],
              "temperature": 0, "max_tokens": 12,
              "json_schema": {"type": "object",
                              "properties": {"dirigida": {"type": "boolean"}},
                              "required": ["dirigida"]}}
    pet = urllib.request.Request(f"{url}/v1/chat/completions", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(pet, timeout=60))
    crudo = d["choices"][0]["message"]["content"].strip()
    # Si el tope truncó el JSON, el true/false ya emitido decide igual: la
    # gramatica garantiza que solo puede aparecer el del veredicto.
    try:
        json.loads(crudo)
    except ValueError:
        bajo = crudo.lower()
        if "true" in bajo or "false" in bajo:
            return json.dumps({"dirigida": "true" in bajo})
    return crudo


def _decidir_minimax(guion, modelo):
    # thinking desactivado; aun así el modelo piensa algo (medido: ~3,3 s),
    # de modo que max_tokens tiene que dejar sitio al pensamiento Y al SI/NO.
    cuerpo = {"model": modelo, "max_tokens": 700, "stream": False,
              "thinking": {"type": "disabled"},
              "system": COMPUERTA_SISTEMA +
              "\nResponde EXACTAMENTE una palabra: SI o NO.",
              "messages": [{"role": "user", "content": guion}]}
    pet = urllib.request.Request(
        MINIMAX_API, method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 "anthropic-version": "2023-06-01", "x-api-key": clave_minimax()})
    d = json.load(urllib.request.urlopen(pet, timeout=60))
    return "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text")


# ---- la respuesta, con historial ---------------------------------------

def _con_hablante(h):
    """Apunte de historial -> texto con quién lo dijo, si se sabe. El nombre
    viaja DENTRO del contenido -- [Juan]: ... -- porque los papeles de la API
    solo distinguen usuario y asistente, y aquí puede haber varias personas."""
    quien = h.get("quien")
    texto = h.get("texto", "")
    return f"[{quien}]: {texto}" if quien else texto


def preguntar_con_historial(pregunta, historial, modelo, url_ollama,
                            sistema=None, hablante=None):
    """Como asistente.preguntar(), pero con la conversación detrás.
    Devuelve trozos de texto según los escribe el modelo."""
    recorte = (historial or [])[-RECUERDO_RESPUESTA:]
    if hablante:
        sistema = ((sistema or "") + " Estás en una conversación de voz con "
                   f"varias personas; ahora mismo te habla {hablante}.")
    # Si la última respuesta se quedó a medias, el modelo tiene que saber por
    # dónde iba -- y saberlo COMO INSTRUCCIÓN, no como algo que él dijo. Ver
    # la cabecera de este fichero y interrupcion.nota_de_corte.
    nota = nota_de_corte(recorte)
    if nota:
        sistema = (sistema or "") + nota
    if modelo.lower().startswith("minimax"):
        return _historial_minimax(pregunta, recorte, modelo, sistema, hablante)
    return _historial_ollama(pregunta, recorte, modelo, url_ollama, sistema,
                             hablante)


def _historial_minimax(pregunta, recorte, modelo, sistema, hablante,
                       maximo=1024):
    # La API exige papeles alternados: apuntes seguidos del mismo lado se
    # funden en uno. Pasa si la compuerta dejó pasar dos frases seguidas de
    # personas distintas sin respuesta en medio.
    mensajes = []
    for h in recorte:
        papel = "assistant" if h.get("rol") == "asistente" else "user"
        contenido = (h.get("texto", "") if papel == "assistant"
                     else _con_hablante(h))
        if mensajes and mensajes[-1]["role"] == papel:
            mensajes[-1]["content"] += "\n" + contenido
        else:
            mensajes.append({"role": papel, "content": contenido})
    ahora = f"[{hablante}]: {pregunta}" if hablante else pregunta
    if mensajes and mensajes[-1]["role"] == "user":
        mensajes[-1]["content"] += "\n" + ahora
    else:
        mensajes.append({"role": "user", "content": ahora})
    cuerpo = {"model": modelo, "max_tokens": maximo, "stream": True,
              "messages": mensajes}
    if sistema:
        cuerpo["system"] = sistema
    pet = urllib.request.Request(
        MINIMAX_API, method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 "anthropic-version": "2023-06-01", "x-api-key": clave_minimax()})
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


def _historial_ollama(pregunta, recorte, modelo, url, sistema, hablante):
    # /api/chat y no /api/generate: es el endpoint que entiende de mensajes.
    mensajes = []
    if sistema:
        mensajes.append({"role": "system", "content": sistema})
    for h in recorte:
        papel = "assistant" if h.get("rol") == "asistente" else "user"
        contenido = (h.get("texto", "") if papel == "assistant"
                     else _con_hablante(h))
        mensajes.append({"role": papel, "content": contenido})
    mensajes.append({"role": "user",
                     "content": f"[{hablante}]: {pregunta}" if hablante else pregunta})
    cuerpo = {"model": modelo, "stream": True, "messages": mensajes}
    pet = urllib.request.Request(f"{url}/api/chat", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(pet, timeout=300)
    for linea in r:
        if not linea.strip():
            continue
        d = json.loads(linea)
        contenido = (d.get("message") or {}).get("content")
        if contenido:
            yield contenido
        if d.get("done"):
            break
