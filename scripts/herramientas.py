#!/usr/bin/env python
"""HERRAMIENTAS DEL ASISTENTE: el catalogo, el ejecutor y el ciclo del LLM.

    pkgs/vibevoice/.venv/bin/python scripts/herramientas.py listar
    pkgs/vibevoice/.venv/bin/python scripts/herramientas.py llamar consultar_calendario dia=manana
    pkgs/vibevoice/.venv/bin/python scripts/herramientas.py ciclo "¿que tengo manana?"

QUE ES ESTO Y QUE NO ES
Es el hueco que perfiles.py dejo declarado y vacio ("HERRAMIENTAS: EL HUECO
ESTA HECHO, Y ESTA VACIO A PROPOSITO"), ya lleno. Lo que hay dentro son
herramientas SIMULADAS: contestan con datos de mentira que viven en
herramientas_simuladas.json, en la raiz del repo. Lo simulado son los DATOS;
el mecanismo es el de verdad -- el modelo decide, se ejecuta, se le devuelve
el resultado y contesta con el -- porque eso es justo lo que hay que poder
probar antes de conectar nada real. Cada herramienta lleva escrito en
`sustituir_por` que hay que cambiar el dia que se enchufe de verdad.

POR QUE USO DE HERRAMIENTAS NATIVO Y NO "escribe JSON y yo lo parseo"
MiniMax sirve la API de Anthropic, que lleva uso de herramientas nativo. La
diferencia no es de comodidad, es de SEGURIDAD DEL AUDIO: el modelo emite los
argumentos en bloques `tool_use` / `input_json_delta`, que son un canal
distinto del texto. Como el narrador solo se alimenta de `text_delta`, el JSON
de la herramienta NO PUEDE acabar en el altavoz aunque el modelo se equivoque.
Con el truco de "que escriba JSON y lo saco con una expresion regular", el
JSON viaja POR el canal de texto y basta un fallo del filtro para que el
asistente se ponga a leer llaves y comillas en voz alta. Ese es el fallo
tipico de este montaje y aqui es imposible por construccion, no por filtro.

Y de regalo: el modelo suele escribir una frase ANTES de llamar a la
herramienta ("voy a mirar tu calendario"). Esa frase es texto normal, sale por
`text_delta`, y por tanto se narra ya -- medido, a 0,7 s de la pregunta --
mientras la herramienta se ejecuta y el modelo redacta la respuesta de verdad.
Es un relleno gratis, escrito por el propio modelo y a medida de lo que se le
ha pedido.

LA REGLA DE ORO: LEER ES GRATIS, ESCRIBIR SE CONFIRMA EN VOZ ALTA
whisper se equivoca. En este proyecto esta medido: 9,7 % de WER medio y 11,1 %
en el peor caso con el prompt bueno. Un 10 % de palabras mal en "borra la nota
del huerto" es una frase que puede no ser la que se dijo. Leer un dato mal
entendido cuesta una respuesta absurda y una risa; MANDAR UN CORREO mal
entendido no se deshace. Por eso:

  - Las herramientas de lectura (`escribe: False`) se ejecutan y ya esta.
  - Las de escritura NO EJECUTAN NADA. Nunca, ni con el mejor de los
    argumentos. Lo unico que hacen es PREPARAR: anotan lo que se haria y
    devuelven un resumen en castellano.
  - Lo preparado lo remata `Ejecutor.confirmar()`, y esa es la unica funcion
    del programa que escribe. No hay otra rama que llegue alli.

EL SI NO PASA POR EL MODELO, Y ESO COSTO TRES INTENTOS
El diseño evidente -- que el modelo pregunte y que el modelo recoja el si --
NO FUNCIONA. Medido contra MiniMax-M3, en este orden:

  1. Con un parametro `confirmado: true` en cada herramienta de escritura, el
     modelo veia una herramienta "peligrosa" con una bandera y se volvia
     prudente de mas: preguntaba POR SU CUENTA sin llamarla nunca. El usuario
     decia que si, el ejecutor no tenia nada anotado, y habia que preguntar
     otra vez. Bucle: 0 de 4 acciones completadas.
  2. Partiendolo en dos herramientas (preparar + `confirmar_accion`) se
     arreglo lo primero -- el modelo llama sin miedo a algo que no puede
     hacer daño -- pero aparecio lo peor de todo: con un "si, borrala" ya en
     el historial, el modelo llamaba a la herramienta, LEIA
     "preparada_sin_ejecutar" y aun asi remataba con «Hecho, queda borrada».
     Mentia. Un asistente que dice que ha borrado algo que no ha borrado es
     mas dañino que uno que no borra.
  3. Lo que si funciona: el modelo NO participa en la confirmacion. Llama a la
     herramienta, la herramienta prepara, y AHI SE ACABA SU TURNO. La pregunta
     ("voy a X, ¿lo hago?") la compone el programa con el mismo resumen que se
     va a ejecutar, y la respuesta la clasifica clasificar_respuesta(), que es
     una lista cerrada de palabras y ni siquiera toca la red.

De regalo, el remate sale casi gratis. "Enviado." es una de las frases
pregeneradas del perfil (categoria 'confirmando'), asi que ni LLM, ni sesion de
voz, ni VM: el navegador la tiene decodificada desde que abrio la pagina.
Medido, 0,004 s de media contra los 1,2-2 s que costaria una vuelta mas.

LO QUE NO ESTA EN LA LISTA NO ES UN SI. clasificar_respuesta devuelve 'otra'
para cualquier cosa que no sea un si o un no reconocibles, y 'otra' DESCARTA
la accion y trata la frase como una pregunta nueva. Equivocarse hacia "no te
he entendido" cuesta repetir; equivocarse hacia "si" cuesta un correo enviado.

Al confirmar se ejecutan los argumentos QUE SE DIJERON EN VOZ ALTA, guardados
en el pendiente: lo que se oye y lo que se hace salen del mismo sitio y no
pueden separarse ni por una palabra.

Y UNA CONFIRMACION SOLO VALE EN EL TURNO SIGUIENTE (ademas de caducar a los 5
minutos). Un "si" que llega media conversacion despues no es un si a esto: es
un si a otra cosa que se ha cruzado por el camino.
"""
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asistente import MINIMAX_API, clave_minimax  # noqa: E402

# Cuanto vive una confirmacion pendiente. Ver la cabecera.
CADUCIDAD_CONFIRMACION = 300.0

# Vueltas maximas del ciclo modelo -> herramienta -> modelo. Tres bastan para
# "mira el calendario y el correo y dime que hago hoy" (dos lecturas y la
# respuesta) y ponen techo a un modelo que se enrede llamando en bucle: cada
# vuelta son 0,7-2 s que el usuario espera callado.
VUELTAS_MAXIMAS = 4

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha(offset: int) -> str:
    d = date.today() + timedelta(days=offset)
    return f"{DIAS[d.weekday()]} {d.day} de {MESES[d.month - 1]}"


def _sin_tildes(t: str) -> str:
    tabla = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return (t or "").translate(tabla).lower().strip()


# Las palabras que no distinguen nada. Sin quitarlas, "la mudanza del NAS"
# casa con cualquier nota que lleve un "de".
VACIAS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
          "al", "a", "en", "y", "o", "que", "lo", "con", "por", "para", "mi",
          "mis", "su", "sus", "sobre", "cosa", "tema", "todo", "todos"}


def _palabras(t: str) -> list:
    return [p for p in re.split(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+", _sin_tildes(t))
            if len(p) >= 3 and p not in VACIAS]


def _casa(consulta: str, *campos) -> int:
    """Cuantas palabras de la consulta aparecen en los campos.

    Se busca POR PALABRAS y no por la frase entera A PROPOSITO: el modelo
    manda lo que ha entendido ("mudanza NAS") y el titulo es otro ("Mudanza
    del NAS"). Con subcadena eso da cero resultados y el asistente contesta
    "no encuentro nada" teniendo la nota delante -- pasa siempre, y es el
    fallo que hace que las herramientas parezcan tontas."""
    heno = _sin_tildes(" ".join(str(c) for c in campos if c))
    ps = _palabras(consulta)
    if not ps:
        return 0
    return sum(1 for p in ps if p in heno)


# ---------------------------------------------------------------------------
# EL CATALOGO
#
# `esquema` es JSON Schema tal cual lo pide la API de Anthropic. La descripcion
# es lo UNICO que el modelo lee para decidir si una herramienta le sirve, asi
# que esta escrita para el y no para nosotros: dice cuando usarla y cuando no.
#
# `escribe` es el interruptor de la regla de seguridad: True obliga a
# confirmar en voz alta. No se deduce del nombre a proposito -- que una
# herramienta sea peligrosa tiene que estar DICHO, no adivinado.
#
# `dominio` es lo que se pinta en la pagina y lo que agrupa las herramientas
# por perfil.
# ---------------------------------------------------------------------------
CATALOGO = [
    {
        "nombre": "consultar_calendario",
        "dominio": "calendario",
        "escribe": False,
        "sustituir_por": "CalDAV o la API de Google Calendar. Devuelve la "
                         "misma lista de eventos y no cambia nada mas.",
        "descripcion":
            "Consulta los eventos del calendario del usuario. Úsala siempre "
            "que te pregunten por su agenda, por si tiene algo a una hora, o "
            "por si está libre. No la uses para crear nada.",
        "esquema": {
            "type": "object",
            "properties": {
                "cuando": {
                    "type": "string",
                    "description": "Qué tramo mirar: 'hoy', 'mañana', "
                                   "'pasado', 'semana' (los próximos siete "
                                   "días) o un número de días desde hoy.",
                },
            },
            "required": ["cuando"],
        },
    },
    {
        "nombre": "crear_evento",
        "dominio": "calendario",
        "escribe": True,
        "sustituir_por": "CalDAV (PUT de un .ics) o calendar.events.insert.",
        "descripcion":
            "Apunta un evento nuevo en el calendario del usuario. Úsala en "
            "cuanto te pidan poner algo en la agenda, con los datos que te "
            "hayan dado; lo que falte, ponlo tú con sentido común. ""Llámala aunque pienses preguntar después: es ella la que te "
            "da el texto exacto que hay que decir, y preguntar sin haberla "
            "llamado obliga a preguntar dos veces.",
        "esquema": {
            "type": "object",
            "properties": {
                "titulo": {"type": "string", "description": "De qué es."},
                "cuando": {"type": "string",
                           "description": "'hoy', 'mañana', 'pasado' o días desde hoy."},
                "hora": {"type": "string", "description": "En formato 24 h, p. ej. '17:30'."},
                "duracion_min": {"type": "integer", "description": "Minutos. Por defecto 60."},
                "con": {"type": "array", "items": {"type": "string"},
                        "description": "Quién va, si se ha dicho."},
            },
            "required": ["titulo", "cuando", "hora"],
        },
    },
    {
        "nombre": "leer_correo",
        "dominio": "correo",
        "escribe": False,
        "sustituir_por": "IMAP, o el conector de Gmail. El resumen de cada "
                         "mensaje lo hace hoy un humano en el JSON; con "
                         "correo real lo hace el propio modelo.",
        "descripcion":
            "Lee la bandeja de entrada del usuario: remitentes, asuntos y un "
            "resumen de cada mensaje. Úsala para '¿tengo correo?', '¿qué "
            "quería fulano?' o '¿hay algo urgente?'.",
        "esquema": {
            "type": "object",
            "properties": {
                "filtro": {
                    "type": "string",
                    "description": "'nuevos' (sin leer, es lo normal), "
                                   "'importantes', 'todos', o un nombre o "
                                   "palabra para buscar en remitente y asunto.",
                },
                "cuantos": {"type": "integer",
                            "description": "Cuántos como mucho. Por defecto 5."},
            },
        },
    },
    {
        "nombre": "enviar_correo",
        "dominio": "correo",
        "escribe": True,
        "sustituir_por": "SMTP, o gmail.users.messages.send.",
        "descripcion":
            "Manda un correo. Úsala en cuanto te pidan escribir a alguien: "
            "redacta tú el asunto y el cuerpo con lo que te hayan contado, "
            "corto y en castellano. ""Llámala aunque pienses preguntar después: es ella la que te "
            "da el texto exacto que hay que decir, y preguntar sin haberla "
            "llamado obliga a preguntar dos veces.",
        "esquema": {
            "type": "object",
            "properties": {
                "para": {"type": "string",
                         "description": "El NOMBRE de la persona basta ('Elena', "
                                        "'Diego'): la dirección se resuelve sola "
                                        "en la agenda. No se la pidas al usuario."},
                "asunto": {"type": "string"},
                "cuerpo": {"type": "string", "description": "El texto del mensaje."},
            },
            "required": ["para", "asunto", "cuerpo"],
        },
    },
    {
        "nombre": "estado_servidor",
        "dominio": "servidor",
        "escribe": False,
        "sustituir_por": "systemctl y df por SSH, o el /health de la propia "
                         "VM (http://192.168.2.54:8082/health) mas un "
                         "exportador de nodo.",
        "descripcion":
            "Estado de la máquina de casa: unidades de systemd, carga, "
            "memoria y disco. Úsala para '¿cómo va el servidor?', '¿se ha "
            "caído algo?' o '¿cuánto disco queda?'.",
        "esquema": {
            "type": "object",
            "properties": {
                "que": {
                    "type": "string",
                    "description": "'resumen' (por defecto), 'servicios', "
                                   "'disco', 'carga', o el nombre de una "
                                   "unidad concreta.",
                },
            },
        },
    },
    {
        "nombre": "buscar_notas",
        "dominio": "notas",
        "escribe": False,
        "sustituir_por": "el MCP de Obsidian que ya esta declarado en el "
                         "perfil 'servidor' (obsidian_notas): search_vault y "
                         "get_vault_file.",
        "descripcion":
            "Busca en las notas de proyectos del usuario: de qué va cada uno, "
            "cuándo se tocó por última vez y qué queda pendiente. Úsala para "
            "'¿por dónde iba lo del NAS?' o '¿qué tengo abandonado?'.",
        "esquema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string",
                             "description": "Qué buscar. Vacío o 'todo' devuelve todas."},
            },
        },
    },
    {
        "nombre": "borrar_nota",
        "dominio": "notas",
        "escribe": True,
        "sustituir_por": "delete_vault_file del MCP de Obsidian.",
        "descripcion":
            "Borra una nota de la bóveda. Úsala en cuanto te pidan quitar "
            "una nota, y llámala DIRECTAMENTE: no hace falta que sepas el "
            "título exacto, pásale lo que te hayan dicho y ella lo resuelve "
            "sola. No busques antes con buscar_notas. "
            "Llámala aunque pienses preguntar después: es ella la que te "
            "da el texto exacto que hay que decir, y preguntar sin haberla "
            "llamado obliga a preguntar dos veces.",
        "esquema": {
            "type": "object",
            "properties": {
                "titulo": {"type": "string",
                           "description": "Título de la nota, tal y como salió en buscar_notas."},
            },
            "required": ["titulo"],
        },
    },
    {
        "nombre": "ver_recordatorios",
        "dominio": "recordatorios",
        "escribe": False,
        "sustituir_por": "la app de Recordatorios por AppleScript, o "
                         "cualquier lista en CalDAV VTODO.",
        "descripcion":
            "Lista los recordatorios pendientes del usuario, y dice cuáles "
            "están vencidos. Úsala para '¿qué tengo pendiente?'.",
        "esquema": {
            "type": "object",
            "properties": {
                "filtro": {"type": "string",
                           "description": "'pendientes' (por defecto), "
                                          "'vencidos', 'hoy' o 'todos'."},
            },
        },
    },
    {
        "nombre": "crear_recordatorio",
        "dominio": "recordatorios",
        "escribe": True,
        "sustituir_por": "la app de Recordatorios por AppleScript.",
        "descripcion":
            "Apunta un recordatorio. Úsala en cuanto te pidan que les "
            "recuerdes algo o que lo apuntes. ""Llámala aunque pienses preguntar después: es ella la que te "
            "da el texto exacto que hay que decir, y preguntar sin haberla "
            "llamado obliga a preguntar dos veces.",
        "esquema": {
            "type": "object",
            "properties": {
                "texto": {"type": "string", "description": "Qué hay que recordar."},
                "cuando": {"type": "string",
                           "description": "Cuándo avisar, tal y como lo dijo el "
                                          "usuario. Puede ir vacío."},
            },
            "required": ["texto"],
        },
    },
    {
        "nombre": "buscar_en_web",
        "dominio": "web",
        "escribe": False,
        "sustituir_por": "una API de busqueda (Brave, Tavily, SearXNG en la "
                         "propia LAN). Es la unica herramienta que de verdad "
                         "tarda: cuenta con 0,5-2 s mas de espera.",
        "descripcion":
            "Busca en internet y devuelve un resumen y unos cuantos "
            "resultados. Úsala cuando te pregunten algo que no está en el "
            "calendario, el correo, las notas ni el servidor: noticias, "
            "datos, cómo se hace algo.",
        "esquema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "Qué buscar."},
            },
            "required": ["consulta"],
        },
    },
    # NO HAY HERRAMIENTA DE CONFIRMACION, Y ESO ES EL DISEÑO.
    # Hubo una, `confirmar_accion`, y se quito: ver "EL SI NO PASA POR EL
    # MODELO" en la cabecera. El si y el no los resuelve el puente con
    # clasificar_respuesta() y Ejecutor.confirmar(), sin LLM de por medio.
]

POR_NOMBRE = {h["nombre"]: h for h in CATALOGO}
DOMINIOS = sorted({h["dominio"] for h in CATALOGO})
ESCRITURAS = [h["nombre"] for h in CATALOGO if h["escribe"]]
# Las que van en todos los juegos pase lo que pase.
SIEMPRE = [h["nombre"] for h in CATALOGO if h.get("siempre")]

# Lo que se dice al rematar. CORTO, y fijo: es la unica parte de la respuesta
# que no puede permitirse un matiz. "Enviado" no admite dos lecturas.
HECHO = {"crear_evento": "Apuntado.", "enviar_correo": "Enviado.",
         "crear_recordatorio": "Apuntado.", "borrar_nota": "Borrada."}


# ------------------------------------------------- el si y el no, sin LLM --
# Ver "EL SI NO PASA POR EL MODELO" en la cabecera. Las listas son cerradas a
# proposito: lo que no este aqui NO es un si, y por tanto no ejecuta nada.
# Equivocarse hacia "no lo he entendido" cuesta repetir la frase; equivocarse
# hacia "si" cuesta un correo enviado.
_SI = {"si", "sii", "siii", "claro", "vale", "venga", "dale", "adelante",
       "hazlo", "mandalo", "enviala", "envialo", "borrala", "borralo",
       "apuntalo", "creala", "crealo", "ponlo", "ponla", "guardalo",
       "guardala", "confirmo", "confirmado", "correcto", "exacto", "eso es",
       "perfecto", "ok", "okey", "de acuerdo", "por supuesto", "afirmativo",
       "sip", "sisi", "si si", "venga si", "si por favor", "si claro",
       "si hazlo", "si mandalo", "si envialo", "si borrala", "si adelante",
       "que si", "tira", "palante", "pa lante"}
_NO = {"no", "nop", "nel", "para", "dejalo", "deja", "cancela", "cancelalo",
       "anula", "anulalo", "olvidalo", "olvida", "mejor no", "no hace falta",
       "no lo hagas", "no lo mandes", "no lo envies", "no la borres",
       "no gracias", "no no", "que no", "espera", "quieto", "todavia no",
       "aun no", "ahora no", "dejalo estar", "nada", "negativo"}
# Lo que suele acompañar al si o al no sin cambiarlo. Se quita antes de mirar.
_ADORNO = re.compile(r"^(pues|bueno|eh|em|mmm|a ver|hombre|oye|venga va)\s+|"
                     r"\s+(porfa|por favor|gracias|anda|hombre|please)$")


def pregunta_de_confirmacion(resumen: str) -> str:
    """Lo que se dice en voz alta antes de escribir. Sale del MISMO resumen
    que se ejecutara, asi que lo que se oye y lo que se hace no pueden
    separarse ni por una palabra."""
    r = (resumen or "").rstrip(" .")
    return f" Te lo repito antes de hacerlo: voy a {r}. ¿Lo hago?"


def nota_de_pregunta(resumen: str) -> str:
    """Como se guarda esa pregunta en el historial DEL MODELO. Ver _mensajes:
    en tercera persona, para que no la copie como si fuera su estilo."""
    return (f"(el sistema le preguntó al usuario si confirmaba «{resumen}»; "
            f"esta frase no la dijiste tú)")


def nota_de_remate(resumen: str, hecho: bool) -> str:
    if hecho:
        return (f"El sistema aplicó «{resumen}» tras el sí del usuario, y ya "
                f"se lo ha dicho. No lo repitas ni lo vuelvas a hacer.")
    return (f"El usuario dijo que no a «{resumen}» y el sistema la descartó. "
            f"No la vuelvas a preparar si no te lo pide otra vez.")


def notas_recientes(historial, cuantas=2) -> str:
    """Los hechos que dejaron las frases del programa, para el bloque de
    sistema. Ver _mensajes: ahi es donde SI se pueden contar sin que el modelo
    los tome por un ejemplo de como contestar."""
    ns = [h["nota"] for h in (historial or []) if h.get("nota")]
    return " ".join(ns[-cuantas:])


def clasificar_respuesta(texto: str) -> str:
    """'si', 'no' u 'otra'. Deterministico, sin modelo y sin red.

    'otra' es el caso importante: cualquier cosa que no sea un si o un no
    reconocible descarta la accion y se trata como una pregunta nueva. El
    silencio, un cambio de tema o un "¿cómo?" NO ejecutan nada."""
    t = _sin_tildes(texto or "")
    t = re.sub(r"[¿?¡!.,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(3):                      # "pues venga, por favor"
        nuevo = _ADORNO.sub("", t).strip()
        if nuevo == t:
            break
        t = nuevo
    if not t:
        return "otra"
    if t in _SI:
        return "si"
    if t in _NO:
        return "no"
    # Frases de dos o tres palabras que empiezan por un si/no claro:
    # "si, mandalo", "no, dejalo". Mas largas ya son otra cosa y se tratan
    # como pregunta nueva.
    ps = t.split()
    if len(ps) <= 3:
        if ps[0] in ("si", "vale", "claro", "venga", "dale", "ok", "okey"):
            return "si"
        if ps[0] in ("no", "nop", "mejor", "cancela", "olvidalo", "dejalo"):
            return "no"
    return "otra"


def esquemas_anthropic(nombres=None) -> list:
    """El catalogo en el formato que espera la API: name/description/input_schema."""
    return [{"name": h["nombre"], "description": h["descripcion"],
             "input_schema": h["esquema"]}
            for h in CATALOGO if nombres is None or h["nombre"] in nombres]


def esquemas_ollama(nombres=None) -> list:
    """El mismo catalogo en el formato de Ollama (estilo OpenAI)."""
    return [{"type": "function",
             "function": {"name": h["nombre"], "description": h["descripcion"],
                          "parameters": h["esquema"]}}
            for h in CATALOGO if nombres is None or h["nombre"] in nombres]


def por_dominios(dominios) -> list:
    """Nombres de herramienta de unos dominios dados. Es como un perfil elige
    su juego: 'calendario, correo, recordatorios' y no las once. Las de
    `SIEMPRE` se cuelan aunque no se pidan."""
    dd = set(dominios or [])
    return [h["nombre"] for h in CATALOGO
            if h["dominio"] in dd or h.get("siempre")]


# ---------------------------------------------------------------- los datos --
def ruta_datos(explicita=None) -> Path:
    if explicita:
        return Path(explicita).expanduser()
    v = os.environ.get("VOZ_HERRAMIENTAS_DATOS")
    if v:
        return Path(v).expanduser()
    return Path(__file__).resolve().parent.parent / "herramientas_simuladas.json"


class Simulacion:
    """Los datos de mentira, releidos solos cuando cambia el fichero.

    Se recarga por fecha de modificacion y no por reinicio A PROPOSITO: la
    gracia de tener los datos fuera del codigo es poder retocarlos mientras se
    enseña la demo -- añadir un evento, tumbar un servicio -- y que la
    siguiente pregunta ya conteste con eso."""

    def __init__(self, ruta=None):
        self.ruta = ruta_datos(ruta)
        self._mtime = 0.0
        self._datos = {}

    @property
    def datos(self) -> dict:
        try:
            m = self.ruta.stat().st_mtime
        except OSError as e:
            raise RuntimeError(f"no se puede leer {self.ruta}: {e}")
        if m != self._mtime:
            self._datos = json.loads(self.ruta.read_text(encoding="utf-8"))
            self._mtime = m
        return self._datos

    def guardar(self) -> None:
        """Escribe el fichero. Solo lo llaman las herramientas de escritura, y
        solo para APUNTAR lo que han hecho: asi 'mándale un correo a Elena' y,
        despues, '¿le mandé algo a Elena?' cuadran entre si."""
        self.ruta.write_text(
            json.dumps(self._datos, ensure_ascii=False, indent=1), encoding="utf-8")
        self._mtime = self.ruta.stat().st_mtime


# ------------------------------------------------------------- el ejecutor --
class SinConfirmar(Exception):
    """No es un error: es la mitad buena del asunto. La levanta el ejecutor
    cuando una escritura necesita el si del usuario."""

    def __init__(self, resumen):
        super().__init__(resumen)
        self.resumen = resumen


def _cuando_a_dias(cuando) -> int | None:
    """'hoy'/'mañana'/'3' -> desplazamiento en dias. None si no se entiende."""
    if cuando is None:
        return 0
    if isinstance(cuando, (int, float)):
        return int(cuando)
    t = _sin_tildes(str(cuando))
    if t in ("", "hoy", "hoy mismo", "este dia"):
        return 0
    if t in ("manana", "el dia siguiente"):
        return 1
    if t in ("pasado", "pasado manana"):
        return 2
    if t == "ayer":
        return -1
    if t in ("semana", "esta semana", "proxima semana", "los proximos dias"):
        return None
    try:
        return int(t)
    except ValueError:
        for i, d in enumerate(DIAS):
            if d in t:            # "el jueves": el proximo que caiga
                hoy = date.today().weekday()
                return (i - hoy) % 7 or 7
    return 0


class Ejecutor:
    """Ejecuta una herramienta y devuelve (resultado, resumen_para_la_pagina).

    El `resumen` no es decoracion: es lo que se pinta en la pagina para que se
    VEA que se ha usado la herramienta y con que resultado. El `resultado` es
    lo que se le devuelve al modelo, y nunca pasa por el narrador.
    """

    def __init__(self, ruta=None, sim=None):
        self.sim = sim or Simulacion(ruta)
        self.pendientes = {}        # sesion -> accion a la espera del si
        self.turnos = {}            # sesion -> cuantas veces ha hablado el usuario
        self.registro = []          # ultimas llamadas, para la pagina y las pruebas
        # Que herramientas se llamaron en cada pregunta anterior. Ver
        # _mensajes: sin esto el modelo deja de llamarlas a los tres turnos.
        self.memoria = {}           # sesion -> [{pregunta, llamadas}]

    def recordar(self, sesion, pregunta, llamadas, tope=4) -> None:
        m = self.memoria.setdefault(sesion, [])
        m.append({"pregunta": pregunta, "llamadas": llamadas})
        del m[:-tope]

    def recordado(self, sesion) -> list:
        return self.memoria.get(sesion) or []

    # -- confirmaciones -----------------------------------------------------
    def turno(self, sesion="web") -> int:
        """El puente lo llama al empezar CADA intervencion del usuario.

        Sirve para que una confirmacion caduque enseguida: vale en el turno
        siguiente al que la pidio y en UNO MAS de gracia. Sin esto, un 'sí'
        que en realidad iba de otra cosa -- media conversacion despues --
        podia rematar una accion que el usuario ya habia olvidado. Un si
        tardio no es un si.

        El turno de gracia no sobra: con la escucha continua, un 'sí' puede
        gastarse sin confirmar nada porque la compuerta decide que iba
        dirigido a otra persona (ver el puente). Sin margen, esa intervencion
        ajena tiraria la accion y habria que pedirla otra vez."""
        self.turnos[sesion] = self.turnos.get(sesion, 0) + 1
        return self.turnos[sesion]

    def pendiente(self, sesion="web") -> dict | None:
        p = self.pendientes.get(sesion)
        if not p:
            return None
        viejo = (time.time() - p["t"] > CADUCIDAD_CONFIRMACION
                 or self.turnos.get(sesion, 0) - p["turno"] > 1)
        if viejo:
            del self.pendientes[sesion]
            return None
        return p

    def cancelar(self, sesion="web") -> bool:
        return self.pendientes.pop(sesion, None) is not None

    # -- la puerta ----------------------------------------------------------
    def ejecutar(self, nombre, args, sesion="web") -> tuple[dict, str]:
        h = POR_NOMBRE.get(nombre)
        if h is None:
            return ({"error": f"no existe la herramienta {nombre!r}",
                     "disponibles": sorted(POR_NOMBRE)}, f"{nombre}: no existe")
        args = dict(args or {})
        # `confirmado` ya no existe en ningun esquema. Si un modelo se lo
        # inventa, se tira: la confirmacion no se pide por parametro, se pide
        # por la herramienta confirmar_accion, que es la unica que ejecuta.
        args.pop("confirmado", None)
        t0 = time.time()
        if h["escribe"]:
            res, resumen = self._preparar(h, args, sesion)
        else:
            res, resumen = getattr(self, f"_{nombre}")(args), None
            resumen = resumen or self._resumen_lectura(nombre, args, res)
        self.registro.append({"nombre": nombre, "args": args, "resumen": resumen,
                              "ms": round((time.time() - t0) * 1000, 1),
                              "t": time.time()})
        del self.registro[:-40]
        return res, resumen

    def _preparar(self, h, args, sesion):
        """Una herramienta de escritura, llamada. NO EJECUTA NADA, NUNCA.

        Y no es una comprobacion que se pueda saltar: aqui no hay ninguna rama
        que llame al metodo que escribe. El unico sitio del programa donde se
        crea un evento, sale un correo o se borra una nota es
        _confirmar_accion(), y para llegar alli hace falta que ESTE metodo haya
        anotado antes lo que se iba a hacer. Un modelo que se equivoque, o al
        que le cuelen una instruccion, como mucho consigue que el asistente
        PREGUNTE."""
        nombre = h["nombre"]
        resumen = self.resumen_accion(nombre, args)
        self.pendientes[sesion] = {"nombre": nombre, "args": args,
                                   "resumen": resumen, "t": time.time(),
                                   "turno": self.turnos.get(sesion, 0)}
        return ({"estado": "preparada_sin_ejecutar",
                 "resumen": resumen,
                 "instruccion": "Todavía no se ha aplicado. Di este resumen "
                                "en voz alta con tus palabras y pregunta si lo "
                                "hace. No llames a nada más en este turno. "
                                "Cuando el usuario conteste, llama a "
                                "confirmar_accion con su respuesta."},
                f"⏸ {resumen} — esperando el sí")

    def confirmar(self, sesion="web", si=True) -> tuple[dict, str, str]:
        """El sí (o el no). Es el UNICO sitio del programa que escribe.

        Lo llama el PUENTE, no el modelo, y con un veredicto que sale de
        clasificar_respuesta(). Devuelve tambien la frase que hay que decir,
        que es fija y corta: sin LLM de por medio, el «hecho» se oye a los
        ~0,35 s en vez de a los 1,2-2 s que costaria una vuelta mas."""
        p = self.pendiente(sesion)
        if p is None:
            return {"estado": "nada_que_confirmar"}, "", ""
        del self.pendientes[sesion]
        if not si:
            return ({"estado": "descartada", "era": p["resumen"]},
                    f"✖ descartada: {p['resumen']}", "Vale, lo dejo.")
        # SE EJECUTA LO QUE SE DIJO EN VOZ ALTA, que es lo que el usuario oyo y
        # aprobo, palabra por palabra: `p["args"]` es lo que genero el resumen.
        res = getattr(self, f"_{p['nombre']}")(p["args"])
        res["estado"] = "hecho"
        res["confirmado_como"] = p["resumen"]
        return res, f"✔ {p['resumen']}", HECHO.get(p["nombre"], "Hecho.")

    def resumen_accion(self, nombre, args) -> str:
        """Una frase en castellano con lo que se va a hacer. Es lo que el
        usuario tiene que oir ANTES de que pase, asi que se escribe entera
        aqui y no se deja a que el modelo la improvise."""
        a = args or {}
        if nombre == "crear_evento":
            dias = _cuando_a_dias(a.get("cuando"))
            cuando = _fecha(dias if dias is not None else 0)
            con = a.get("con") or []
            return (f"crear «{a.get('titulo', 'sin título')}» el {cuando} a las "
                    f"{a.get('hora', 'sin hora')}"
                    + (f", con {', '.join(con)}" if con else ""))
        if nombre == "enviar_correo":
            # El cuerpo SE LEE ENTERO hasta donde da, y no se resume: lo que
            # se confirma es lo que se manda, palabra por palabra. Lo unico
            # que se hace es apretar los espacios -- el modelo escribe listas
            # con saltos de linea y guiones, y eso hablado son pausas raras.
            cuerpo = re.sub(r"\s*[-*]\s+", ", ", a.get("cuerpo") or "")
            cuerpo = re.sub(r"\s+", " ", cuerpo).strip().rstrip(".")
            if len(cuerpo) > 200:
                cuerpo = cuerpo[:197] + "…"
            c = self.contacto(a.get("para"))
            # El NOMBRE, nunca la direccion: esto se dice en voz alta y
            # "elena punto cardona arroba empresa punto com" no es una frase.
            quien = c["nombre"] if c else (a.get("para") or "?")
            return (f"enviar un correo a {quien}, asunto "
                    f"«{a.get('asunto', 'sin asunto')}», que dice: {cuerpo}")
        if nombre == "crear_recordatorio":
            cu = (a.get("cuando") or "").strip()
            return (f"crear el recordatorio «{a.get('texto', '?')}»"
                    + (f" para {cu}" if cu else " sin fecha"))
        if nombre == "borrar_nota":
            return f"BORRAR la nota «{a.get('titulo', '?')}»"
        return f"{nombre} con {json.dumps(a, ensure_ascii=False)}"

    def _resumen_lectura(self, nombre, args, res) -> str:
        n = res.get("cuantos")
        if n is None:
            n = len(res.get("resultados") or res.get("mensajes")
                    or res.get("eventos") or res.get("documentos")
                    or res.get("recordatorios") or [])
        detalle = {"consultar_calendario": args.get("cuando", "hoy"),
                   "leer_correo": args.get("filtro", "nuevos"),
                   "estado_servidor": args.get("que", "resumen"),
                   "buscar_notas": args.get("consulta", "todo"),
                   "ver_recordatorios": args.get("filtro", "pendientes"),
                   "buscar_en_web": args.get("consulta", "")}.get(nombre, "")
        cola = f" → {n}" if nombre != "estado_servidor" else ""
        return f"{nombre}({detalle}){cola}"

    # -- las herramientas de LECTURA ---------------------------------------
    def _consultar_calendario(self, a) -> dict:
        cal = self.sim.datos["calendario"]["eventos"]
        d = _cuando_a_dias(a.get("cuando"))
        if d is None:                       # 'semana': de hoy a +7
            sel = [e for e in cal if 0 <= e["dia"] <= 7]
            tramo = "los próximos siete días"
        else:
            sel = [e for e in cal if e["dia"] == d]
            tramo = _fecha(d)
        sel = sorted(sel, key=lambda e: (e["dia"], e["hora"]))
        creados = self.sim.datos["escrituras"]["eventos_creados"]
        for e in creados:
            if d is None and 0 <= e["dia"] <= 7 or e["dia"] == d:
                sel.append(dict(e, nota="creado por ti en esta conversación"))
        return {"tramo": tramo, "hoy_es": _fecha(0), "cuantos": len(sel),
                "eventos": [{"fecha": _fecha(e["dia"]), "hora": e["hora"],
                             "fin": e.get("fin"), "titulo": e["titulo"],
                             "donde": e.get("donde"), "con": e.get("con") or [],
                             "estado": e.get("estado", "confirmado"),
                             "nota": e.get("nota")} for e in sel]}

    def _leer_correo(self, a) -> dict:
        msgs = self.sim.datos["correo"]["mensajes"]
        filtro = (a.get("filtro") or "nuevos").strip()
        f = _sin_tildes(filtro)
        if f in ("nuevos", "sin leer", "no leidos", ""):
            sel = [m for m in msgs if not m["leido"]]
        elif f in ("importantes", "urgentes"):
            sel = [m for m in msgs if m.get("importante")]
        elif f in ("todos", "todo"):
            sel = list(msgs)
        else:
            sel = sorted((m for m in msgs
                          if _casa(filtro, m["de"], m["asunto"], m["resumen"])),
                         key=lambda m: -_casa(filtro, m["de"], m["asunto"],
                                              m["resumen"]))
        tope = int(a.get("cuantos") or 5)
        return {"filtro": filtro, "sin_leer": sum(1 for m in msgs if not m["leido"]),
                "cuantos": len(sel[:tope]), "de_un_total_de": len(sel),
                "mensajes": [{"de": m["de"], "asunto": m["asunto"],
                              "cuando": m["hace"], "importante": m.get("importante", False),
                              "leido": m["leido"], "resumen": m["resumen"]}
                             for m in sel[:tope]],
                "enviados_en_esta_conversacion":
                    self.sim.datos["escrituras"]["correos_enviados"]}

    def _estado_servidor(self, a) -> dict:
        s = self.sim.datos["servidor"]
        que = _sin_tildes(a.get("que") or "resumen")
        base = {"maquina": s["maquina"], "arriba_desde": s["arriba_desde"]}
        if que in ("servicios", "unidades", "systemd"):
            return dict(base, servicios=s["servicios"], avisos=s["avisos"])
        if que in ("disco", "espacio", "almacenamiento"):
            return dict(base, disco=s["disco"],
                        avisos=[x for x in s["avisos"] if "%" in x])
        if que in ("carga", "cpu", "memoria", "ram"):
            return dict(base, carga=s["carga"], nucleos=s["nucleos"],
                        memoria=s["memoria"])
        if que not in ("resumen", "", "estado", "todo"):
            hit = [u for u in s["servicios"] if que in _sin_tildes(u["nombre"])]
            if hit:
                return dict(base, servicios=hit)
        malos = [u for u in s["servicios"] if u["estado"] not in ("activo",)]
        return dict(base, carga=s["carga"], nucleos=s["nucleos"],
                    memoria=s["memoria"], disco=s["disco"],
                    servicios_con_algo=malos,
                    servicios_ok=[u["nombre"] for u in s["servicios"]
                                  if u["estado"] == "activo"],
                    avisos=s["avisos"])

    def _buscar_notas(self, a) -> dict:
        docs = self.sim.datos["notas"]["documentos"]
        q = _sin_tildes(a.get("consulta") or "")
        borradas = {n["titulo"] for n in self.sim.datos["escrituras"]["notas_borradas"]}
        docs = [d for d in docs if d["titulo"] not in borradas]
        if q and q not in ("todo", "todas", "todos"):
            def puntos(d):
                return _casa(q, d["titulo"], d["resumen"], " ".join(d["etiquetas"]),
                             " ".join(d["pendientes"]), d["carpeta"])
            sel = sorted([d for d in docs if puntos(d)], key=lambda d: -puntos(d))
        else:
            sel = list(docs)
        return {"consulta": a.get("consulta") or "todo", "cuantos": len(sel),
                "documentos": sel,
                # Los titulos SIEMPRE, tambien cuando no hay resultados: es lo
                # que le deja al modelo corregir el tiro en la misma vuelta en
                # vez de contestar "no encuentro nada" y dejarlo ahi.
                "todos_los_titulos": [d["titulo"] for d in docs],
                "borradas_en_esta_conversacion": sorted(borradas)}

    def _ver_recordatorios(self, a) -> dict:
        rs = list(self.sim.datos["recordatorios"]["lista"])
        rs += [dict(r, creado_aqui=True)
               for r in self.sim.datos["escrituras"]["recordatorios_creados"]]
        f = _sin_tildes(a.get("filtro") or "pendientes")
        if f in ("vencidos", "atrasados"):
            sel = [r for r in rs if r.get("vencido") and not r["hecho"]]
        elif f == "hoy":
            sel = [r for r in rs if r.get("dia") == 0 and not r["hecho"]]
        elif f in ("todos", "todo"):
            sel = rs
        else:
            sel = [r for r in rs if not r["hecho"]]
        return {"filtro": f, "hoy_es": _fecha(0), "cuantos": len(sel),
                "vencidos": sum(1 for r in sel if r.get("vencido")),
                "recordatorios": [{"texto": r["texto"], "cuando": r["cuando"],
                                   "vencido": bool(r.get("vencido")),
                                   "hecho": r["hecho"]} for r in sel]}

    def _buscar_en_web(self, a) -> dict:
        web = self.sim.datos["web"]
        q = a.get("consulta") or ""
        t = _sin_tildes(q)
        for clave, datos in web["consultas"].items():
            if clave in t or t in clave:
                return dict(datos, consulta=q, fuente="simulada",
                            cuantos=len(datos["resultados"]))
        g = web["generica"]
        return {"consulta": q, "fuente": "simulada", "resumen": g["resumen"],
                "cuantos": len(g["resultados"]), "resultados": g["resultados"],
                "aviso": "búsqueda simulada: los resultados no son reales, "
                         "dilo si el dato importa"}

    # -- las herramientas de ESCRITURA (solo se llega aqui confirmado) ------
    def _crear_evento(self, a) -> dict:
        d = _cuando_a_dias(a.get("cuando"))
        d = 0 if d is None else d
        ev = {"dia": d, "hora": a.get("hora", "09:00"),
              "titulo": a.get("titulo", "sin título"),
              "donde": a.get("donde", ""), "con": a.get("con") or [],
              "duracion_min": int(a.get("duracion_min") or 60),
              "estado": "confirmado"}
        self.sim.datos["escrituras"]["eventos_creados"].append(ev)
        self.sim.guardar()
        return {"creado": dict(ev, fecha=_fecha(d))}

    def contacto(self, quien) -> dict | None:
        """Nombre -> ficha de la agenda. Es lo que evita que el asistente pida
        una direccion de correo en voz alta, que es lo peor que puede pasarle a
        una conversacion hablada: nadie deletrea un arroba a gusto."""
        q = _sin_tildes(quien or "")
        if not q:
            return None
        for c in self.sim.datos.get("contactos", {}).get("gente", []):
            if (q == _sin_tildes(c["nombre"]) or q in [_sin_tildes(x) for x in c["alias"]]
                    or q == _sin_tildes(c["correo"])):
                return c
        for c in self.sim.datos.get("contactos", {}).get("gente", []):
            if _casa(q, c["nombre"], " ".join(c["alias"])):
                return c
        return None

    def _enviar_correo(self, a) -> dict:
        c = self.contacto(a.get("para"))
        m = {"para": (f"{c['nombre']} <{c['correo']}>" if c else a.get("para")),
             "asunto": a.get("asunto"),
             "cuerpo": a.get("cuerpo"), "cuando": datetime.now().strftime("%H:%M")}
        self.sim.datos["escrituras"]["correos_enviados"].append(m)
        self.sim.guardar()
        return {"enviado": m,
                "desconocido": None if c else a.get("para"),
                "aviso": "simulado: NO ha salido ningún correo de esta máquina"}

    def _crear_recordatorio(self, a) -> dict:
        r = {"texto": a.get("texto"), "cuando": a.get("cuando") or "sin fecha",
             "dia": None, "hecho": False, "vencido": False}
        self.sim.datos["escrituras"]["recordatorios_creados"].append(r)
        self.sim.guardar()
        return {"creado": r}

    def _borrar_nota(self, a) -> dict:
        titulo = a.get("titulo") or ""
        docs = self.sim.datos["notas"]["documentos"]
        hit = next((d for d in docs
                    if _sin_tildes(d["titulo"]) == _sin_tildes(titulo)), None)
        if hit is None:
            hit = next((d for d in docs
                        if _sin_tildes(titulo) in _sin_tildes(d["titulo"])), None)
        if hit is None:
            return {"error": f"no hay ninguna nota que se llame {titulo!r}",
                    "titulos": [d["titulo"] for d in docs]}
        self.sim.datos["escrituras"]["notas_borradas"].append(
            {"titulo": hit["titulo"], "cuando": datetime.now().strftime("%H:%M")})
        self.sim.guardar()
        return {"borrada": hit["titulo"],
                "aviso": "simulado: la nota sigue en el fichero, solo se ha "
                         "apuntado el borrado"}


# --------------------------------------------------- lo que se le dice al LLM --
def instrucciones(ejecutor=None, sesion="web") -> str:
    """El anexo de sistema que acompaña a las herramientas.

    Va aparte de la instruccion del perfil A PROPOSITO: el perfil dice QUIEN
    es el asistente y esto dice COMO se usan las herramientas. Cambiar de
    perfil no debe poder cargarse la regla de confirmacion.
    """
    # EL ORDEN IMPORTA. La regla de escritura va la PRIMERA porque es la que
    # el modelo se salta cuando se le pone al final: medido, con la regla al
    # final y una confirmacion previa en el historial el modelo contestaba
    # "apuntado" sin haber llamado a nada.
    t = [
        f"Hoy es {_fecha(0)}.",
        # SIN VOCABULARIO QUE NARRAR. Esta regla decia antes «de preguntarle
        # al usuario se encarga el sistema», y el modelo se puso a decirlo en
        # voz alta: «el sistema está a punto de crear un recordatorio, ¿lo
        # dejo?», sin llamar a nada. Explicarle el mecanismo le da algo que
        # contar en vez de algo que hacer. Ahora solo se le dice QUE HACER.
        "Si te piden apuntar, crear, mandar o borrar algo, LLAMA A LA "
        "HERRAMIENTA. Es la única forma de que ocurra: sin la llamada no pasa "
        "nada, lo digas como lo digas. Llámala en cuanto te lo pidan y sin "
        "preguntar antes, con los datos que te hayan dado; lo que falte, "
        "ponlo tú con sentido común. Da igual lo que se haya hablado antes.",
        "Nunca digas que algo está apuntado, mandado, creado o borrado si no "
        "lo has visto en el resultado de una herramienta.",
        "Tienes además herramientas para mirar el calendario, el correo, el "
        "estado del servidor de casa, las notas de proyectos, los "
        "recordatorios y para buscar en internet.",
        "Úsalas en cuanto la pregunta dependa de un dato real. NUNCA contestes "
        "de memoria sobre la agenda, el correo, el servidor, las notas o los "
        "recordatorios: eso cambia y tú no lo tienes. Aunque ya lo hayas "
        "mirado antes en esta misma conversación, vuelve a mirarlo; inventarte "
        "una hora, un remitente o un servicio caído es el peor fallo posible.",
        "ESTO SE DICE EN VOZ ALTA: nunca leas identificadores, direcciones de "
        "correo completas, JSON ni nombres de campo. Cuenta lo que ponga con "
        "tus palabras, en una o dos frases, y da primero lo que importa.",
        "Si algo son muchos elementos, di cuántos hay y cuenta los dos o tres "
        "que importen; no recites la lista entera.",
        "Antes de llamar a una herramienta puedes decir en una frase corta qué "
        "vas a mirar: eso se oye mientras se consulta y evita el silencio.",
        "Si una herramienta devuelve un error, dilo en una frase y no lo "
        "vuelvas a intentar más de una vez.",
        # Cada perfil lleva SU juego, asi que hay preguntas que este asistente
        # no puede contestar. Sin esta linea, el modelo va probando las que
        # tiene a ver si cuela: medido en el perfil 'agenda', pidiendole el
        # estado del servidor, llamaba a recordatorios y a calendario antes de
        # rendirse, y lo iba diciendo en voz alta.
        "Si no tienes ninguna herramienta para lo que te piden, dilo en una "
        "frase y ya está: no pruebes con otras a ver si cuela.",
    ]
    return " ".join(t)


# El modelo puede colar un bloque de codigo o un objeto JSON en el texto pese a
# todo. No se puede "limpiar" a lo bruto sin estropear frases legitimas, asi
# que se quitan solo las dos formas que NUNCA son habla: vallas de codigo y una
# linea entera que sea un objeto o una lista.
_VALLA = re.compile(r"```.*?```", re.S)
_JSON_SUELTO = re.compile(r"(?m)^\s*[\[{].*[\]}]\s*$")


def sin_json(texto: str) -> str:
    return _JSON_SUELTO.sub("", _VALLA.sub("", texto))


# ------------------------------------------------------- el ciclo con el LLM --
def _mensajes(pregunta, historial, hablante, memoria=None):
    """Historial -> mensajes de la API, con los papeles alternados.

    LO QUE DICE EL PROGRAMA NO ENTRA EN EL HUECO DEL MODELO. Ni tal cual, ni
    reescrito. Es lo que evita el peor fallo que tuvo esto, y costo dos
    intentos averiguarlo:

      - Con la pregunta de confirmacion y el "Borrada." metidos como respuestas
        suyas, el modelo aprende el estilo y a la siguiente contesta
        "Apuntado." sin llamar a ninguna herramienta. Medido: 10/12 aciertos
        sin ese historial, 4/12 con el, y dos veces repitiendo mi frase
        palabra por palabra.
      - Reescribiendolas en tercera persona ("(el sistema aplicó...)") sale
        IGUAL DE MAL, 4/12, y encima peor: el modelo se pone a contestar
        entre parentesis, imitando la nota. Lo que copia no es el contenido,
        es el molde.

Asi que los apuntes marcados con `nota` se SALTAN. El hecho no se pierde: va
al bloque de sistema, que es texto que el modelo lee como instruccion y no
como ejemplo de lo que tiene que escribir (ver notas_recientes).

Y POR LO MISMO, LAS HERRAMIENTAS DE LOS TURNOS ANTERIORES SE VUELVEN A CONTAR.
Este es el fallo que mas caro salio de todos y el mas facil de no ver, porque
las dos primeras preguntas funcionan de maravilla:

    turno 1  «¿qué tengo mañana?»           -> consultar_calendario   bien
    turno 2  «¿algo urgente en el correo?»  -> leer_correo            bien
    turno 3  «¿cómo va el servidor?»        -> nada, y se lo INVENTA
    turno 4  «¿por dónde iba el NAS?»       -> nada, y se lo inventa

Y no es la longitud del contexto: con un historial de dos chistes, igual de
largo, el turno 3 llama a la herramienta perfectamente (probado, 2 de 2 con
chistes y 0 de 2 con datos). Lo que pasa es que un historial donde el
asistente suelta horas, remitentes y estados de servicios SIN QUE SE VEA UNA
SOLA LLAMADA le enseña que en esta conversacion se contesta de memoria. Y
obedece.

La causa es que el historial que lleva la pagina es lo que se DIJO, y las
llamadas no se dicen. Asi que al reconstruir los mensajes se vuelven a meter
en su sitio: assistant(tool_use) -> user(tool_result) -> assistant(texto). El
resultado va RESUMIDO y no entero -- "leer_correo(importantes) → 2" -- porque
lo que hay que restituir es la forma, no los datos, y los datos ya estan
contados en la respuesta que viene detras.
    """
    from conversacion import RECUERDO_RESPUESTA, _con_hablante
    porpregunta = {r["pregunta"]: r["llamadas"] for r in (memoria or [])}
    ms = []
    recorte = (historial or [])[-RECUERDO_RESPUESTA:]
    # Los turnos de confirmacion se van EN PAREJA: la pregunta del usuario y
    # la frase del programa. Quitar solo la segunda dejaba un «mándale un
    # correo a Elena» sin respuesta ninguna, y el modelo -- con razon -- creia
    # que le quedaba pendiente y lo volvia a preparar dos turnos despues, con
    # el usuario ya hablando de otra cosa. Lo que paso con esa accion se
    # cuenta por el bloque de sistema, no por el historial.
    fuera = set()
    for i, h in enumerate(recorte):
        if h.get("rol") == "asistente" and h.get("nota"):
            fuera.add(i)
            if i and recorte[i - 1].get("rol") != "asistente":
                fuera.add(i - 1)
    for i, h in enumerate(recorte):
        if i in fuera:
            continue
        papel = "assistant" if h.get("rol") == "asistente" else "user"
        cont = h.get("texto", "") if papel == "assistant" else _con_hablante(h)
        if papel == "assistant":
            previa = recorte[i - 1].get("texto") if i else None
            llamadas = porpregunta.get(previa)
            if llamadas:
                ms.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": c["id"], "name": c["nombre"],
                     "input": c["args"]} for c in llamadas]})
                ms.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": c["id"],
                     "content": c["resumen"]} for c in llamadas]})
                ms.append({"role": "assistant", "content": cont})
                continue
        if ms and ms[-1]["role"] == papel and isinstance(ms[-1]["content"], str):
            ms[-1]["content"] += "\n" + cont
        else:
            ms.append({"role": papel, "content": cont})
    ahora = f"[{hablante}]: {pregunta}" if hablante else pregunta
    if ms and ms[-1]["role"] == "user" and isinstance(ms[-1]["content"], str):
        ms[-1]["content"] += "\n" + ahora
    else:
        ms.append({"role": "user", "content": ahora})
    return ms


def _una_vuelta_minimax(mensajes, sistema, modelo, esquemas, maximo=1024):
    """Una llamada en flujo. Va soltando ('texto', str) segun llega y termina
    devolviendo ('fin', bloques, motivo).

    Los bloques `tool_use` se acumulan enteros y NO se emiten como texto: ese
    es el aislamiento que impide que el JSON acabe en el altavoz.
    """
    cuerpo = {"model": modelo, "max_tokens": maximo, "stream": True,
              "messages": mensajes}
    if sistema:
        cuerpo["system"] = sistema
    if esquemas:
        cuerpo["tools"] = esquemas
    pet = urllib.request.Request(
        MINIMAX_API, method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 "anthropic-version": "2023-06-01", "x-api-key": clave_minimax()})
    r = urllib.request.urlopen(pet, timeout=300)
    bloques, actual, motivo = [], None, None
    for linea in r:
        linea = linea.strip()
        if not linea.startswith(b"data:"):
            continue
        try:
            d = json.loads(linea[5:])
        except ValueError:
            continue
        tipo = d.get("type")
        if tipo == "content_block_start":
            b = d.get("content_block") or {}
            actual = {"tipo": b.get("type"), "texto": "", "json": "",
                      "id": b.get("id"), "nombre": b.get("name")}
        elif tipo == "content_block_delta" and actual is not None:
            delta = d.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                actual["texto"] += delta["text"]
                yield ("texto", delta["text"])
            elif delta.get("type") == "input_json_delta":
                actual["json"] += delta.get("partial_json") or ""
        elif tipo == "content_block_stop" and actual is not None:
            bloques.append(actual)
            actual = None
        elif tipo == "message_delta":
            motivo = (d.get("delta") or {}).get("stop_reason") or motivo
        elif tipo == "message_stop":
            break
    yield ("fin", bloques, motivo)


def _una_vuelta_ollama(mensajes, sistema, modelo, url, esquemas):
    """Lo mismo contra Ollama. Con `tools` el flujo deja de ser fluido en
    algunos modelos -- mandan el texto de golpe al final -- pero el contrato
    de salida es identico, asi que el resto del ciclo no se entera."""
    ms = ([{"role": "system", "content": sistema}] if sistema else []) + mensajes
    cuerpo = {"model": modelo, "stream": True, "messages": ms}
    if esquemas:
        cuerpo["tools"] = esquemas
    pet = urllib.request.Request(f"{url}/api/chat", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(pet, timeout=300)
    bloques, texto = [], ""
    for linea in r:
        if not linea.strip():
            continue
        d = json.loads(linea)
        m = d.get("message") or {}
        if m.get("content"):
            texto += m["content"]
            yield ("texto", m["content"])
        for i, tc in enumerate(m.get("tool_calls") or []):
            f = tc.get("function") or {}
            bloques.append({"tipo": "tool_use", "texto": "",
                            "json": json.dumps(f.get("arguments") or {}),
                            "id": tc.get("id") or f"call_{len(bloques)}",
                            "nombre": f.get("name")})
        if d.get("done"):
            break
    if texto:
        bloques.insert(0, {"tipo": "text", "texto": texto, "json": "",
                           "id": None, "nombre": None})
    yield ("fin", bloques,
           "tool_use" if any(b["tipo"] == "tool_use" for b in bloques) else "end_turn")


def ciclo(pregunta, historial, modelo, sistema, hablante, esquemas, ejecutor,
          sesion="web", url_ollama="http://localhost:11434",
          vueltas=VUELTAS_MAXIMAS):
    """El ciclo entero: el modelo pide, se ejecuta, se le devuelve, contesta.

    Es un GENERADOR y produce dos cosas distintas:
      - `str`  : texto para decir en voz alta. Sale de `text_delta` y de nada
                 mas, asi que no puede llevar JSON dentro.
      - `dict` : un suceso para la pagina ({"tipo": "herramienta", ...}).
    El puente distingue por el tipo y no hay que tocar su troceador.

    Sin `esquemas` se comporta exactamente como preguntar_con_historial(): una
    vuelta, texto y ya. Asi un perfil sin herramientas no paga nada.
    """
    es_minimax = modelo.lower().startswith("minimax")
    mensajes = _mensajes(pregunta, historial, hablante,
                         ejecutor.recordado(sesion) if ejecutor else None)
    del_turno = []      # lo llamado AHORA, para que el turno siguiente lo vea
    for vuelta in range(vueltas):
        bloques, motivo = [], None
        fuente = (_una_vuelta_minimax(mensajes, sistema, modelo, esquemas)
                  if es_minimax else
                  _una_vuelta_ollama(mensajes, sistema, modelo, url_ollama,
                                     esquemas))
        hubo_texto = False
        for item in fuente:
            if item[0] == "texto":
                hubo_texto = True
                yield item[1]
            else:
                _, bloques, motivo = item
        llamadas = [b for b in bloques if b["tipo"] == "tool_use"]
        if not llamadas:
            if ejecutor is not None and del_turno:
                ejecutor.recordar(sesion, pregunta, del_turno)
            return
        # El separador entre vueltas. Sin el, "Voy a mirarlo." y "Mañana
        # tienes..." se pegan sin espacio y el troceador de frases del
        # narrador se come el corte: sale una frase larguisima y mal
        # entonada. Un espacio cuesta cero y lo arregla.
        if hubo_texto:
            yield " "
        if vuelta == vueltas - 1:
            # Techo alcanzado: se le dice al modelo que conteste con lo que
            # tenga en vez de dejar al usuario esperando otra vuelta.
            if ejecutor is not None and del_turno:
                ejecutor.recordar(sesion, pregunta, del_turno)
            yield {"tipo": "herramienta", "fase": "tope",
                   "texto": f"{vueltas} vueltas de herramienta: se corta"}
            return
        # El apunte del asistente tiene que ir con los MISMOS bloques que
        # mando: si se pierde el tool_use, el tool_result de despues no casa
        # con nada y la API lo rechaza.
        contenido = []
        for b in bloques:
            if b["tipo"] == "text" and b["texto"]:
                contenido.append({"type": "text", "text": b["texto"]})
            elif b["tipo"] == "tool_use":
                try:
                    entrada = json.loads(b["json"] or "{}")
                except ValueError:
                    entrada = {}
                b["args"] = entrada
                contenido.append({"type": "tool_use", "id": b["id"],
                                  "name": b["nombre"], "input": entrada})
        mensajes.append({"role": "assistant", "content": contenido})
        resultados, preparada = [], None
        for b in llamadas:
            args = b.get("args", {})
            h = POR_NOMBRE.get(b["nombre"]) or {}
            yield {"tipo": "herramienta", "fase": "llamando",
                   "id": b["id"], "nombre": b["nombre"], "args": args,
                   "dominio": h.get("dominio"), "escribe": bool(h.get("escribe"))}
            try:
                res, resumen = ejecutor.ejecutar(b["nombre"], args, sesion)
                fase = ("confirmar"
                        if res.get("estado") == "preparada_sin_ejecutar"
                        else "hecho")
                if fase == "confirmar" and preparada is None:
                    preparada = res["resumen"]
                fallo = None
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}
                resumen = f"{b['nombre']}: {type(e).__name__}"
                fase, fallo = "error", f"{type(e).__name__}: {e}"
            yield {"tipo": "herramienta", "fase": fase, "id": b["id"],
                   "nombre": b["nombre"], "resumen": resumen,
                   "dominio": h.get("dominio"), "escribe": bool(h.get("escribe")),
                   "error": fallo}
            resultados.append({"type": "tool_result", "tool_use_id": b["id"],
                               "content": json.dumps(res, ensure_ascii=False),
                               "is_error": fallo is not None})
            # SOLO LAS DE LECTURA ENTRAN EN LA MEMORIA. Una escritura que se
            # quedo preparada y sin confirmar no puede volver al historial
            # como si hubiera pasado: medido, el modelo la ve en el turno
            # siguiente, cree que le queda pendiente y la vuelve a llamar --
            # se acumulaban tres a la vez, y aunque ninguna llegaba a
            # ejecutarse, el asistente hablaba de un correo que nadie habia
            # vuelto a mencionar. Lo que paso con las escrituras se cuenta por
            # el bloque de sistema (nota_de_remate), que no se imita.
            if not h.get("escribe"):
                del_turno.append({"id": b["id"], "nombre": b["nombre"],
                                  "args": args, "resumen": str(resumen)[:120]})
        # AQUI SE ACABA EL TURNO SI HAY ALGO QUE CONFIRMAR, y el modelo no
        # vuelve a hablar. La pregunta la pone el programa, palabra por
        # palabra, a partir del mismo resumen que se ejecutara si el usuario
        # dice que si. Dejarsela al modelo tenia dos fallos MEDIDOS contra
        # MiniMax-M3: parafraseaba el resumen (se confirmaba una cosa y se iba
        # a hacer otra) y, con un "si, borrala" ya en el historial, remataba
        # con un "hecho, borrada" habiendo la herramienta devuelto
        # "preparada_sin_ejecutar" -- o sea, mentia. Ademas se ahorra una
        # vuelta de LLM entera.
        if preparada:
            if ejecutor is not None and del_turno:
                ejecutor.recordar(sesion, pregunta, del_turno)
            yield pregunta_de_confirmacion(preparada)
            return
        if es_minimax:
            mensajes.append({"role": "user", "content": resultados})
        else:
            for r, b in zip(resultados, llamadas):
                mensajes.append({"role": "tool", "content": r["content"],
                                 "name": b["nombre"]})


# ------------------------------------------------------------------- CLI ---
def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orden", choices=["listar", "llamar", "ciclo", "datos"])
    ap.add_argument("resto", nargs="*")
    ap.add_argument("--datos", default=None)
    ap.add_argument("--modelo", default=os.environ.get("ASISTENTE_MODELO", "MiniMax-M3"))
    ap.add_argument("--sesion", default="cli")
    a = ap.parse_args(argv)
    ej = Ejecutor(a.datos)
    if a.orden == "listar":
        for h in CATALOGO:
            print(f"{'ESCRIBE' if h['escribe'] else 'lee    '}  "
                  f"{h['nombre']:22} [{h['dominio']}]")
            print(f"           {h['descripcion'].splitlines()[0][:100]}")
            print(f"           de verdad: {h['sustituir_por']}")
        return 0
    if a.orden == "datos":
        print(json.dumps(ej.sim.datos, ensure_ascii=False, indent=1)[:4000])
        return 0
    if a.orden == "llamar":
        if not a.resto:
            print("uso: llamar NOMBRE clave=valor ...", file=sys.stderr)
            return 2
        args = {}
        for kv in a.resto[1:]:
            k, _, v = kv.partition("=")
            args[k] = {"true": True, "false": False}.get(v, v)
        res, resumen = ej.ejecutar(a.resto[0], args, a.sesion)
        print(f"# {resumen}")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    # ciclo
    pregunta = " ".join(a.resto) or "¿qué tengo hoy?"
    t0 = time.time()
    dicho, primero = "", None
    for x in ciclo(pregunta, [], a.modelo,
                   "Eres un asistente de voz en castellano. Frases cortas. " +
                   instrucciones(ej, a.sesion),
                   None, esquemas_anthropic(), ej, a.sesion):
        if isinstance(x, dict):
            print(f"  [{x['fase']:10}] {x.get('nombre', '')} "
                  f"{x.get('resumen') or x.get('args') or ''}", file=sys.stderr)
        else:
            if primero is None:
                primero = time.time() - t0
            dicho += x
    print(f"\n{dicho.strip()}\n")
    print(f"--- primer texto {primero or 0:.2f}s · total {time.time() - t0:.2f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
