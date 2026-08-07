#!/usr/bin/env python
"""Qué querías cuando le cortaste: parar, esperar, seguir, o preguntar algo.

PARAR Y ENTENDER SON DOS COSAS DISTINTAS
El asistente ya calla por huella de voz en medio segundo (oido.barrera), sin
saber todavía qué le has dicho. Este módulo es el paso siguiente: con la
transcripción ya en la mano, decidir qué hacer con el silencio que acaba de
abrirse. Y la mayoría de las veces la respuesta no necesita un LLM.

    «espera»          -> ESPERAR: calla y pregunta «¿qué pasa?»
    «para, déjalo»    -> PARAR:   calla y se queda callado
    «vale, sigue»     -> SEGUIR:  retoma por donde iba
    «¿qué has dicho?» -> REPETIR: repite lo último que dijo
    cualquier otra    -> None:    camino normal, al LLM, con el contexto
                                  de lo que estaba diciendo

POR QUE UNA TABLA DE FRASES Y NO UN MODELO
Porque el modelo ya está en el camino y cuesta lo que cuesta: la compuerta de
destinatario son 0,47-0,70 s medidos, y el LLM que redacta la respuesta,
1,15-6,5 s hasta la primera frase. Un «espera» que tarda dos segundos en
recibir un «¿qué pasa?» no se siente como una interrupción, se siente como
una avería. Estas cuatro intenciones son un puñado de frases hechas, no
lenguaje abierto: casan con expresiones regulares en microsegundos y con eso
el asistente contesta en cuanto la voz sale del sintetizador.

Lo abierto -- «detalla eso último», «vuelve a lo del contenedor» -- NO se
intenta adivinar aquí: cae en None y va al LLM, que es quien sabe.

DOS REGLAS QUE EVITAN LOS FALSOS POSITIVOS, Y POR QUE ESTAN
  - SE EXIGE QUE CASE LA FRASE ENTERA, no un trozo. «espera» es esperar;
    «espera, ¿cuánto consume el servidor?» es una PREGUNTA que empieza por
    espera, y tratarla como un «¿qué pasa?» sería perder lo que de verdad
    querías. Buscar la palabra suelta dentro del texto se comería ese caso.
  - SEGUIR SE MIRA ANTES QUE PARAR. «vale, sigue» contiene el «vale» de
    «vale ya», y en ese orden el asistente se callaría justo cuando le
    acabas de decir que continúe.

TOLERANCIA A WHISPER: el texto se normaliza sin tildes, sin signos y sin
mayúsculas antes de comparar, porque con frases de una palabra whisper
titubea con la puntuación y los abre-interrogación («¡Como!», «como.»,
«¿Cómo?» son la misma orden). Lo que no se hace es corregir palabras: si
whisper oye «espira» por «espera», esto no casa y la frase va al LLM, que es
el repliegue correcto -- responde de más, nunca calla de menos.
"""
import re
import unicodedata

PARAR = "parar"
ESPERAR = "esperar"
SEGUIR = "seguir"
REPETIR = "repetir"


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin signos: lo que sobrevive a whisper."""
    t = unicodedata.normalize("NFD", (texto or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9ñ ]+", " ", t).split())


# Muletillas que pueden envolver cualquiera de estas órdenes sin cambiarlas:
# «eh, para», «oye, sigue», «vale, para ya, gracias».
_RELLENO = r"(?:\b(?:eh|ey|oye|oiga|vale|bueno|pues|va|venga|porfa|por favor|"
_RELLENO += r"perdona|perdon|perdone|disculpa|gracias|ya|a ver|hey|ok|okey)\b\s*)*"


def _patron(alternativas):
    """La frase ENTERA, con muletillas delante y detrás (ver cabecera)."""
    return re.compile(rf"^{_RELLENO}(?:{alternativas})\s*{_RELLENO}$")


# El orden importa: seguir antes que parar (ver cabecera).
_INTENCIONES = [
    (SEGUIR, _patron(
        r"sigue(?:\s+sigue)*|continua|continue|prosigue|adelante|dale|"
        r"puedes seguir|sigue con eso|sigue por donde ibas|sigue contando|"
        r"tira|continua porfa|no nada sigue|nada sigue")),
    (REPETIR, _patron(
        r"repite(?:melo)?|repite eso|que has dicho|que decias|que dijiste|"
        r"no te (?:he )?(?:oido|escuchado|entendido)|no te oi|otra vez|"
        r"como dices|que era")),
    (PARAR, _patron(
        r"para(?:\s+para)*|parate|para ya|deja(?:lo)?|dejalo ya|dejalo estar|"
        # «vale ya» y «ya vale» van enteras y no por partes: las dos palabras
        # están en las muletillas, así que sueltas se las come el relleno y
        # al núcleo no le queda nada que casar.
        r"calla(?:te)?|callate ya|silencio|basta|basta ya|ya esta|ya vale|"
        r"vale ya|"
        r"olvidalo|olvidate|olvida eso|da igual|no importa|nada|nada nada|"
        r"stop|corta|cancela|cancelalo|para el rollo|no sigas|no me lo "
        r"cuentes|no hace falta")),
    (ESPERAR, _patron(
        r"espera(?:\s+espera)*|esperate|espera un momento|espera un segundo|"
        r"un momento|un momentito|un segundo|un segundito|momento|"
        # «para un momento» es «detente un rato», no la orden de parar del
        # todo: se pide atencion, no silencio. Salio de una prueba real --
        # whisper transcribio «Para, para un momento.» asi y caia al LLM.
        r"para(?:\s+para)*\s+un (?:momento|momentito|segundo|rato)|"
        r"parate un momento|"
        r"como|comorl|que|que|eh|perdon|perdona|oye|escucha|"
        r"a ver a ver|que que|espera espera espera")),
]


def clasificar(texto: str):
    """Texto oído -> PARAR / ESPERAR / SEGUIR / REPETIR, o None si es otra cosa.

    None NO es un fallo: es «esto no es una frase hecha, que lo mire el LLM».
    """
    t = normalizar(texto)
    if not t:
        return None
    # Una orden de estas es corta por definición. El tope evita que una
    # parrafada que casualmente termine en «sigue» se coma el camino bueno.
    if len(t.split()) > 6:
        return None
    for intencion, patron in _INTENCIONES:
        if patron.match(t):
            return intencion
    return None


# ---- lo que dice el asistente sin pasar por el LLM ----------------------
# Fijas y cortas A PROPOSITO: son la prueba de que te ha oído, no una
# respuesta. Cuanto antes suenen, mejor; el LLM ya vendrá después si hace
# falta. PARAR no lleva frase: callarse ES la respuesta.
FRASES = {
    ESPERAR: "¿Qué pasa?",
    REPETIR: None,      # repetir necesita el texto de antes: lo pone quien llama
    SEGUIR: None,       # seguir retoma la locución: tampoco es una frase fija
    PARAR: None,
}


def que_decir(intencion, ultimo_dicho=None):
    """(texto_a_decir, retoma). `retoma` pide continuar la locución cortada."""
    if intencion == ESPERAR:
        return FRASES[ESPERAR], False
    if intencion == REPETIR:
        return (ultimo_dicho or "").strip() or None, False
    if intencion == SEGUIR:
        return None, True
    return None, False


# ---- por donde iba cuando le cortaste -----------------------------------

def apunte_cortado(dicho: str, restante: str) -> dict:
    """El apunte de historial de una respuesta que se quedó a medias.

    Se guarda partido en dos -- lo que la otra persona LLEGÓ A OÍR y lo que
    quedó dentro -- porque el LLM necesita las dos cosas y significan cosas
    distintas: «detalla eso último» habla de lo dicho, y «sigue» habla de lo
    que faltaba.
    """
    apunte = {"rol": "asistente", "texto": (dicho or "").strip()}
    restante = (restante or "").strip()
    apunte["cortado"] = True
    if restante:
        apunte["restante"] = restante
    return apunte


# Tope del trozo pendiente que se le enseña al LLM. Con una respuesta larga,
# lo que quedaba por decir puede ser un folio entero y no aporta: lo que se
# quiere es que sepa POR DONDE iba, no repetirle su propio borrador.
TOPE_RESTANTE = 400


def texto_para_la_compuerta(apunte: dict) -> str:
    """Apunte del asistente -> la línea que ve la COMPUERTA.

    Aquí la marca del corte sí va dentro del texto, entre paréntesis: la
    compuerta solo devuelve un booleano por salida estructurada, así que no
    puede repetir la nota, y saber que la respuesta anterior se quedó a medias
    es justo lo que le hace entender un «sigue» suelto detrás.
    """
    texto = (apunte.get("texto") or "").strip()
    if not apunte.get("cortado"):
        return texto
    return f"{texto} (le interrumpieron aquí)".strip()


def nota_de_corte(historial) -> str:
    """La frase que se le añade a la INSTRUCCIÓN DE SISTEMA, o "".

    ESTO NO PUEDE IR DENTRO DEL TURNO DEL ASISTENTE, y costó verlo: puesta
    ahí -- «...dijo esto (te interrumpieron aquí; te quedaba por decir «X»)»
    -- el modelo la toma por algo que él mismo escribió y la CONTINÚA. En una
    prueba real, qwen3:4b arrancó su respuesta con «(te quedaba por decir:
    «…asignada será la del…» y eso salió por el altavoz tal cual.

    En el sistema no pasa: ahí es una instrucción sobre la conversación, no un
    trozo de la conversación, y además es donde el modelo espera encontrar las
    reglas de qué no repetir.
    """
    ultimo = None
    for h in reversed(historial or []):
        if h.get("rol") == "asistente":
            ultimo = h
            break
    if not ultimo or not ultimo.get("cortado"):
        return ""
    dicho = (ultimo.get("texto") or "").strip()
    restante = (ultimo.get("restante") or "").strip()
    if len(restante) > TOPE_RESTANTE:
        restante = restante[:TOPE_RESTANTE].rsplit(" ", 1)[0] + "…"
    nota = " Tu última intervención se quedó a medias porque te interrumpieron"
    if dicho:
        nota += f'; lo que la otra persona llegó a oír fue: «{dicho}»'
    if restante:
        nota += f'. Lo que te quedaba por decir era: «{restante}»'
    nota += (". Úsalo para entender a qué se refieren, no repitas lo que ya "
             "dijiste y no menciones esta nota.")
    return nota


# ---- la batería con la que se afina esto --------------------------------
# Verdad de terreno para scripts/escucha_fidelidad.py. Las frases con None
# son las que DEBEN caer al LLM: son el caso peligroso, porque tratarlas como
# una orden hecha se comería lo que de verdad se preguntaba.
BATERIA = [
    ("espera", ESPERAR), ("espera espera", ESPERAR), ("¡Espera!", ESPERAR),
    ("un momento", ESPERAR), ("un segundo", ESPERAR), ("¿Cómo?", ESPERAR),
    ("Para, para un momento.", ESPERAR), ("para un momento", ESPERAR),
    ("como", ESPERAR), ("¿Qué?", ESPERAR), ("oye", ESPERAR),
    ("perdona", ESPERAR), ("eh, espera", ESPERAR),
    ("para", PARAR), ("para ya", PARAR), ("¡Cállate!", PARAR),
    ("callate un momento", None),          # lleva complemento: al LLM
    ("déjalo", PARAR), ("vale ya", PARAR), ("ya está", PARAR),
    ("olvídalo", PARAR), ("da igual", PARAR), ("nada", PARAR),
    ("basta", PARAR), ("no sigas", PARAR), ("vale, para ya, gracias", PARAR),
    ("sigue", SEGUIR), ("vale, sigue", SEGUIR), ("continúa", SEGUIR),
    ("perdona, sigue", SEGUIR), ("nada, sigue", SEGUIR), ("adelante", SEGUIR),
    ("puedes seguir", SEGUIR),
    ("repite", REPETIR), ("¿qué has dicho?", REPETIR),
    ("no te he oído", REPETIR), ("otra vez", REPETIR),
    # las que NO son órdenes hechas y tienen que ir al LLM
    ("espera, ¿cuánto consume el servidor?", None),
    ("detalla eso último", None),
    ("vuelve atrás, a lo del contenedor", None),
    ("¿y en euros cuánto sale?", None),
    ("para el contenedor de whisper", None),
    ("sigue con el resumen pero solo los titulares", None),
    ("¿qué has dicho del despliegue de anoche?", None),
    ("no, me refería a esta semana", None),
    ("olvídate del resumen y dime la hora", None),
]
