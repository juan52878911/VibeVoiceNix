#!/usr/bin/env python
"""PERFILES DE ASISTENTE: identidad, voz, vocabulario, rellenos y herramientas.

    pkgs/vibevoice/.venv/bin/python scripts/perfiles.py generar
    pkgs/vibevoice/.venv/bin/python scripts/perfiles.py listar
    pkgs/vibevoice/.venv/bin/python scripts/perfiles.py verificar
    pkgs/vibevoice/.venv/bin/python scripts/perfiles.py banco-bufer

QUE PROBLEMA RESUELVE
El asistente calla entre que dejas de hablar y que empieza a contestar, y ese
hueco es largo y ademas VARIABLE. Medido en este proyecto, de punta a punta:

    whisper nativo             0,30 s
    compuerta (¿era para mi?)  0,70 s   -- en paralelo con el LLM
    MiniMax, primer token      1,15-6,5 s
    primer token -> primer sonido  0,53 s

O sea: entre 1,7 y 7,0 s de silencio. Un humano no calla siete segundos: dice
"mmm, a ver" y sigue pensando en voz alta. Eso es lo que hay aqui -- audios
pregenerados con LA MISMA VOZ que suenan mientras el LLM redacta -- y ademas
son la unica forma honesta de conseguir un bufer de reproduccion de verdad:
mientras suena el relleno, el audio real se acumula sin que nadie espere.

Y NO ES UN CATALOGO, SON PERFILES. La idea de "un puñado de WAV sueltos" se
queda corta en cuanto hay mas de un asistente: uno para las notas, otro para
el servidor, otro para el correo. Cada uno quiere su voz, su registro y su
vocabulario, y por tanto SUS rellenos ("mirando las notas" no vale para el que
vigila el servidor). Asi que la unidad es el PERFIL y el catalogo es una parte
suya.

POR QUE JSON Y NO TOML
  - La biblioteca estandar de Python lee JSON *y lo escribe*; de TOML solo lee
    (tomllib, 3.11+) y no hay escritor. Aqui hace falta escribir: el manifiesto
    del catalogo generado es un fichero mas.
  - El perfil viaja tal cual al navegador (GET /perfiles del puente) sin
    convertir nada.
  - Nix puede generarlo con builtins.toJSON si algun dia se declara desde el
    modulo, igual que ya se hace con otras cosas de esta VM.
  - El repo ya habla JSON en todas partes (perfiles_voz.json, el protocolo de
    marcos, los manifiestos). Una sintaxis menos que aprender.
El precio es que JSON no tiene comentarios. Se compensa con el campo "_nota",
que el cargador ignora y que se usa para explicar cada perfil por dentro.

LA CACHE VA POR HASH DEL CONTENIDO, Y ESO ES LO IMPORTANTE
Cada relleno se identifica por el sha256 de lo que lo determina: texto, voz,
semilla, cfg_scale, pasos y la version del formato. El fichero se llama con
ese hash. Consecuencia: cambiar UNA frase regenera UNA frase; cambiar la voz
del perfil regenera ese perfil entero y no toca los demas; y arrancar dos
veces seguidas no sintetiza nada la segunda vez. Sin hash habria que
regenerar todo ante cualquier cambio, que con 20 frases a ~1 s de audio cada
una son ~25 s de VM por arranque.

DONDE VIVE LA CACHE
NO en el repo ni en el store de Nix (que es de solo lectura). En este orden:
  1. --cache si se pasa
  2. $VOZ_PERFILES_CACHE
  3. $STATE_DIRECTORY (lo pone systemd con StateDirectory=, que es como
     voz-stream.nix resuelve /var/lib/voz para sus IR de OpenVINO)
  4. $XDG_STATE_HOME/voz-perfiles, o ~/.local/state/voz-perfiles
Es todo regenerable: borrar la cache no pierde nada, solo cuesta un arranque.

SI EL SERVICIO DE VOZ NO ESTA, NO PASA NADA
generar() no lanza: devuelve el catalogo con lo que haya y una lista de
fallos. El asistente arranca igual y se queda sin rellenos -- que es una
mejora ausente, no una averia --, y en el arranque siguiente lo intenta otra
vez. Por eso el puente lo llama en un hilo aparte: ni siquiera retrasa el
arranque.

HERRAMIENTAS: EL HUECO ESTA HECHO, Y ESTA VACIO A PROPOSITO
Cada perfil lleva una lista "herramientas" con su forma definitiva (id, tipo,
descripcion, habilitada, config, esquema). Nada la consume todavia. Se
declara ahora para que cuando lleguen -- notas de Obsidian, estado del
servidor, recordatorios, calendario, correo, busqueda web -- no haya que
rehacer el formato ni migrar los ficheros de nadie. validar() comprueba la
forma; el resto es de quien las implemente.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RITMO = 24_000

# Sube cuando cambie ALGO que afecte al audio generado y que no sea uno de los
# campos que ya entran en el hash (por ejemplo, si un dia se recorta el
# silencio de cabecera al guardar). Cambiarla invalida la cache entera.
VERSION_FORMATO = 1

# Las cinco categorias, y cuando suena cada una. Los umbrales de tiempo NO
# estan aqui: son politica del puente (asistente_web.py), porque dependen de
# lo que tarde el LLM y eso el catalogo no lo sabe.
CATEGORIAS = {
    "afirmacion": "acuse de recibo: 'vale', 'entendido'. Suena en cuanto se "
                  "sabe que la frase iba dirigida al asistente y el LLM aun "
                  "no ha escrito nada.",
    "pensando":   "el LLM esta redactando y tarda: 'mmm, a ver'.",
    "esperando":  "sigue tardando despues del primero: 'un momento mas'.",
    "cerrando":   "remate de la respuesta. Por defecto NO se usa en un final "
                  "normal (añadir palabras que el LLM no dijo molesta a la "
                  "tercera vez); esta para cuando la locucion se corta por "
                  "error y hay que cerrarla con algo.",
    "negacion":   "no se ha podido: averia del LLM o de la voz. Mejor eso "
                  "que un cartel rojo y silencio.",
    "consultando": "una herramienta esta consultando y el modelo NO dijo nada "
                   "antes de llamarla. Cuando si lo dice ('voy a mirar tu "
                   "calendario') no suena: seria pisarle con una coletilla "
                   "peor y mas generica que la suya.",
    "confirmando": "el remate de una accion confirmada por voz: 'Enviado.', "
                   "'Borrada.'. NO es una coletilla intercambiable como las "
                   "demas -- se elige POR TEXTO (Catalogo.buscar), porque "
                   "decir 'apuntado' cuando lo que se hizo fue borrar seria "
                   "mentir. Al estar pregenerado, el remate suena en el acto "
                   "en vez de costar una locucion entera.",
}

# El perfil de serie. Se escribe a disco con `perfiles.py init` y es lo que se
# carga si no hay fichero: el asistente de siempre, con voz sp-Spk1_man.
#
# LAS VARIANTES SON TRES O CUATRO POR CATEGORIA A PROPOSITO. Con una sola, a
# la tercera vez que suena la misma coletilla molesta mas que el silencio que
# vino a tapar; con tres, la repeticion inmediata se evita ademas eligiendo
# distinto del ultimo (Catalogo.elegir).
PERFIL_BASE = {
    "version": VERSION_FORMATO,
    "perfil_por_defecto": "general",
    "perfiles": {
        "general": {
            "nombre": "General",
            "descripcion": "El asistente de casa, sin especialidad.",
            "_nota": "Perfil de arranque. Copia este bloque y cambia voz, "
                     "sistema y rellenos para tener otro.",
            "voz": {
                "voz": "sp-Spk1_man",
                "semilla": 11,
                "cfg_scale": 3.5,
                "pasos": 6,
            },
            "sistema": "Eres un asistente de voz en castellano. Responde "
                       "corto y directo, en frases que se puedan decir en "
                       "voz alta. Nada de listas ni markdown.",
            "vocabulario": {
                "registro": "cercano, de tu a tu, sin formalismos",
                "terminos": [],
            },
            "rellenos": {
                "afirmacion": ["Vale.", "Entendido.", "Sí, claro.",
                               "Ahora mismo."],
                "pensando": ["Mmm, a ver.", "Déjame pensarlo.",
                             "A ver, un segundo.", "Vamos a ver."],
                "esperando": ["Sigo en ello.", "Un momento más.",
                              "Casi lo tengo."],
                "cerrando": ["Y eso es todo.", "Ya está.", "Hasta aquí."],
                "negacion": ["No he podido con eso.",
                             "Lo siento, no me ha salido.",
                             "Ahora mismo no puedo."],
            },
            "herramientas": [],
        },
    },
}

# La forma que tendra una herramienta cuando las haya. Se documenta aqui para
# que quien añada la primera no tenga que inventarse el formato.
HERRAMIENTA_EJEMPLO = {
    "id": "obsidian_notas",
    "tipo": "mcp",              # mcp | http | interno
    "descripcion": "Busca y escribe notas en la boveda de Obsidian.",
    "habilitada": False,        # nada la consume todavia
    "config": {},               # servidor, credencial, rutas... segun tipo
    "esquema": {"parametros": {}},   # JSON Schema de la llamada
}
TIPOS_HERRAMIENTA = ("mcp", "http", "interno")


# --------------------------------------------------------------- la cache --
def directorio_cache(explicito=None) -> Path:
    """Donde se guardan los WAV y el manifiesto. Ver la cabecera."""
    if explicito:
        return Path(explicito).expanduser()
    for var in ("VOZ_PERFILES_CACHE", "STATE_DIRECTORY"):
        v = os.environ.get(var)
        if v:
            # STATE_DIRECTORY puede traer varias rutas separadas por ':'
            return Path(v.split(":")[0]) / ("rellenos" if var == "STATE_DIRECTORY"
                                            else "")
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(base) / "voz-perfiles"


def huella(texto: str, voz: dict) -> str:
    """El identificador de un relleno: todo lo que decide como suena.

    Cambiar el texto, la voz, la semilla, el cfg o los pasos da otra huella y
    por tanto otro fichero; NO cambiarlos da la misma y no se regenera. Es
    toda la logica de la cache, y cabe en cuatro lineas."""
    crudo = json.dumps({
        "v": VERSION_FORMATO,
        "t": texto,
        "voz": voz.get("voz"),
        "semilla": voz.get("semilla"),
        "cfg": voz.get("cfg_scale"),
        "pasos": voz.get("pasos"),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(crudo.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------- el fichero --
def dominios_de(perfil: dict) -> list:
    """Los dominios de herramienta HABILITADOS de un perfil.

    Un perfil no lista herramientas sueltas sino areas ('calendario',
    'correo'...). Lo que se reparte entre asistentes son areas -- el del
    servidor no tiene por que poder mandar correos -- y asi añadir una
    herramienta nueva a un area no obliga a tocar los tres perfiles."""
    ds = []
    for h in perfil.get("herramientas") or []:
        if h.get("habilitada"):
            ds += (h.get("config") or {}).get("dominios") or []
    return sorted(set(ds))


def cargar(ruta=None) -> dict:
    """Lee el fichero de perfiles. Si no existe, devuelve el de serie."""
    if ruta is None:
        ruta = os.environ.get("VOZ_PERFILES")
    if ruta is None:
        por_defecto = Path(__file__).resolve().parent.parent / "perfiles_asistente.json"
        ruta = por_defecto if por_defecto.exists() else None
    if ruta is None:
        return json.loads(json.dumps(PERFIL_BASE))
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    validar(datos)
    return datos


def validar(datos: dict) -> None:
    """Comprueba la forma y avisa de lo que no encaja. Lanza ValueError.

    Es deliberadamente estricto con los NOMBRES (una categoria mal escrita
    seria un relleno que nunca suena y nadie se entera) y laxo con lo que aun
    no se usa: de las herramientas solo se mira que la forma sea la buena."""
    if not isinstance(datos, dict):
        raise ValueError("el fichero de perfiles no es un objeto JSON")
    if datos.get("version") != VERSION_FORMATO:
        raise ValueError(f"version {datos.get('version')!r}: este codigo lee "
                         f"la {VERSION_FORMATO}")
    perfiles = datos.get("perfiles")
    if not isinstance(perfiles, dict) or not perfiles:
        raise ValueError("falta el objeto 'perfiles' o esta vacio")
    defecto = datos.get("perfil_por_defecto")
    if defecto not in perfiles:
        raise ValueError(f"perfil_por_defecto {defecto!r} no esta en 'perfiles'")
    for nombre, p in perfiles.items():
        if not isinstance(p, dict):
            raise ValueError(f"perfil {nombre!r}: no es un objeto")
        voz = p.get("voz")
        if not isinstance(voz, dict) or not voz.get("voz"):
            raise ValueError(f"perfil {nombre!r}: falta 'voz.voz'")
        for campo, tipo in (("semilla", int), ("pasos", int)):
            if campo in voz and not isinstance(voz[campo], tipo):
                raise ValueError(f"perfil {nombre!r}: voz.{campo} no es entero")
        if "cfg_scale" in voz and not isinstance(voz["cfg_scale"], (int, float)):
            raise ValueError(f"perfil {nombre!r}: voz.cfg_scale no es numero")
        rellenos = p.get("rellenos") or {}
        if not isinstance(rellenos, dict):
            raise ValueError(f"perfil {nombre!r}: 'rellenos' no es un objeto")
        for cat, frases in rellenos.items():
            if cat not in CATEGORIAS:
                raise ValueError(f"perfil {nombre!r}: categoria {cat!r} "
                                 f"desconocida; las validas son "
                                 f"{sorted(CATEGORIAS)}")
            if not isinstance(frases, list) or not all(
                    isinstance(f, str) and f.strip() for f in frases):
                raise ValueError(f"perfil {nombre!r}, {cat}: se esperaba una "
                                 f"lista de textos no vacios")
        for h in p.get("herramientas") or []:
            if not isinstance(h, dict) or not h.get("id"):
                raise ValueError(f"perfil {nombre!r}: herramienta sin 'id'")
            if h.get("tipo") not in TIPOS_HERRAMIENTA:
                raise ValueError(f"perfil {nombre!r}, herramienta "
                                 f"{h.get('id')!r}: tipo {h.get('tipo')!r} no "
                                 f"es uno de {TIPOS_HERRAMIENTA}")


# ---------------------------------------------------------- la generacion --
def _wav(ruta: Path, pcm: bytes) -> None:
    with open(ruta, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, RITMO,
                                      RITMO * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)


def _pcm_de_wav(datos: bytes) -> bytes:
    """El trozo 'data' de un WAV. El servicio devuelve una cabecera de flujo
    (tamaños 0xFFFFFFFF), asi que no vale fiarse de las longitudes: se busca
    el trozo y se coge todo lo que venga detras."""
    i = datos.find(b"data")
    if i < 0:
        raise ValueError("la respuesta no parece un WAV (no hay trozo 'data')")
    return datos[i + 8:]


def sintetizar(url: str, token: str, texto: str, voz: dict,
               plazo: float = 120.0) -> bytes:
    """Un relleno, por POST /tts/stream. Devuelve PCM s16 mono a 24 kHz.

    Por la via HTTP y no por una sesion websocket A PROPOSITO: son frases
    sueltas y sin continuidad entre ellas, que es justo para lo que
    /tts/stream esta bien, y ademas asi el generador no depende del codigo de
    sesiones."""
    cuerpo = json.dumps({
        "texto": texto,
        "voz": voz.get("voz"),
        "cfg_scale": voz.get("cfg_scale", 3.5),
        "semilla": voz.get("semilla"),
        "pasos": voz.get("pasos"),
    }).encode()
    pet = urllib.request.Request(f"{url.rstrip('/')}/tts/stream", data=cuerpo,
                                 method="POST")
    pet.add_header("content-type", "application/json")
    if token:
        pet.add_header("authorization", f"Bearer {token}")
    with urllib.request.urlopen(pet, timeout=plazo) as r:
        return _pcm_de_wav(r.read())


def generar(datos: dict, url: str, token: str, cache=None,
            forzar: bool = False, podar: bool = True, avisar=print) -> dict:
    """Genera lo que falte y devuelve el manifiesto.

    NO LANZA si el servicio de voz no contesta: lo apunta en 'fallos' y sigue.
    El asistente tiene que poder arrancar sin voz.
    """
    dir_cache = Path(cache) if cache else directorio_cache()
    dir_cache.mkdir(parents=True, exist_ok=True)
    manifiesto = {"version": VERSION_FORMATO, "generado": time.time(),
                  "cache": str(dir_cache), "perfiles": {}, "fallos": []}
    vivos = set()
    nuevos = faltaban = 0
    for nombre, p in datos["perfiles"].items():
        voz = p["voz"]
        entrada = {"voz": voz, "categorias": {}}
        for cat, frases in (p.get("rellenos") or {}).items():
            lista = []
            for texto in frases:
                h = huella(texto, voz)
                fichero = dir_cache / f"{h}.wav"
                vivos.add(fichero.name)
                if fichero.exists() and not forzar:
                    ms = _duracion_ms(fichero)
                    lista.append({"id": h, "texto": texto, "ms": ms,
                                  "fichero": fichero.name})
                    continue
                faltaban += 1
                try:
                    pcm = sintetizar(url, token, texto, voz)
                except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
                    manifiesto["fallos"].append(
                        {"perfil": nombre, "categoria": cat, "texto": texto,
                         "error": f"{type(e).__name__}: {e}"})
                    avisar(f"[perfiles] sin generar {nombre}/{cat} "
                           f"{texto!r}: {type(e).__name__}: {e}")
                    continue
                _wav(fichero, pcm)
                nuevos += 1
                lista.append({"id": h, "texto": texto,
                              "ms": round(len(pcm) / 2 / RITMO * 1000, 1),
                              "fichero": fichero.name})
            if lista:
                entrada["categorias"][cat] = lista
        manifiesto["perfiles"][nombre] = entrada
    if podar:
        for f in dir_cache.glob("*.wav"):
            if f.name not in vivos:
                # Todo esto es regenerable: un WAV que ya no lo pide ningun
                # perfil es basura de una version anterior del fichero.
                f.unlink()
                avisar(f"[perfiles] sobra y se borra: {f.name}")
    (dir_cache / "catalogo.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    avisar(f"[perfiles] catalogo listo en {dir_cache}: {nuevos} generados, "
           f"{faltaban - nuevos} fallidos, "
           f"{sum(len(v) for e in manifiesto['perfiles'].values() for v in e['categorias'].values()) - nuevos} "
           f"reutilizados de la cache")
    return manifiesto


def _duracion_ms(ruta: Path) -> float:
    datos = ruta.read_bytes()
    return round(len(_pcm_de_wav(datos)) / 2 / RITMO * 1000, 1)


# ------------------------------------------------------------ en ejecucion --
# CUANDO SUENA CADA COSA. Los numeros salen de las latencias MEDIDAS del
# ciclo, no de la intuicion:
#
#   whisper nativo                 0,30 s
#   compuerta (¿iba para mi?)      0,47-0,70 s, en paralelo con el LLM
#   MiniMax, primer token          1,15-6,5 s
#   primer token -> primer sonido  0,53 s
#   => primer sonido real          1,7-7,0 s desde que dejas de hablar
#
# Y los rellenos duran 0,8-1,2 s (medido al generarlos; sale en el manifiesto).
#
# LA REGLA ES QUE UN RELLENO SOLO VALE SI EL HUECO QUE TAPA ES MAS LARGO QUE EL.
# Si el asistente iba a contestar en 300 ms, meter un "mmm, a ver" de 900 ms lo
# EMPEORA: retrasa la respuesta de verdad y ademas suena a tic. De ahi que no
# haya ningun relleno incondicional y que todos sean por plazo: si el audio real
# llega antes del plazo, no suena nada y el ciclo queda exactamente como estaba.
#
#   ACUSE (afirmacion) a 0,70 s. Es el veredicto de la compuerta: el primer
#   instante en que se SABE que la frase iba dirigida al asistente. Antes de
#   eso no se puede abrir la boca (se contestaria a una conversacion ajena).
#   Como el primer sonido real no llega nunca antes de 1,7 s, este acuse cae
#   siempre dentro del hueco y lo recorta de 1,7-7,0 s a 0,7 s.
#   Sin compuerta (pregunta escrita, o --sin-solapar) no hay veredicto que
#   esperar y no se usa: ahi manda "pensando".
#
#   PENSANDO a 1,20 s. Por debajo del suelo del primer sonido real (1,7 s), asi
#   que cuando el LLM va rapido el relleno ya ha terminado o casi. Por encima de
#   la suma whisper + compuerta, para no pisar al acuse.
#
#   ESPERANDO a 3,50 s. Solo se llega aqui con un LLM lento (el tercio alto del
#   rango medido). Un segundo relleno antes de esto se solaparia con el primero.
#
# NUNCA DOS A LA VEZ, Y NUNCA SOBRE EL HABLA REAL: el que decide es el puente,
# que apaga el temporizador en cuanto baja el primer PCM, y la pagina encola
# todo -- rellenos y habla -- en el MISMO reloj (`cabeza`), asi que solaparse es
# imposible por construccion, no por suerte.
UMBRALES = {"afirmacion": 0.70, "pensando": 1.20, "esperando": 3.50}


class Politica:
    """Que relleno toca, si es que toca alguno. Sin estado global y sin relojes
    propios: se le dice cuanto lleva la peticion y ella contesta.

    Se separa del puente A PROPOSITO: los umbrales son una decision medida y
    tienen que poder revisarse (y volver a medirse) sin tocar el servidor HTTP.
    """

    def __init__(self, catalogo, con_compuerta: bool, umbrales=None):
        self.cat = catalogo
        self.con_compuerta = con_compuerta
        self.umbrales = dict(UMBRALES)
        self.umbrales.update(umbrales or {})
        self.sonados = []          # categorias ya disparadas, en orden
        self.libre_en = 0.0        # segundo hasta el que hay relleno sonando

    def toca(self, transcurrido: float, hay_audio: bool,
             veredicto: bool | None) -> str | None:
        """Categoria a disparar ahora mismo, o None.

        `hay_audio` corta en seco: en cuanto suena el habla de verdad no se
        mete nada mas. `veredicto` es el de la compuerta (None = todavia no).
        """
        if hay_audio or transcurrido < self.libre_en:
            return None
        # CON COMPUERTA NO SE ABRE LA BOCA HASTA EL VEREDICTO, ni para un
        # "mmm". Si la frase no iba dirigida al asistente, el ciclo entero se
        # descarta sin que suene un solo byte -- y un relleno disparado por
        # plazo se saltaria justo esa garantia: se estaria contestando a una
        # conversacion ajena con una coletilla.
        if self.con_compuerta and veredicto is not True:
            return None
        for cat in ("afirmacion", "pensando", "esperando"):
            if cat in self.sonados or not self.cat.hay(cat):
                continue
            # Sin compuerta no hay nada que acusar: la pregunta la escribio el
            # propio usuario y ya sabe que le han leido. Ahi manda "pensando".
            if cat == "afirmacion" and not self.con_compuerta:
                continue
            if transcurrido >= self.umbrales[cat]:
                return cat
        return None

    # LAS OTRAS TRES NO VAN POR PLAZO, VAN POR SUCESO, y por eso no caben en
    # toca(): no hay un reloj que las dispare, hay algo que pasa.
    #
    #   CONSULTANDO cuando arranca una herramienta y el modelo no habia dicho
    #   nada antes de llamarla. Si dijo "voy a mirar tu calendario" NO suena:
    #   esa frase ya es el relleno, y encima es mejor que cualquiera de los
    #   nuestros porque habla de lo que se esta consultando de verdad.
    #
    #   NEGACION cuando la respuesta se cae SIN HABER SONADO NADA: una
    #   herramienta que revienta, el LLM que no contesta, la sesion de voz que
    #   se rompe antes del primer PCM. Sin esto el asistente se queda callado
    #   y el usuario no sabe si le ha oido.
    #
    #   CERRANDO cuando la locucion se corta A MITAD, con audio ya sonando: el
    #   freno de descarrile, el plazo de silencio, un error de la sesion
    #   despues del primer PCM. Es el unico caso en que añadir palabras que el
    #   LLM no dijo MEJORA la cosa: la alternativa es una frase que se corta
    #   sola en seco, que suena a averia. En un final normal no se usa, y esa
    #   es la diferencia con negacion: negacion es "no ha salido", cerrando es
    #   "salio a medias y lo remato".
    #
    # Son excluyentes por construccion (`hay_audio` decide cual de las dos) y
    # no se repiten, como las de plazo.
    def por_suceso(self, categoria: str) -> str | None:
        """La categoria si toca y hay audio para ella, o None."""
        if self.cat is None or categoria in self.sonados:
            return None
        return categoria if self.cat.hay(categoria) else None

    def remate(self, hay_audio: bool) -> str | None:
        """Que decir cuando la respuesta se ha ido al traste."""
        return self.por_suceso("cerrando" if hay_audio else "negacion")

    def apuntar(self, categoria: str, transcurrido: float, ms: float) -> None:
        """Se acaba de mandar uno: hasta que termine no se manda otro."""
        self.sonados.append(categoria)
        self.libre_en = transcurrido + ms / 1000.0


class Catalogo:
    """Los rellenos ya generados de UN perfil, listos para elegir.

    Lo unico que hace en caliente es elegir variante, y lo hace evitando la
    ultima que sonó de esa categoria: repetir la misma coletilla dos veces
    seguidas es lo que convierte una ayuda en un tic."""

    def __init__(self, manifiesto: dict, perfil: str, dir_cache: Path):
        self.perfil = perfil
        self.dir = Path(dir_cache)
        self.cats = (manifiesto.get("perfiles", {})
                     .get(perfil, {}).get("categorias", {}))
        self._ultimo = {}

    def hay(self, categoria: str) -> bool:
        return bool(self.cats.get(categoria))

    def buscar(self, categoria: str, texto: str) -> dict | None:
        """El relleno con ESE texto exacto, o None.

        Existe por 'confirmando': ahi el audio no es intercambiable. Si el
        usuario acaba de decir que si a borrar una nota, el remate tiene que
        ser 'Borrada.' y no un 'Apuntado.' elegido al azar de la misma
        categoria."""
        for x in self.cats.get(categoria) or []:
            if x["texto"].strip().lower() == (texto or "").strip().lower():
                return x
        return None

    def elegir(self, categoria: str, azar=None) -> dict | None:
        import random
        lista = self.cats.get(categoria) or []
        if not lista:
            return None
        if len(lista) > 1:
            lista = [x for x in lista if x["id"] != self._ultimo.get(categoria)]
        elegido = (azar or random.choice)(lista)
        self._ultimo[categoria] = elegido["id"]
        return elegido

    def ruta(self, id_: str) -> Path:
        return self.dir / f"{id_}.wav"


# ------------------------------------------------------------------- CLI ---
def _cli_banco_bufer(a) -> int:
    """Cuanto bufer necesita el reproductor para no dar microcortes.

    Se mide de verdad: se abre una sesion contra el servicio como hace el
    asistente -- frase a frase --, se apuntan los instantes de llegada de cada
    trozo de PCM y luego se simula el reproductor. Un reproductor que encola
    con reloj propio (que es lo que hace la pagina con `cabeza`) da un HUECO
    cada vez que el trozo llega despues de que el anterior haya terminado de
    sonar; con P segundos de bufer inicial, el hueco solo aparece si el
    retraso acumulado se come esos P.
    """
    import asyncio
    import websockets

    frases = a.texto.split("|")

    async def correr():
        u = (a.url.replace("http://", "ws://").replace("https://", "wss://")
             + "/tts/sesion/ws")
        llegadas = []
        async with websockets.connect(
                u, additional_headers={"Authorization": f"Bearer {a.token}"},
                max_size=None) as ws:
            await ws.send(json.dumps({"accion": "abrir", "voz": a.voz,
                                      "cfg_scale": a.cfg, "semilla": a.semilla}))
            t0 = time.perf_counter()
            for f in frases:
                await ws.send(json.dumps({"accion": "texto", "texto": f}))
            await ws.send(json.dumps({"accion": "fin"}))
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=300)
                tipo, largo = struct.unpack(">BI", m[:5])
                if tipo == 0:
                    llegadas.append((time.perf_counter() - t0,
                                     (largo // 2) / RITMO))
                else:
                    ev = json.loads(m[5:5 + largo])
                    if ev.get("tipo") in ("hecho", "error"):
                        break
        return llegadas

    llegadas = asyncio.run(correr())
    if not llegadas:
        print("sin audio")
        return 1
    total = sum(d for _, d in llegadas)
    print(f"trozos {len(llegadas)}  audio {total:.2f} s  "
          f"pared {llegadas[-1][0]:.2f} s  RTF que ve el reproductor "
          f"{llegadas[-1][0] / total:.3f}")
    print(f"{'prebufer s':>10} {'huecos':>7} {'hueco total ms':>15} "
          f"{'peor hueco ms':>14}")
    for p in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0):
        cabeza = llegadas[0][0] + p
        huecos, total_ms, peor = 0, 0.0, 0.0
        for t, dur in llegadas:
            if cabeza < t:
                huecos += 1
                total_ms += (t - cabeza) * 1000
                peor = max(peor, (t - cabeza) * 1000)
                cabeza = t
            cabeza += dur
        print(f"{p:10.2f} {huecos:7d} {total_ms:15.1f} {peor:14.1f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perfiles", default=os.environ.get("VOZ_PERFILES"),
                    help="fichero JSON de perfiles")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL",
                                                    "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    sub = ap.add_subparsers(dest="orden", required=True)
    sub.add_parser("init", help="escribe el fichero de perfiles de serie")
    sub.add_parser("listar", help="que perfiles y rellenos hay")
    g = sub.add_parser("generar", help="genera lo que falte en la cache")
    g.add_argument("--forzar", action="store_true")
    g.add_argument("--sin-podar", action="store_true")
    sub.add_parser("verificar", help="comprueba fichero y cache")
    b = sub.add_parser("banco-bufer", help="mide el bufer que hace falta")
    b.add_argument("--texto", default="El tren llega a las siete de la tarde.|"
                                      "Mañana por la mañana vamos al parque.|"
                                      "No olvides comprar pan y leche.")
    b.add_argument("--voz", default="sp-Spk1_man")
    b.add_argument("--cfg", type=float, default=3.5)
    b.add_argument("--semilla", type=int, default=11)
    a = ap.parse_args(argv)

    if a.orden == "init":
        destino = Path(a.perfiles or (Path(__file__).resolve().parent.parent
                                      / "perfiles_asistente.json"))
        if destino.exists():
            print(f"ya existe: {destino}")
            return 1
        destino.write_text(json.dumps(PERFIL_BASE, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        print(f"escrito {destino}")
        return 0

    datos = cargar(a.perfiles)

    if a.orden == "listar":
        print(json.dumps({n: {"voz": p["voz"],
                              "rellenos": {c: len(v) for c, v in
                                           (p.get("rellenos") or {}).items()},
                              "herramientas": len(p.get("herramientas") or [])}
                          for n, p in datos["perfiles"].items()},
                         ensure_ascii=False, indent=1))
        return 0

    if a.orden == "generar":
        m = generar(datos, a.url, a.token, a.cache, forzar=a.forzar,
                    podar=not a.sin_podar)
        return 1 if m["fallos"] else 0

    if a.orden == "verificar":
        dir_cache = Path(a.cache) if a.cache else directorio_cache()
        faltan = []
        for nombre, p in datos["perfiles"].items():
            for cat, frases in (p.get("rellenos") or {}).items():
                for t in frases:
                    h = huella(t, p["voz"])
                    if not (dir_cache / f"{h}.wav").exists():
                        faltan.append(f"{nombre}/{cat}: {t!r} ({h})")
        print(f"cache: {dir_cache}")
        print(f"faltan {len(faltan)} rellenos")
        for f in faltan:
            print("  " + f)
        return 1 if faltan else 0

    if a.orden == "banco-bufer":
        return _cli_banco_bufer(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
