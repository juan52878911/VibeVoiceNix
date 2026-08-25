"""Fachada compatible con la API de OpenAI: `POST /v1/audio/speech`.

POR QUE EXISTE
La API nativa (`/tts`, `/tts/stream`, las sesiones) es la buena: expone lo que
este stack sabe hacer y nada mas. El problema es que NADIE la habla. Cualquier
cliente que ya sintetiza voz -- una libreria, un agente, un plugin de
navegador -- habla el dialecto de OpenAI, y para enchufarlo aqui hoy hay que
escribir un adaptador.

Esto es ese adaptador, hecho una sola vez y del lado del servidor. No sustituye
a nada: convive con `/tts` y traduce en los dos sentidos.

LO QUE NO CABE EN ESTE CONTRATO, Y ES DELIBERADO
El modo sesion -- una locucion continua a la que se le va metiendo texto, que
es lo que de verdad distingue a VibeVoice aqui -- no tiene equivalente en la
API de OpenAI: alli una peticion es un texto entero y una respuesta es un
audio entero. Asi que por `/v1` se sirve la sintesis de una tacada y el modo
sesion se queda en su endpoint nativo. Degradarlo para que entrase seria
perder justo lo que vale.

DIFERENCIAS QUE UN CLIENTE DEBE CONOCER (van todas en cabeceras `X-`)
  * `speed` se aplica ENTERO en Piper y RECORTADO en VibeVoice: alli el rango
    util es [0,85, 1,20] porque estirar mas alla se oye (ver estirar.py). Se
    recorta en vez de rechazar la peticion, y se dice en `X-Velocidad`.
  * `instructions` se ignora: ningun motor de aqui admite direccion de
    interpretacion por texto. Se avisa en `X-Instrucciones`.
  * Las voces canonicas de OpenAI (`alloy`, `nova`, ...) no existen aqui. Se
    mapean con VOZ_OPENAI_VOCES y, si no hay mapa, caen a la voz por defecto
    diciendolo en `X-Voz-Sustituida` -- devolver un 400 romperia al cliente
    justo por lo unico que no puede cambiar.

ENTORNO
  VOZ_VIBEVOICE_URL   donde escucha voz-stream (por defecto 127.0.0.1:8082)
  VOZ_VIBEVOICE_TOKEN su bearer; si no esta, se reutiliza VOZ_TOKEN
  VOZ_OPENAI_VOCES    mapa "alloy=es_MX-claude-high,nova=es_ES-sharvard-medium"
"""

import io
import os
import time
import wave

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from voz_api.api import (
    TOKEN,
    VOZ_DEFECTO,
    autorizar,
    convertir,
    sintetizar_piper,
    voces_disponibles,
)

VIBEVOICE_URL = os.environ.get("VOZ_VIBEVOICE_URL", "http://127.0.0.1:8082").rstrip("/")
VIBEVOICE_TOKEN = os.environ.get("VOZ_VIBEVOICE_TOKEN", TOKEN).strip()

# Rango de `velocidad` que admite voz-stream. Duplicado a proposito: este
# servicio no importa nada de vibevoice-cli (son dos paquetes y dos entornos
# de Python distintos), asi que el contrato se comprueba aqui y si alla cambia,
# lo que se ve es un 422 del proxy, no un audio raro.
VIBEVOICE_VEL = (0.85, 1.20)

# Ritmo de muestreo de VibeVoice. Sirve para reparar la cabecera del flujo.
VIBEVOICE_HZ = 24000

# --------------------------------------------------------------------------
# Formatos. Son los seis de OpenAI, y ninguno coincide del todo con los tres
# de `/tts`: por eso hay dos tablas y no una compartida. `wav` pasa igualmente
# por ffmpeg (a diferencia de `/tts`, que lo sirve tal cual) porque el WAV que
# llega del proxy trae la cabecera de flujo con los tamanos a 0xFFFFFFFF.
# --------------------------------------------------------------------------
FORMATOS = {
    "mp3": ["-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-f", "ogg"],
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
    # OpenAI define `pcm` como "wav sin cabecera, 24 kHz, 16 bits, mono".
    # El ritmo se fuerza: Piper sale a 22,05 kHz y el cliente no recibe
    # ninguna cabecera donde enterarse.
    "pcm": ["-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", "-f", "s16le"],
}
MIMES = {
    "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
    "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
}

# --------------------------------------------------------------------------
# Modelos. Los nombres canonicos de OpenAI van TODOS a Piper, incluido
# `tts-1-hd`, y no es pereza: voz-stream es un servicio opcional (perfil
# `pesado` en Docker, `services.vibevoice.enable` en NixOS) y en la mayoria de
# despliegues no esta levantado. Mandar el modelo por defecto de un cliente a
# un servicio que igual no existe convierte "compatible" en "falla a veces".
# Quien quiera VibeVoice lo pide por su nombre.
# --------------------------------------------------------------------------
MOTORES = {
    "tts-1": "piper",
    "tts-1-hd": "piper",
    "gpt-4o-mini-tts": "piper",
    "piper": "piper",
    "vibevoice": "vibevoice",
    "vibevoice-realtime-0.5b": "vibevoice",
    "voz-stream": "vibevoice",
}

# Las once voces canonicas de OpenAI. Aqui no existen; la lista sirve para
# distinguir "me pidieron una voz de OpenAI sin mapear" (-> voz por defecto)
# de "me pidieron una voz que no existe en ningun sitio" (-> 400).
VOCES_OPENAI = {
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse",
}


def _mapa_voces() -> dict[str, str]:
    """VOZ_OPENAI_VOCES = "alloy=es_MX-claude-high,nova=es_ES-sharvard-medium"."""
    mapa = {}
    for par in os.environ.get("VOZ_OPENAI_VOCES", "").split(","):
        if "=" in par:
            k, v = par.split("=", 1)
            mapa[k.strip().lower()] = v.strip()
    return mapa


MAPA_VOCES = _mapa_voces()

router = APIRouter(prefix="/v1", tags=["openai"])


# --------------------------------------------------------------------------
# Errores con la forma que espera un cliente de OpenAI:
#   {"error": {"message", "type", "param", "code"}}
# El SDK oficial lee `error.message` para construir la excepcion; con el
# `{"detail": ...}` de FastAPI el usuario ve "unknown error" y no sabe que
# paso.
# --------------------------------------------------------------------------
class ErrorOpenAI(StarletteHTTPException):
    def __init__(self, status: int, mensaje: str, tipo: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(status_code=status, detail=mensaje)
        self.tipo, self.param, self.code = tipo, param, code


def _cuerpo(mensaje: str, tipo: str, param: str | None, code: str | None) -> dict:
    return {"error": {"message": mensaje, "type": tipo, "param": param, "code": code}}


def instalar(app: FastAPI) -> None:
    """Monta /v1 y hace que SOLO ahi los errores salgan con forma de OpenAI.

    Se llama desde `voz_api/__init__.py` y no desde `api.py` para no crear un
    ciclo de importacion: este modulo importa de `api`, y `api` no importa de
    este.
    """
    app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        if not request.url.path.startswith("/v1/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return JSONResponse(
            _cuerpo(str(exc.detail),
                    getattr(exc, "tipo", "invalid_request_error"),
                    getattr(exc, "param", None),
                    getattr(exc, "code", None)),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validacion(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/v1/"):
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        e = exc.errors()[0] if exc.errors() else {}
        campo = ".".join(str(x) for x in e.get("loc", ())[1:]) or None
        # 400 y no 422: OpenAI nunca devuelve 422 y hay clientes que lo tratan
        # como un fallo del transporte y reintentan en bucle.
        return JSONResponse(
            _cuerpo(e.get("msg", "peticion invalida"), "invalid_request_error",
                    campo, "invalid_value"),
            status_code=400,
        )


class PeticionSpeech(BaseModel):
    model: str
    input: str = Field(..., min_length=1, max_length=8000)
    voice: str = ""
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    # Se aceptan para no rechazar peticiones legitimas de un cliente al dia,
    # pero aqui no hacen nada. Ver la cabecera X-Instrucciones.
    instructions: str | None = None
    stream_format: str | None = None


def _motor(modelo: str) -> str:
    m = MOTORES.get(modelo.strip().lower())
    if m is None:
        raise ErrorOpenAI(404, f"modelo '{modelo}' no existe en este servidor; "
                               f"disponibles: {sorted(MOTORES)}",
                          tipo="invalid_request_error", param="model",
                          code="model_not_found")
    return m


def _voz(pedida: str, motor: str) -> tuple[str, str | None]:
    """Devuelve (voz que se usara, aviso de sustitucion o None)."""
    pedida = (pedida or "").strip()
    defecto = VOZ_DEFECTO if motor == "piper" else ""  # "" = la de voz-stream

    if not pedida:
        return defecto, None
    if pedida.lower() in MAPA_VOCES:
        return MAPA_VOCES[pedida.lower()], None
    if motor == "vibevoice":
        # No se valida contra el catalogo remoto a proposito: haria falta una
        # llamada extra por peticion o una cache que se queda vieja. Si el
        # nombre no existe, voz-stream devuelve 404 y se traduce abajo.
        if pedida.lower() in VOCES_OPENAI:
            return defecto, f"{pedida} -> voz por defecto de voz-stream"
        return pedida, None
    if pedida in voces_disponibles():
        return pedida, None
    if pedida.lower() in VOCES_OPENAI:
        return defecto, f"{pedida} -> {defecto}"
    raise ErrorOpenAI(400, f"voz '{pedida}' no existe; instaladas: "
                           f"{voces_disponibles()}; o una de OpenAI: "
                           f"{sorted(VOCES_OPENAI)}",
                      param="voice", code="invalid_value")


def _reparar_wav(crudo: bytes) -> bytes:
    """Pone los tamanos reales en una cabecera RIFF de longitud desconocida.

    Hace falta en DOS sitios, y por el mismo motivo -- quien escribe no puede
    volver atras a rellenar el hueco:

      * voz-stream emite `cabecera_wav_flujo()` con los tamanos a 0xFFFFFFFF
        antes de saber cuanto audio habra, que es lo correcto para reproducir
        mientras se descarga.
      * ffmpeg hace lo MISMO al escribir en `pipe:1`, asi que el WAV que sale
        de `convertir()` tampoco trae tamanos. Medido: 48.078 bytes de audio
        con RIFFsize = 0xFFFFFFFF.

    Al tener el buffer entero los tamanos si se saben, y ponerlos importa: con
    0xFFFFFFFF el modulo `wave` de Python calcula 2.147.483.647 fotogramas, y
    hay clientes estrictos que se plantan.

    Se recorren los chunks en vez de asumir la cabecera canonica de 44 bytes:
    ffmpeg mete un `LIST` de 26 bytes entre `fmt ` y `data` (comprobado), asi
    que un offset fijo apuntaria a mitad de otra cosa.
    """
    if len(crudo) < 12 or crudo[:4] != b"RIFF" or crudo[8:12] != b"WAVE":
        return crudo
    b = bytearray(crudo)
    b[4:8] = (len(b) - 8).to_bytes(4, "little")
    i = 12
    while i + 8 <= len(b):
        cid = bytes(b[i:i + 4])
        tam = int.from_bytes(b[i + 4:i + 8], "little")
        if cid == b"data":
            b[i + 4:i + 8] = (len(b) - i - 8).to_bytes(4, "little")
            break
        if tam == 0xFFFFFFFF or i + 8 + tam > len(b):
            break  # cabecera rara: mejor devolverla como esta que estropearla
        i += 8 + tam + (tam & 1)  # los chunks van a tamano par
    return bytes(b)


def _duracion(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav)) as wf:
            return wf.getnframes() / wf.getframerate()
    except (wave.Error, ZeroDivisionError):
        # Repliegue: PCM de 16 bits mono al ritmo de VibeVoice.
        return max(0, len(wav) - 44) / (VIBEVOICE_HZ * 2)


async def _sintetizar_vibevoice(texto: str, voz: str, velocidad: float) -> bytes:
    cuerpo: dict = {"texto": texto, "velocidad": velocidad}
    if voz:
        cuerpo["voz"] = voz
    cabeceras = {"Authorization": f"Bearer {VIBEVOICE_TOKEN}"} if VIBEVOICE_TOKEN else {}
    try:
        async with httpx.AsyncClient(timeout=300.0) as cli:
            r = await cli.post(f"{VIBEVOICE_URL}/tts/stream", json=cuerpo, headers=cabeceras)
    except httpx.RequestError as e:
        raise ErrorOpenAI(503, f"voz-stream no responde en {VIBEVOICE_URL}: {e}. "
                               "Es un servicio opcional: si no lo tienes levantado, "
                               "usa model=tts-1 (Piper).",
                          tipo="api_error", code="upstream_unavailable")
    if r.status_code == 404:
        raise ErrorOpenAI(400, f"voz-stream rechazo la voz: {r.text[:300]}",
                          param="voice", code="invalid_value")
    if r.status_code != 200:
        raise ErrorOpenAI(502, f"voz-stream devolvio {r.status_code}: {r.text[:300]}",
                          tipo="api_error", code="upstream_error")
    return _reparar_wav(r.content)


@router.post("/audio/speech")
async def speech(pet: PeticionSpeech, _=Depends(autorizar)):
    formato = pet.response_format.strip().lower()
    if formato not in FORMATOS:
        raise ErrorOpenAI(400, f"response_format '{pet.response_format}' no admitido; "
                               f"usa uno de {sorted(FORMATOS)}",
                          param="response_format", code="invalid_value")

    motor = _motor(pet.model)
    voz, sustituida = _voz(pet.voice, motor)

    ini = time.perf_counter()
    if motor == "piper":
        # OJO CON EL SENTIDO. `speed` de OpenAI es un multiplicador de rapidez
        # (2.0 = el doble de rapido) y `length_scale` de Piper es de duracion
        # (2.0 = el doble de largo, o sea la mitad de rapido). Van INVERTIDOS,
        # y ademas al reves que el campo `velocidad` de voz-stream, que si es
        # de rapidez. Aqui esta el unico sitio donde eso se traduce.
        wav = sintetizar_piper(pet.input, voz, 1.0 / pet.speed)
        velocidad = pet.speed
    else:
        velocidad = min(max(pet.speed, VIBEVOICE_VEL[0]), VIBEVOICE_VEL[1])
        wav = await _sintetizar_vibevoice(pet.input, voz, velocidad)
    proc = time.perf_counter() - ini

    dur = _duracion(wav)
    audio = convertir(wav, FORMATOS[formato])
    if formato == "wav":
        # ffmpeg escribe en una tuberia y deja los tamanos sin rellenar.
        audio = _reparar_wav(audio)

    cabeceras = {
        "X-Motor": motor,
        "X-Voz": voz or "(defecto de voz-stream)",
        "X-Duracion-S": f"{dur:.2f}",
        "X-Proceso-S": f"{proc:.2f}",
        "X-RTF": f"{proc / dur:.3f}" if dur else "0",
        "Content-Disposition": f'inline; filename="speech.{formato}"',
    }
    if abs(velocidad - pet.speed) > 1e-3:
        cabeceras["X-Velocidad"] = (
            f"pedida {pet.speed}, aplicada {velocidad:.2f} "
            f"(VibeVoice solo admite {VIBEVOICE_VEL[0]}-{VIBEVOICE_VEL[1]})")
    if sustituida:
        cabeceras["X-Voz-Sustituida"] = sustituida
    if pet.instructions:
        cabeceras["X-Instrucciones"] = "ignoradas: ningun motor de este stack las admite"

    return Response(content=audio, media_type=MIMES[formato], headers=cabeceras)


@router.get("/models")
def modelos(_=Depends(autorizar)) -> dict:
    """El catalogo, con la forma de `GET /v1/models` de OpenAI.

    VibeVoice se sondea de verdad antes de anunciarlo: listarlo cuando no esta
    levantado convierte un 503 posterior en una sorpresa.
    """
    try:
        vibe_ok = httpx.get(f"{VIBEVOICE_URL}/health", timeout=2.0).status_code < 500
    except Exception:
        vibe_ok = False

    datos = []
    for nombre, motor in sorted(MOTORES.items()):
        if motor == "vibevoice" and not vibe_ok:
            continue
        datos.append({"id": nombre, "object": "model", "created": 0,
                      "owned_by": motor})
    return {"object": "list", "data": datos}
