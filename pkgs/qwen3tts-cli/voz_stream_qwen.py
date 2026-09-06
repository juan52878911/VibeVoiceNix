"""Servidor de :8082 con Qwen3-TTS detras, mismo contrato que voz_stream.py.

    QWEN3TTS_BIN=.../qwen_tts QWEN3TTS_MODELO=.../0.6b-base \
    QWEN3TTS_VOCES=./voces python voz_stream_qwen.py

QUE ES
Un shim fino: recibe POST /tts/stream con el mismo cuerpo que el servidor de
VibeVoice (texto, voz, semilla, velocidad...), lanza el motor C de Qwen3
(gabriele-mastrapasqua/qwen3-tts) y devuelve el PCM en flujo con la misma
cabecera WAV de longitud desconocida y las mismas cabeceras X-*. Los clientes
(dobla, la fachada OpenAI de voz-api, la consola) no notan el cambio.

POR QUE UN PROCESO POR PETICION Y NO EL SERVIDOR DEL MOTOR
El servidor HTTP del motor C fija la voz clonada AL ARRANCAR (--load-voice) y
no admite una voz por peticion. Aqui las voces cambian por peticion (cada
hablante del doblaje es una), asi que se invoca el binario por peticion con
la voz que toque. Cuesta la carga del modelo (~2-3 s: los pesos van por mmap
y la cuantizacion int8 se rehace) y se paga entera en el primer sonido. Para
el lote es irrelevante; para tiempo real es la cifra que decide la puerta.

VOCES
Un directorio con, por voz, cualquiera de:
    <voz>.bin      x-vector de 8 KB (--xvector-only): limpio, sin la sala
    <voz>.qvoice   injerto ICL (~25 MB): maximo parecido de timbre
    <voz>.wav      referencia cruda: se codifica en cada peticion (+1 s)
Se elige en ese orden. Los fabrica scripts/clonar_voz_qwen.py.

TROCEO
Qwen3 acelera el ritmo pasados ~100-150 caracteres (issue #239, cerrado como
"not planned"). Se corta el texto por frontera de frase en trozos de hasta
QWEN3TTS_TROZO caracteres, todos con la misma voz y la misma semilla, y se
emiten seguidos con una pausa corta. Es la version 1 de C4 del plan: el
ritmo se mantiene; la continuidad prosodica entre trozos es la que da la
puntuacion.

LO QUE NO HAY
Sesiones KV (/tts/sesion/*) y websocket: son cache del modelo de VibeVoice.
/health lo anuncia (sesiones.activas = false) para que el asistente web no
las use. cfg_scale, pasos, neg_cada y cola_final se aceptan y se ignoran.
"""
import asyncio
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# estirar.py (WSOLA, numpy puro) vive en vibevoice-cli; en el store se copia al
# lado de este fichero, en el repo se importa de su sitio.
_aqui = Path(__file__).resolve().parent
sys.path[:0] = [str(_aqui), str(_aqui.parent / "vibevoice-cli")]
from estirar import estirar  # noqa: E402

RITMO = 24000
BIN = os.environ.get("QWEN3TTS_BIN", "qwen_tts")
MODELO = os.environ.get("QWEN3TTS_MODELO", "")
VOCES_DIR = Path(os.environ.get("QWEN3TTS_VOCES", "voces"))
CUANT = os.environ.get("QWEN3TTS_CUANT", "int8")
HILOS = int(os.environ.get("QWEN3TTS_HILOS", "0")) or (os.cpu_count() or 4)
VOZ_DEFECTO = os.environ.get("QWEN3TTS_VOZ_DEFECTO", "")
IDIOMA_DEFECTO = os.environ.get("QWEN3TTS_IDIOMA", "es")
TROZO = int(os.environ.get("QWEN3TTS_TROZO", "160"))
PAUSA_S = float(os.environ.get("QWEN3TTS_PAUSA", "0.15"))
RTF_MEDIDO = float(os.environ.get("QWEN3TTS_RTF", "1.0"))
PUERTO = int(os.environ.get("VOZ_STREAM_PUERTO", "8082"))
DIRECCION = os.environ.get("VOZ_STREAM_HOST", os.environ.get("VOZ_STREAM_DIRECCION", "0.0.0.0"))
TOKEN = os.environ.get("VOZ_TOKEN", "").strip()

IDIOMAS = {"es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French",
           "it": "Italian", "de": "German", "ru": "Russian", "ja": "Japanese",
           "ko": "Korean", "zh": "Chinese"}
IGNORADOS = ["cfg_scale", "pasos", "neg_cada", "cola_final"]

if not TOKEN:
    print("[AVISO] VOZ_TOKEN vacio: el servicio queda ABIERTO a cualquiera", file=sys.stderr)
if not MODELO or not Path(MODELO).is_dir():
    raise SystemExit(f"QWEN3TTS_MODELO no apunta a un directorio de modelo: '{MODELO}'")
BIN = BIN if os.path.exists(BIN) else (shutil.which(BIN) or BIN)
if not os.path.exists(BIN):
    raise SystemExit(f"no encuentro el motor C: '{BIN}' (QWEN3TTS_BIN)")


def voces_instaladas():
    nombres = set()
    for patron in ("*.bin", "*.qvoice", "*.wav"):
        nombres.update(p.stem for p in VOCES_DIR.glob(patron))
    return sorted(nombres)


def argumentos_voz(voz):
    """La forma de cargar esa voz, en el orden de preferencia del docstring."""
    b, q, w = (VOCES_DIR / f"{voz}.bin", VOCES_DIR / f"{voz}.qvoice", VOCES_DIR / f"{voz}.wav")
    if b.exists():
        return ["--load-voice", str(b), "--xvector-only"]
    if q.exists():
        return ["--load-voice", str(q), "--icl-only"]
    if w.exists():
        return ["--ref-audio", str(w)]
    raise HTTPException(404, f"voz '{voz}' no instalada; hay: {voces_instaladas()}")


def trocear(texto, tope):
    """Trozos de hasta `tope` caracteres cortando por frontera de frase, y si
    una frase sola se pasa, por coma o por espacio. Nunca parte una palabra."""
    if tope <= 0 or len(texto) <= tope:
        return [texto]
    frases = re.split(r"(?<=[.!?;:])\s+", texto.strip())
    trozos, actual = [], ""
    for f in frases:
        while len(f) > tope:                      # frase mas larga que el tope
            corte = max(f.rfind(", ", 0, tope), f.rfind(" ", 0, tope))
            corte = corte if corte > 0 else tope
            trozos.append((actual + " " + f[:corte]).strip()); actual = ""
            f = f[corte:].lstrip(", ")
        if len(actual) + len(f) + 1 > tope and actual:
            trozos.append(actual); actual = f
        else:
            actual = (actual + " " + f).strip()
    if actual:
        trozos.append(actual)
    return [t for t in trozos if t]


def cabecera_wav_flujo(ritmo=RITMO):
    return b"".join([
        b"RIFF", struct.pack("<I", 0xFFFFFFFF), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, ritmo, ritmo * 2, 2, 16),
        b"data", struct.pack("<I", 0xFFFFFFFF),
    ])


_bearer = HTTPBearer(auto_error=False)


def autorizar(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if not TOKEN:
        return
    if cred is None or cred.credentials != TOKEN:
        raise HTTPException(401, "token invalido")


class PeticionTTS(BaseModel):
    texto: str = Field(..., min_length=1, max_length=8000)
    voz: str = VOZ_DEFECTO
    idioma: str = IDIOMA_DEFECTO
    semilla: Optional[int] = Field(None, ge=0, lt=2**31)
    velocidad: float = Field(1.0, ge=0.85, le=1.20)
    # Aceptados por compatibilidad con voz_stream.py; no significan nada aqui.
    cfg_scale: Optional[float] = None
    pasos: Optional[int] = None
    neg_cada: Optional[int] = None
    cola_final: Optional[int] = None


app = FastAPI(title="voz-stream (qwen3tts)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-RTF-Esperado", "X-Ritmo-Hz"])
_candado = asyncio.Lock()          # un motor a la vez: es lo que cabe en la CPU
_ultimo = {"rtf": None, "primer_sonido_s": None}


def orden_motor(texto, voz, idioma, semilla):
    lang = IDIOMAS.get(idioma.lower(), idioma if idioma[:1].isupper() else "Spanish")
    orden = [BIN, "-d", MODELO, *argumentos_voz(voz), "-l", lang, f"--{CUANT}",
             "-j", str(HILOS), "--text", texto, "--stdout"]
    if semilla is not None:
        orden += ["--seed", str(semilla)]
    return orden


async def pcm_del_motor(orden):
    """Lanza el motor y va cediendo el PCM que escribe por stdout."""
    proc = await asyncio.create_subprocess_exec(
        *orden, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    t0, octetos = time.perf_counter(), 0
    try:
        while True:
            trozo = await proc.stdout.read(RITMO * 2 // 5)     # 0,2 s por trozo
            if not trozo:
                break
            octetos += len(trozo)
            yield trozo
    finally:
        err = (await proc.stderr.read()).decode(errors="replace")
        await proc.wait()
        # RTF de pared, con la carga del modelo dentro: es lo que ve el cliente.
        segundos = octetos / 2 / RITMO
        if segundos:
            _ultimo["rtf"] = round((time.perf_counter() - t0) / segundos, 2)
        if proc.returncode != 0:
            print(f"[motor] salio con {proc.returncode}: {err[-400:]}", file=sys.stderr)


@app.get("/", response_class=PlainTextResponse)
def raiz():
    return "voz-stream (qwen3tts): POST /tts/stream, GET /voces, GET /health\n"


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "motor": "qwen3tts",
        "modelo": Path(MODELO).name,
        "dispositivo": "cpu",
        "cuantizacion": CUANT,
        "hilos": {"total": HILOS},
        "voz_defecto": VOZ_DEFECTO,
        "idiomas": sorted(IDIOMAS),
        "idioma_defecto": IDIOMA_DEFECTO,
        "trozo": TROZO,
        "rtf_esperado": RTF_MEDIDO,
        "rtf_ultimo": _ultimo["rtf"],
        "primer_sonido_ultimo_s": _ultimo["primer_sonido_s"],
        "ocupado": _candado.locked(),
        "auth": "bearer" if TOKEN else "abierta",
        "ignorados": IGNORADOS,
        "sesiones": {"activas": False, "abiertas": []},
    }


@app.get("/voces")
def voces(_=Depends(autorizar)):
    return {"voces": voces_instaladas(), "defecto": VOZ_DEFECTO}


@app.post("/tts/stream")
async def tts_stream(pet: PeticionTTS, _=Depends(autorizar)) -> StreamingResponse:
    voz = pet.voz or VOZ_DEFECTO
    if not voz:
        raise HTTPException(400, "sin voz: pon `voz` o QWEN3TTS_VOZ_DEFECTO")
    argumentos_voz(voz)                        # 404 antes de abrir el flujo
    trozos = trocear(pet.texto, TROZO)
    pausa = b"\x00" * (int(PAUSA_S * RITMO) * 2)

    async def generador():
        t0 = time.perf_counter()
        primero = True
        yield cabecera_wav_flujo()
        async with _candado:
            for i, t in enumerate(trozos):
                if i:
                    yield pausa
                orden = orden_motor(t, voz, pet.idioma, pet.semilla)
                if abs(pet.velocidad - 1.0) < 1e-3:
                    async for pcm in pcm_del_motor(orden):
                        if primero:
                            _ultimo["primer_sonido_s"] = round(time.perf_counter() - t0, 2)
                            primero = False
                        yield pcm
                else:
                    # WSOLA necesita el trozo entero: se pierde el flujo, no el tono.
                    partes = [p async for p in pcm_del_motor(orden)]
                    x = np.frombuffer(b"".join(partes), "<i2").astype(np.float32) / 32768
                    y = estirar(x, 1.0 / pet.velocidad)
                    yield (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()

    return StreamingResponse(
        generador(), media_type="audio/wav",
        headers={"Cache-Control": "no-store",
                 "X-Ritmo-Hz": str(RITMO),
                 "X-RTF-Esperado": str(RTF_MEDIDO),
                 "X-Motor": "qwen3tts",
                 "Content-Disposition": 'inline; filename="voz.wav"'})


if __name__ == "__main__":
    import uvicorn
    print(f"voz-stream (qwen3tts) en {DIRECCION}:{PUERTO}; modelo {Path(MODELO).name}, "
          f"{CUANT}, {HILOS} hilos, voces en {VOCES_DIR} ({len(voces_instaladas())})")
    uvicorn.run(app, host=DIRECCION, port=PUERTO, log_level="info")
