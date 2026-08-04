"""API de voz del homelab: TTS con Piper y STT con whisper.cpp.

Pensada para que un agente tipo OpenClaw mande y entienda notas de voz:
  POST /tts   texto  -> audio (ogg/opus por defecto, que es lo que tragan
                        WhatsApp y Telegram como nota de voz)
  POST /stt   audio  -> texto (acepta lo que sea, ffmpeg normaliza a 16 kHz mono)

Las voces de Piper se cargan bajo demanda y se quedan en memoria; whisper corre
como servidor aparte (whisper-server) para no recargar el modelo en cada llamada.

Todo se configura por entorno, que es lo que inyecta el modulo NixOS:
  VOZ_VOICES_DIR   directorio con los .onnx de Piper
  VOZ_DEFECTO      voz por defecto
  VOZ_WHISPER_URL  donde escucha whisper-server
  VOZ_TOKEN        si esta puesto, exige bearer token
  VOZ_PROMPT_STT   prompt de vocabulario para whisper
  VOZ_FFMPEG       ruta del binario de ffmpeg
"""

import io
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from piper import PiperVoice, SynthesisConfig
from pydantic import BaseModel, Field

VOICES_DIR = Path(os.environ.get("VOZ_VOICES_DIR", "/var/lib/voz/voices"))
VOZ_DEFECTO = os.environ.get("VOZ_DEFECTO", "es_MX-claude-high")
WHISPER_URL = os.environ.get("VOZ_WHISPER_URL", "http://127.0.0.1:8081")
TOKEN = os.environ.get("VOZ_TOKEN", "").strip()
FFMPEG = os.environ.get("VOZ_FFMPEG", "ffmpeg")

# whisper acepta un prompt inicial que sesga el vocabulario. Sin el transcribe
# "WireGuard" como "We The War" y "homelab" como "omelab"; con el, acierta.
PROMPT_DEFECTO = os.environ.get(
    "VOZ_PROMPT_STT",
    "Vocabulario tecnico: homelab, Proxmox, WireGuard, Docker, contenedor, LXC, "
    "Caddy, Oracle, systemd, OpenClaw, Piper, whisper, backup, deploy, log, "
    "NixOS, Terraform, Ansible, SSH, VPN, Postgres.",
)

# Formatos de salida: extension -> argumentos de codec para ffmpeg.
FORMATOS = {
    "ogg": ["-c:a", "libopus", "-b:a", "32k", "-application", "voip", "-f", "ogg"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3"],
    "wav": None,  # se sirve tal cual sale de Piper
}
MIMES = {"ogg": "audio/ogg", "mp3": "audio/mpeg", "wav": "audio/wav"}

app = FastAPI(title="API de voz del homelab", version="1.0.0")
_voces: dict[str, PiperVoice] = {}
_bearer = HTTPBearer(auto_error=False)


def autorizar(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """Exige bearer token solo si VOZ_TOKEN esta configurado."""
    if not TOKEN:
        return
    if cred is None or cred.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="token invalido o ausente")


def voces_disponibles() -> list[str]:
    if not VOICES_DIR.is_dir():
        return []
    return sorted(p.stem for p in VOICES_DIR.glob("*.onnx"))


def cargar_voz(nombre: str) -> PiperVoice:
    if nombre not in _voces:
        modelo = VOICES_DIR / f"{nombre}.onnx"
        if not modelo.exists():
            raise HTTPException(
                status_code=404,
                detail=f"voz '{nombre}' no encontrada; disponibles: {voces_disponibles()}",
            )
        _voces[nombre] = PiperVoice.load(modelo)
    return _voces[nombre]


def convertir(wav_bytes: bytes, formato: str) -> bytes:
    """Pasa el WAV de Piper al formato pedido con ffmpeg."""
    args = FORMATOS[formato]
    if args is None:
        return wav_bytes
    proc = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg fallo al convertir a {formato}: {proc.stderr.decode()[:300]}",
        )
    return proc.stdout


class PeticionTTS(BaseModel):
    texto: str = Field(..., min_length=1, max_length=8000)
    voz: str | None = None
    formato: str = "ogg"
    # >1 habla mas lento, <1 mas rapido. 1.0 = velocidad nativa de la voz.
    velocidad: float = Field(1.0, gt=0.3, lt=3.0)


@app.get("/health")
def health() -> dict:
    try:
        r = httpx.get(f"{WHISPER_URL}/", timeout=2.0)
        stt_ok = r.status_code < 500
    except Exception:
        stt_ok = False
    return {
        "estado": "ok",
        "tts": {
            "motor": "piper",
            "voces": voces_disponibles(),
            "por_defecto": VOZ_DEFECTO,
            "cargadas": sorted(_voces),
        },
        "stt": {"motor": "whisper.cpp", "url": WHISPER_URL, "disponible": stt_ok},
        "auth": "bearer" if TOKEN else "abierta",
    }


@app.get("/voces")
def voces() -> dict:
    return {"voces": voces_disponibles(), "por_defecto": VOZ_DEFECTO}


@app.post("/tts")
def tts(pet: PeticionTTS, _=Depends(autorizar)) -> Response:
    formato = pet.formato.lower()
    if formato not in FORMATOS:
        raise HTTPException(400, f"formato debe ser uno de {sorted(FORMATOS)}")

    voz = cargar_voz(pet.voz or VOZ_DEFECTO)
    ini = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voz.synthesize_wav(pet.texto, wf, syn_config=SynthesisConfig(length_scale=pet.velocidad))
    wav_bytes = buf.getvalue()
    proc = time.perf_counter() - ini

    with wave.open(io.BytesIO(wav_bytes)) as wf:
        dur = wf.getnframes() / wf.getframerate()

    audio = convertir(wav_bytes, formato)
    return Response(
        content=audio,
        media_type=MIMES[formato],
        headers={
            "X-Duracion-S": f"{dur:.2f}",
            "X-Proceso-S": f"{proc:.2f}",
            "X-RTF": f"{proc / dur:.3f}" if dur else "0",
            "X-Voz": pet.voz or VOZ_DEFECTO,
            "Content-Disposition": f'inline; filename="voz.{formato}"',
        },
    )


@app.post("/stt")
async def stt(
    archivo: UploadFile = File(...),
    idioma: str = Form("es"),
    prompt: str | None = Form(None),
    _=Depends(autorizar),
) -> JSONResponse:
    crudo = await archivo.read()
    if not crudo:
        raise HTTPException(400, "archivo vacio")

    # whisper.cpp solo acepta WAV PCM de 16 kHz mono; normalizamos siempre.
    with tempfile.TemporaryDirectory() as tmp:
        entrada = Path(tmp) / "entrada"
        entrada.write_bytes(crudo)
        salida = Path(tmp) / "audio16k.wav"
        conv = subprocess.run(
            [FFMPEG, "-loglevel", "error", "-y", "-i", str(entrada),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(salida)],
            capture_output=True,
        )
        if conv.returncode != 0:
            raise HTTPException(400, f"no pude decodificar el audio: {conv.stderr.decode()[:300]}")

        with wave.open(str(salida)) as wf:
            dur = wf.getnframes() / wf.getframerate()

        ini = time.perf_counter()
        try:
            r = httpx.post(
                f"{WHISPER_URL}/inference",
                files={"file": ("audio16k.wav", salida.read_bytes(), "audio/wav")},
                data={
                    "language": idioma,
                    "response_format": "json",
                    "temperature": "0",
                    "prompt": PROMPT_DEFECTO if prompt is None else prompt,
                },
                timeout=300.0,
            )
        except httpx.RequestError as e:
            raise HTTPException(503, f"whisper-server no responde en {WHISPER_URL}: {e}")
        proc = time.perf_counter() - ini

    if r.status_code != 200:
        raise HTTPException(502, f"whisper devolvio {r.status_code}: {r.text[:300]}")

    texto = (r.json().get("text") or "").strip()
    return JSONResponse({
        "texto": texto,
        "idioma": idioma,
        "duracion_s": round(dur, 2),
        "proceso_s": round(proc, 2),
        "rtf": round(proc / dur, 3) if dur else None,
    })
