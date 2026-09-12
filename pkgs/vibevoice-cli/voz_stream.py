"""Servidor de TTS con respuesta HTTP en streaming.

    POST /tts/stream  {"texto": "..."}  ->  audio/wav troceado

El cliente empieza a oír ~0,2 s después de pedirlo, mientras el resto se
genera. Verificado: los trozos emitidos son bit a bit identicos al audio
completo (mismo md5 que la generacion no-streaming), salvo los fotogramas
callados de la ENTRADA, que desde el 11-09-2026 no se emiten -- son 0,27-0,40 s
de aire delante de la primera palabra, y lo que queda sigue siendo byte a byte
la cola del original (ver RecorteEntrada; VIBEVOICE_RECORTE_ENTRADA=0 lo
devuelve).

Y para narrar algo que aun se esta escribiendo -- la salida de un LLM, por
ejemplo -- hay ademas SESIONES, que son una sola locucion continua a la que se
le va metiendo texto:

    POST /tts/sesion/{id}        {"texto": "..."}    encola texto
    GET  /tts/sesion/{id}/audio                      un WAV, toda la locucion
    POST /tts/sesion/{id}/fin                        cierra la locucion

y la MISMA sesion por websocket, en una sola conexion bidireccional:

    WS   /tts/sesion/ws                              texto JSON -> marcos PCM

Los dos caminos comparten SesionViva entera, asi que dan el MISMO audio (se
comprueba por md5, ver el bloque WEBSOCKET mas abajo). El HTTP se queda porque
es lo que esta en produccion y lo que se puede probar con curl; el websocket
existe porque la interaccion es de ida y vuelta y de larga duracion -- se mete
texto mientras sale audio -- y partirla en peticiones sueltas es justo lo que
obliga a reabrir contexto una y otra vez.

Frase a frase con /tts/stream, cada una empieza desde cero y suena a lista de
frases sueltas. En una sesion el modelo no deja de hablar entre frases. Y no es
"parecido" a pasar todo el texto de golpe: es EL MISMO AUDIO, byte a byte
(medido con 6 semillas, ver scripts/sesiones_fidelidad.py). El bloque SESIONES,
mas abajo, explica como y que se probo antes.

Medido en la VM: primer sonido 23,21 s -> 0,20 s. El tiempo TOTAL no cambia
(22,4 s frente a 23,2), pero la espera percibida se divide por 116.

POR QUE UN PROCESO APARTE Y NO DENTRO DE voz-api
Este servicio carga VibeVoice (~2,3 GB residentes); voz-api solo tiene las
voces de Piper (~100 MB) y responde en decimas de segundo. Juntarlos haria
que un TTS pesado bloqueara las notas de voz rapidas, y en una VM de 5 GB la
regla es un modelo por proceso.

HONESTIDAD SOBRE EL RTF
Con RTF ~2,2 el flujo produce audio a ~0,5x tiempo real: el primer sonido
llega enseguida, pero un reproductor ingenuo se quedara sin datos y
entrecortara. Para reproduccion continua hace falta un bufer inicial de ~10,7 s
(aun asi, el doble de bueno que esperar 23). Se anuncia en la cabecera
X-RTF-Esperado para que el cliente decida su politica.

Si algun dia el RTF baja de 1, ESTE MISMO codigo da reproduccion continua con
0,2 s de espera sin tocar una linea: streaming y RTF son ortogonales.

Configuracion por entorno, igual que el resto del stack:
  VIBEVOICE_MODELO   directorio del modelo
  VIBEVOICE_VOCES    directorio de los .pt de voz
  VIBEVOICE_PASOS    pasos de difusion (6)
  VIBEVOICE_VOZ      voz por defecto
  VOZ_STREAM_PUERTO  puerto de escucha (8082)
  VOZ_TOKEN          si esta puesto, exige bearer token
"""

import asyncio
import contextlib
import copy
import ctypes
import gc
import hashlib
import json
import os
import platform
import queue
import re
import secrets
import sys
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estirar import estirar  # noqa: E402
import uvicorn
from fastapi import (
    Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

MODELO_DIR = os.environ["VIBEVOICE_MODELO"]
VOCES_DIR = Path(os.environ["VIBEVOICE_VOCES"])
PASOS = int(os.environ.get("VIBEVOICE_PASOS", "6"))
VOZ_DEFECTO = os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man")
# Semilla del ruido de la difusion cuando el cliente no manda ninguna. Vacia
# (el defecto) = sorteo por peticion, como siempre: cada llamada con el mismo
# texto da un audio distinto. Con un numero, el servicio es determinista por
# defecto -- mismas entradas, mismo md5 -- y un cliente que QUIERA variedad
# manda "semilla": null. Se anuncia en /health como semilla_defecto. Cual poner,
# si alguna, lo decide el banco, no este fichero.
_semilla_env = os.environ.get("VIBEVOICE_SEMILLA", "").strip()
SEMILLA_DEFECTO: Optional[int] = int(_semilla_env) if _semilla_env else None
TOKEN = os.environ.get("VOZ_TOKEN", "").strip()
if not TOKEN:
    # Legitimo en una maquina aislada, pero tiene que VERSE: un .env mal
    # generado deja el servicio abierto sin que nadie lo note.
    print(
        "[AVISO] VOZ_TOKEN vacio: el servicio queda ABIERTO a cualquiera "
        "que alcance este puerto.",
        flush=True,
    )
def nucleos_fisicos() -> int:
    """Nucleos FISICOS utilizables, no hilos logicos ni nucleos de la maquina.

    Los tres son numeros distintos y confundirlos es justo el error que este
    proyecto ya midio: en el i7-8700T (6 fisicos / 12 logicos) usar los 12
    empeora el RTF un 24 % -- los hilos hermanos de un mismo nucleo comparten
    la unidad AVX2 y el puerto de memoria, asi que se estorban en vez de
    sumar. Ver docs/optimizacion.md.

    Por orden de fiabilidad:
      1. sched_getaffinity: lo que el cgroup/taskset deja usar DE VERDAD. En un
         contenedor con `--cpuset-cpus 0-3` esto da 4 y os.cpu_count() da 12.
      2. /proc/cpuinfo agrupando por (physical id, core id): separa fisicos de
         hermanos SMT en Linux.
      3. hw.perflevel0.physicalcpu (macOS): los nucleos de RENDIMIENTO de un
         Apple Silicon. Contar tambien los de eficiencia mete en el reparto
         nucleos ~3x mas lentos, y con trabajo repartido a partes iguales el
         lote entero va al ritmo del mas lento.
      4. os.cpu_count() // 2 como ultimo recurso, asumiendo SMT.
    """
    permitidos = None
    if hasattr(os, "sched_getaffinity"):
        try:
            permitidos = len(os.sched_getaffinity(0))
        except OSError:
            permitidos = None

    fisicos = None
    try:
        if sys.platform == "darwin":
            import subprocess
            for clave in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
                r = subprocess.run(["sysctl", "-n", clave],
                                   capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    fisicos = int(r.stdout.strip())
                    break
        else:
            nucleos, actual = set(), {}
            for linea in Path("/proc/cpuinfo").read_text().splitlines():
                if ":" not in linea:
                    if actual:
                        nucleos.add((actual.get("physical id", "0"),
                                     actual.get("core id", str(len(nucleos)))))
                        actual = {}
                    continue
                k, _, v = linea.partition(":")
                actual[k.strip()] = v.strip()
            if actual:
                nucleos.add((actual.get("physical id", "0"),
                             actual.get("core id", str(len(nucleos)))))
            fisicos = len(nucleos) or None
    except Exception:
        fisicos = None

    if fisicos is None:
        fisicos = max(1, (os.cpu_count() or 2) // 2)
    # La afinidad es un TOPE, no una alternativa: si el cgroup da 2 CPUs, dan
    # igual los 6 nucleos que tenga la maquina por debajo.
    if permitidos:
        fisicos = min(fisicos, permitidos)
    return max(1, fisicos)


def detectar_hilos() -> int:
    """Hilos de inferencia. OMP_NUM_THREADS manda; si no, los fisicos.

    Se deja UNO libre a partir de 8 nucleos para que el resto del stack --
    whisper, voz-api, el propio servidor HTTP -- pueda responder mientras esto
    genera. Por debajo de 8 no se reserva nada: quitarle un nucleo a una
    maquina de 4 cuesta un 25 % del computo y ahi no sobra.
    """
    puesto = os.environ.get("OMP_NUM_THREADS", "").strip()
    if puesto.isdigit() and int(puesto) > 0:
        return int(puesto)
    n = nucleos_fisicos()
    return n - 1 if n >= 8 else n


HILOS = detectar_hilos()
# Que se vea: si la deteccion se equivoca, este numero es la primera pista.
print(f"[arranque] {HILOS} hilos de inferencia "
      f"({nucleos_fisicos()} nucleos fisicos utilizables)", flush=True)
# torch NO lee OMP_NUM_THREADS cuando su backend es nativo en vez de OpenMP
# (el caso en macOS ARM), asi que se le dice explicitamente. Sin esto el
# reparto adaptativo no llegaba al camino torch.
torch.set_num_threads(HILOS)

# Motor de inferencia: "torch" (RTF 2,19) u "openvino" (RTF 1,09).
MOTOR = os.environ.get("VIBEVOICE_MOTOR", "torch")
# "auto" (por defecto), "cpu", "cuda" o "mps". Ver elegir_dispositivo().
DISPOSITIVO_PEDIDO = os.environ.get("VIBEVOICE_DISPOSITIVO", "auto").strip().lower()
OV_CODIGO = os.environ.get("VIBEVOICE_OV_CODIGO", "")
IR_LM = os.environ.get("VIBEVOICE_IR_LM", "")
IR_CABEZA = os.environ.get("VIBEVOICE_IR_CABEZA", "")
IR_ACUSTICO = os.environ.get("VIBEVOICE_IR_ACUSTICO", "")
# Cuanto se frena la extrapolacion de la guia (0 = nada, 1 = del todo). Ver
# frenar_guia(): sin esto, una locucion larga con cfg alto se desboca de
# volumen hasta recortar. El defecto se midio ahi.
FRENO_GUIA = float(os.environ.get("VIBEVOICE_FRENO_GUIA", "0.75"))


def elegir_dispositivo() -> str:
    """Donde corre el modelo. Por defecto, la mejor opcion que haya.

    Aviso para este homelab en concreto: aqui la unica GPU es una Intel UHD
    630, que torch NI SIQUIERA VE -- no hay backend para ella. Se midio ademas
    con whisper por Vulkan que resulta 2,5 veces MAS LENTA que la CPU, asi que
    tampoco compensaria. En esta maquina 'auto' siempre da cpu, y esta bien.

    Esto sirve para llevarse el servicio a una maquina con NVIDIA, o a un Mac
    con Apple Silicon, sin tocar nada.
    """
    if DISPOSITIVO_PEDIDO != "auto":
        return DISPOSITIVO_PEDIDO
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DISPOSITIVO = elegir_dispositivo()
EN_GPU = DISPOSITIVO != "cpu"
# fp16 en GPU: la mitad de memoria y el doble de rendimiento donde hay tensor
# cores. En CPU no aporta nada -- ahi lo que gana es int8 dinamico, que a su
# vez SOLO esta implementado para CPU. De ahi que sean dos caminos y no un
# parametro.
TIPO = torch.float16 if EN_GPU else torch.float32

RITMO = 24_000  # Hz de salida del modelo
# Se anuncia al cliente en X-RTF-Esperado para que elija su politica de bufer.
# En GPU no hay medida propia: se deja la de CPU, que sobreestima. Equivocarse
# por arriba solo hace que el cliente reserve mas bufer del necesario; por
# abajo le cortaria el audio a mitad.
RTF_MEDIDO = 1.1 if MOTOR == "openvino" else 2.2

_estado: dict = {}
# Un candado: UNA generacion a la vez. El modelo ya satura los 6 nucleos, asi
# que dos en paralelo solo harian ambas mas lentas.
_candado = asyncio.Lock()
# Y este, ademas, porque las sesiones generan desde su PROPIO hilo, que no pasa
# por el candado asincrono de /tts/stream. No es solo cuestion de rendimiento:
# el planificador de difusion es un objeto COMPARTIDO del modelo
# (model.noise_scheduler) con estado interno por solve -- step_index,
# model_outputs --, asi que dos generaciones a la vez se corrompen la una a la
# otra. Sin sesiones vivas nadie lo disputa y tomarlo cuesta nanosegundos.
#
# LO QUE ESTE CANDADO NO CUBRE POR SI SOLO: DOS SESIONES VIVAS A LA VEZ
# Una sesion SUELTA el candado mientras espera texto (TextoEnCurso._esperar),
# que es lo correcto -- callada no debe secuestrar la CPU de nadie --, pero
# significa que otra puede colarse EN MITAD de su locucion. Y la que entra hace
# torch.manual_seed() y consume el RNG GLOBAL, asi que la primera reanudaba con
# otro ruido del que le tocaba.
#
# MEDIDO ANTES DEL ARREGLO, mismas 2 frases y misma semilla (11), alimentando
# frase a frase y esperando la pausa entre ellas:
#   una sola sesion         4,00 s  md5 34b42c3e...
#   dos a la vez, la 1a     4,00 s  md5 34b42c3e...   (igual)
#   dos a la vez, la 2a     4,00 s  md5 b0bfd38b...   (DISTINTO)
# Daba igual el transporte -- HTTP y websocket comparten el codigo de sesion --
# y alimentando de golpe no pasaba: sin pausa nadie suelta el candado a mitad.
#
# No era corrupcion -- el audio sonaba bien --, pero dejaba de cumplirse "misma
# semilla = mismo audio", que es justo lo que promete el campo `semilla`.
#
# EL ARREGLO: CADA SESION SE LLEVA SU ESTADO PUESTO
# SesionViva._pausar/_reanudar fotografian el estado global antes de soltar el
# candado y lo reponen despues de recuperarlo, de modo que cada sesion tiene su
# PROPIO hilo de ruido aunque el generador sea un objeto compartido. Alli esta
# el detalle de por que asi y no pasando un torch.Generator al modelo. El RNG
# fue lo primero que se vio; despues se midio que la intrusa pisa tambien los
# pasos de difusion, neg_cada, el contador de la rampa de arranque y el remate,
# y la foto se amplio a todo eso (foto_generacion/reponer_generacion, junto a
# _REMATE, con la medida).
#
# El noise_scheduler, en cambio, no necesita nada: sample_speech_tokens() lo
# reinicia con set_timesteps() al empezar cada latente y no suelta el candado en
# medio, asi que su estado por solve -- step_index, model_outputs -- nunca cruza
# una pausa. Lo que si sigue haciendo falta es el candado: dos generaciones A LA
# VEZ si se lo corromperian.
#
# CON EL MOTOR OPENVINO HAY UN TERCER ESTADO, Y SE ARREGLA EN OTRO SITIO
# El decodificador acustico compilado guarda las colas de sus convoluciones
# dentro del IR, que es UNO para todo el proceso: no solo cruza las pausas, es
# que cruzaba peticiones enteras. No se cubre desde aqui porque este fichero no
# conoce el motor; lo hace AcusticoOV (pkgs/vibevoice-ov/motor.py) atando el
# estado al objeto cache que generate() crea en cada llamada, que es justo la
# misma unidad de aislamiento que el RNG de aqui.
_candado_modelo = threading.Lock()
_bearer = HTTPBearer(auto_error=False)


class GeneracionCancelada(Exception):
    """Senal interna: el cliente se fue, aborta generate()."""


class LocucionDescarrilada(Exception):
    """El modelo agoto el texto sellado y NO emitio su EOS: sigue pidiendo
    ventanas y generando audio sin texto detras. Se corta la locucion.

    Visto en produccion (2026-08-05 23:31, sesion ws-8247c2d0): una respuesta
    normal de ~330 tokens termino de leerse y el modelo siguio generando; al
    saltar el tope de cache llevaba 766 posiciones de mas ("-766 tokens
    pendientes" en el registro) y aun asi siguio seis minutos, hasta que el
    cliente se desconecto. El tope no podia frenarlo: sellar solo hace que las
    lecturas devuelvan vacio, que es justo lo que ya pasaba. Esta excepcion es
    el freno que faltaba: desmonta generate() desde la lectura de texto."""


def devolver_memoria() -> None:
    """Libera lo suelto y DEVUELVE la memoria al sistema operativo.

    gc.collect() por si solo no basta: glibc conserva en sus arenas lo que
    Python libera, asi que el RSS no baja aunque los objetos hayan muerto.
    Se midio: sin esto el servicio se quedaba en 4400 MB residentes tras el
    calentamiento -de los ~2400 que realmente necesita- y dejaba la VM con
    69 MB libres y el swap casi agotado.

    malloc_trim(0) es de glibc; en otra libc simplemente no existe y no pasa
    nada, de ahi el try.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    # En GPU la VRAM la retiene el asignador de torch, al que gc.collect() no
    # llega. Sin vaciarla, el pico de una sintesis se acumula con el de la
    # siguiente hasta el out-of-memory, que en GPU no perdona.
    if EN_GPU:
        cache = getattr(getattr(torch, DISPOSITIVO, None), "empty_cache", None)
        if cache is not None:
            cache()


def autorizar(cred: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if not TOKEN:
        return
    if cred is None or cred.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="token invalido o ausente")


# ---------------------------------------------------------------------------
# SOLAPAR EL DECODIFICADOR ACUSTICO CON EL RESTO DEL BUCLE
# ---------------------------------------------------------------------------
#
# EL HALLAZGO QUE LO PERMITE: el decodificador acustico es un SUMIDERO.
# Comprobado leyendo el bucle de Microsoft
# (modeling_vibevoice_streaming_inference.py, lineas 776-804):
#
#     speech_latent  = sample_speech_tokens(...)      # cabeza de difusion
#     audio_chunk    = acoustic_tokenizer.decode(...) # <- el 42% del tiempo
#     audio_chunks[idx].append(audio_chunk[i])        # se guarda
#     audio_streamer.put(audio_chunk, ...)            # se emite
#     acoustic_embed = acoustic_connector(speech_latent)   # <- LATENT, no audio
#
# La realimentacion autorregresiva pasa por `acoustic_connector(speech_latent)`.
# El audio decodificado NO vuelve a entrar en el modelo: solo se guarda y se
# emite. Asi que decode() esta en el camino critico unicamente porque se llama
# de forma sincrona, no porque el bucle lo necesite.
#
# MEDIDO en un Apple M4 (motor torch-int8, 6 hilos, texto de 180 caracteres,
# semilla 11), reparto DENTRO de una generate() real:
#
#     tts_lm      189 llamadas   20,47 ms   40,3 %
#     cabeza      540 llamadas    2,56 ms   14,4 %
#     acustico     90 llamadas   45,19 ms   42,4 %   <- se puede solapar
#     resto                                  2,8 %
#
# Con el decodificador fuera del camino critico el techo teorico es
# max(42,4 ; 57,6) = 57,6 % del tiempo, o sea 1,74x. Lo que se consigue de
# verdad depende de cuanto se estorben las dos etapas por ancho de banda de
# memoria, que en esta maquina es el recurso escaso; por eso hay medicion y
# no solo teoria (ver docs/optimizacion.md).
#
# POR QUE UN BUFER PREASIGNADO Y NO UN "FUTURO"
# generate() hace `torch.cat(audio_chunks)` al final PASE LO QUE PASE -- solo
# el `return` mira `return_speech`, no el concat --, asi que lo que devuelva
# decode() tiene que ser un tensor de verdad, no un objeto perezoso. La salida
# tiene forma fija y conocida (un latente -> 3200 muestras), asi que se
# devuelve un tensor VACIO del tamano bueno y el worker lo rellena in situ.
# Quien lo lee ve el dato porque nadie lo lee antes de que el worker acabe.
#
# POR QUE UN SOLO WORKER Y EN ORDEN ESTRICTO
# El decodificador es causal y con estado: las colas de sus convoluciones las
# deja la llamada anterior. Dos decode() a la vez, o en otro orden, darian otro
# audio. Un unico hilo con una cola FIFO conserva el orden exacto de la version
# sincrona, que es lo que hace que el audio salga identico bit a bit.
#
# LA EMISION TAMBIEN SE MUEVE AL WORKER
# Si `streamer.put()` se quedara en el hilo de generate() habria que esperar
# ahi al decode -- y no se solaparia nada. Asi que el worker, tras rellenar el
# bufer, llama el mismo al put() de verdad. Efecto colateral: la cancelacion
# cooperativa (put() lanzando GeneracionCancelada) ya no salta en el hilo de
# generate(), asi que la excepcion se guarda y se relanza en el decode()
# siguiente. El corte tarda como mucho un trozo mas que antes: ~133 ms.
SOLAPAR_DECODER = os.environ.get("VIBEVOICE_SOLAPAR_DECODER", "1") not in ("0", "no")
# Hilos para el decodificador cuando va solapado. El resto son para el camino
# principal (tts_lm + cabeza). Solo el motor OpenVINO puede repartirlos de
# verdad: INFERENCE_NUM_THREADS es por modelo compilado, mientras que en torch
# el pool intra-op es uno para todo el proceso.
HILOS_DECODER = int(os.environ.get("VIBEVOICE_HILOS_DECODER", "0")) or max(1, HILOS // 2)


# --------------------------------------------------------------------- crono --
# CUAL DE LAS DOS TUBERIAS ES EL CUELLO
# Con el decodificador solapado, sumar los tiempos por componente ya no dice
# donde va el reloj de pared: el bucle (tts_lm + cabeza) y el decodificador
# corren A LA VEZ, asi que la suma pasa de 100 %. Lo que hace falta saber es
# quien espera a quien:
#
#   contrapresion  segundos que el BUCLE pasa bloqueado en cola.put() porque el
#                  decodificador no da abasto -> el cuello es el decodificador
#   hambre         segundos que el WORKER pasa esperando trabajo -> el cuello
#                  es el bucle
#
# Se acumula siempre (dos perf_counter() por fotograma, ~0,2 us) y se lee por
# GET /crono, que ademas lo pone a cero: asi cada banco mide SU generacion y no
# arrastra la anterior.
CRONO_TUBERIA = {"generate": 0.0, "generaciones": 0, "audio_s": 0.0,
                 "contrapresion": 0.0, "trabajo_worker": 0.0,
                 "hambre_worker": 0.0, "decodes": 0}


def crono_cero() -> dict:
    previo = dict(CRONO_TUBERIA)
    for k in CRONO_TUBERIA:
        CRONO_TUBERIA[k] = 0.0 if isinstance(CRONO_TUBERIA[k], float) else 0
    return previo


class _Trabajo:
    """Un decode encolado y su emision, que se deciden en hilos distintos.

    El worker rellena el bufer; el hilo de generate() dice a donde va. Cual de
    los dos llega antes NO esta garantizado -- entre decode() y put() solo hay
    un append de lista, pero "casi siempre" no es "siempre" --, asi que el que
    llegue el ultimo es el que emite. El candado hace atomica esa decision.
    """

    __slots__ = ("bufer", "args", "kw", "destino", "indices", "listo", "emitido",
                 "candado")

    def __init__(self, bufer, args, kw):
        self.bufer, self.args, self.kw = bufer, args, kw
        self.destino = self.indices = None
        self.listo = self.emitido = False
        self.candado = threading.Lock()

    def reclamar_emision(self, *, desde_worker: bool):
        """Devuelve (destino, indices) si a QUIEN LLAMA le toca emitir."""
        with self.candado:
            if desde_worker:
                self.listo = True
                if self.destino is None or self.emitido:
                    return None
            else:
                if not self.listo:
                    return None       # ya emitira el worker, que llegara despues
                if self.emitido:
                    return None
            self.emitido = True
            return self.destino, self.indices


class DecodificadorSolapado:
    """Saca acoustic_tokenizer.decode() del camino critico de generate()."""

    def __init__(self, decode_real, profundidad: int = 2):
        self._decode = decode_real
        # Cola CORTA a proposito: es cuanto puede adelantarse el bucle al
        # decodificador. Mas profundidad no acelera -- el cuello es el propio
        # decodificador -- y solo retrasaria la cancelacion y el audio en vuelo.
        self._cola: "queue.Queue" = queue.Queue(maxsize=profundidad)
        self._molde = None          # forma/dtype, del primer decode (sincrono)
        self._fallo = None          # excepcion del worker, para relanzar en generate()
        self._trabajos: dict = {}   # id(bufer) -> _Trabajo pendiente de destino
        self._hilo = threading.Thread(target=self._bucle, name="decoder-acustico",
                                      daemon=True)
        self._hilo.start()

    # ---- hilo trabajador ----
    def _bucle(self):
        while True:
            espera = time.perf_counter()
            trabajo = self._cola.get()
            CRONO_TUBERIA["hambre_worker"] += time.perf_counter() - espera
            try:
                if trabajo is None:
                    return
                if self._fallo is None:
                    faena = time.perf_counter()
                    salida = self._decode(*trabajo.args, **trabajo.kw)
                    CRONO_TUBERIA["trabajo_worker"] += time.perf_counter() - faena
                    CRONO_TUBERIA["decodes"] += 1
                    if tuple(salida.shape) != tuple(trabajo.bufer.shape):
                        raise RuntimeError(
                            "el decodificador cambio de forma: esperaba "
                            f"{tuple(trabajo.bufer.shape)} y dio "
                            f"{tuple(salida.shape)}")
                    trabajo.bufer.copy_(salida)
                    envio = trabajo.reclamar_emision(desde_worker=True)
                    if envio is not None and envio[0] is not None:
                        # El put() de verdad, ya con el audio dentro. Puede
                        # lanzar GeneracionCancelada: es la senal de que el
                        # cliente se fue, y se propaga por _fallo.
                        envio[0].put(trabajo.bufer, envio[1])
            except BaseException as e:      # noqa: BLE001 - se relanza tal cual
                if self._fallo is None:
                    self._fallo = e
            finally:
                self._cola.task_done()

    # ---- cara visible ----
    def decode(self, latents, *a, **kw):
        self._relanzar()
        if self._molde is None:
            # La primera va sincrona: hace falta su forma para poder preasignar
            # las siguientes, y ademas es la que paga el cebado del decoder.
            salida = self._decode(latents, *a, **kw)
            self._molde = (tuple(salida.shape), salida.dtype)
            return salida
        forma, tipo = self._molde
        bufer = torch.empty(forma, dtype=tipo)
        trabajo = _Trabajo(bufer, (latents, *a), kw)
        self._trabajos[id(bufer)] = trabajo
        # put() bloquea cuando la cola esta llena: es la contrapresion que evita
        # que el bucle se adelante sin limite al decodificador.
        frenado = time.perf_counter()
        self._cola.put(trabajo)
        CRONO_TUBERIA["contrapresion"] += time.perf_counter() - frenado
        return bufer

    def emitir(self, bufer, streamer, indices) -> None:
        """Dice a donde va el audio de `bufer`. Lo llama el streamer envuelto."""
        trabajo = self._trabajos.pop(id(bufer), None)
        if trabajo is None:
            # No salio de un decode diferido (la primera, que va sincrona): el
            # dato ya esta, se emite aqui mismo.
            if streamer is not None:
                streamer.put(bufer, indices)
            return
        with trabajo.candado:
            trabajo.destino, trabajo.indices = streamer, indices
        envio = trabajo.reclamar_emision(desde_worker=False)
        if envio is not None and envio[0] is not None:
            envio[0].put(trabajo.bufer, envio[1])
        self._relanzar()

    def drenar(self) -> None:
        """Espera a que no quede audio por decodificar ni por emitir."""
        self._cola.join()
        self._trabajos.clear()
        self._relanzar()

    def _relanzar(self):
        if self._fallo is not None:
            fallo, self._fallo = self._fallo, None
            raise fallo


class StreamerSolapado:
    """Envuelve al streamer real para que la emision la haga el worker.

    put() aqui NO emite: le dice al decodificador que, cuando termine con ese
    bufer, se lo entregue al streamer de verdad. end() drena antes de cerrar,
    que es lo que garantiza que no se pierda ni un trozo ni se adelante el
    cierre al ultimo put().
    """

    def __init__(self, solapado: DecodificadorSolapado, real):
        self._solapado = solapado
        self.real = real

    def put(self, trozos, indices):
        self._solapado.emitir(trozos, self.real, indices)

    def end(self, indices=None):
        self._solapado.drenar()
        self.real.end(indices)

    def __getattr__(self, nombre):
        # cancelado, terminado, flujo(), trozos... el resto del mundo sigue
        # hablando con el streamer real sin enterarse de esta capa.
        return getattr(self.real, nombre)


def solapar_decodificador(modelo) -> Optional[DecodificadorSolapado]:
    """Instala el decodificador solapado. None si esta desactivado."""
    if not SOLAPAR_DECODER:
        return None
    if EN_GPU:
        # En GPU el reparto es otro (el decodificador deja de dominar) y ademas
        # habria que preasignar el bufer en el dispositivo bueno. No se ha
        # medido ahi, asi que no se activa a ciegas.
        print("[arranque] decodificador sin solapar: solo esta medido en CPU",
              flush=True)
        return None
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    if tok is None:
        return None
    solapado = DecodificadorSolapado(tok.decode)
    tok.decode = solapado.decode
    print(f"[arranque] decodificador acustico solapado "
          f"({HILOS - HILOS_DECODER} hilos el bucle, {HILOS_DECODER} el decoder)",
          flush=True)
    return solapado


def cargar_modelo():
    """Carga el modelo con el motor elegido.

    openvino: RTF 1,09 · torch: RTF 2,19 (medido, mismo hardware y texto).
    Si se pide openvino y los IR no estan, se avisa y se cae a torch en vez de
    dejar el servicio muerto: mejor lento que no responder.
    """
    if MOTOR == "openvino":
        faltan = [n for n in (IR_LM, IR_CABEZA, IR_ACUSTICO) if not Path(n).exists()]
        if faltan:
            print(
                "[aviso] faltan IR de OpenVINO (%s); se usa torch"
                % ", ".join(Path(f).name for f in faltan),
                flush=True,
            )
        else:
            sys.path.insert(0, OV_CODIGO)
            from motor import cargar as cargar_ov
            procesador, modelo = cargar_ov(
                MODELO_DIR, HILOS, IR_LM, IR_CABEZA, IR_ACUSTICO,
                # Solo se reparte si el decodificador va a correr EN PARALELO
                # con el bucle; si no, cada etapa tiene la maquina entera para
                # ella durante su turno y partir los hilos solo la frenaria.
                hilos_acustico=HILOS_DECODER if SOLAPAR_DECODER else None,
                neg_cada=NEG_CADA,
            )
            marcar_rama_condicional(modelo)
            # El freno tambien aqui: sample_speech_tokens sigue siendo la de
            # torch con este motor (solo cambian los grafos que llama), y la
            # rampa de volumen se midio en LOS DOS motores.
            frenar_guia(modelo)
            reforzar_guia_arranque(modelo)
            demorar_eos(modelo)
            instrumentar_latentes(modelo)
            modelo.set_ddpm_inference_steps(PASOS)
            _estado["solapado"] = solapar_decodificador(modelo)
            devolver_memoria()
            _estado["motor"] = "openvino"
            return procesador, modelo

    _estado["motor"] = "torch-int8"

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    procesador = VibeVoiceStreamingProcessor.from_pretrained(MODELO_DIR)
    # Se carga a CPU en ambos casos y luego se mueve. Cargar directo a la GPU
    # con device_map exige accelerate y da problemas en mps; el copiado extra
    # de ~1 GB solo cuesta un momento al arrancar, una vez.
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODELO_DIR, dtype=TIPO, device_map="cpu",
        attn_implementation="sdpa",
    )
    modelo.eval()
    soltar_encoder_acustico(modelo)
    acelerar_convoluciones_depthwise(modelo)
    cebar_decoder_acustico(modelo)
    compartir_embeddings_muertos(modelo)
    frenar_guia(modelo)
    reforzar_guia_arranque(modelo)
    demorar_eos(modelo)

    if EN_GPU:
        modelo.to(DISPOSITIVO)
        _estado["motor"] = f"torch-fp16-{DISPOSITIVO}"
    else:
        motor_q = elegir_motor_cuantizacion()
        if motor_q:
            torch.backends.quantized.engine = motor_q
        # inplace=True: sin el se duplica el modelo entero y hay OOM en 5 GB.
        #
        # En try: la cuantizacion es una OPTIMIZACION, no un requisito. Si el
        # backend no traga, es mejor arrancar en fp32 -- mas lento pero vivo --
        # que morir en el arranque. Se midio que sin esto el contenedor entraba
        # en bucle de reinicio sin llegar nunca a servir.
        try:
            torch.ao.quantization.quantize_dynamic(
                modelo, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        except RuntimeError as e:
            print(
                f"[aviso] no se pudo cuantizar a int8 con motor "
                f"'{torch.backends.quantized.engine}': {e}. Se sigue en fp32, "
                f"que va mas lento pero funciona.",
                flush=True,
            )
            _estado["motor"] = "torch-fp32"

    instrumentar_latentes(modelo)
    modelo.set_ddpm_inference_steps(PASOS)
    # Despues de cebar_decoder_acustico: asi el cebado corre DENTRO del worker
    # y no vuelve a meter el decodificador en el camino critico.
    _estado["solapado"] = solapar_decodificador(modelo)
    # Los pesos fp32 que acaban de ser sustituidos siguen ocupando hasta que
    # se recolectan Y se devuelven al sistema.
    devolver_memoria()
    return procesador, modelo


def cebar_decoder_acustico(modelo) -> None:
    """Mete un fotograma de silencio antes de cada sintesis.

    El decodificador es causal y en streaming: sus convoluciones necesitan
    contexto por la izquierda. En la primera llamada no lo tienen, y el audio
    arranca con un salto en vez de desde el silencio -- un chasquido audible
    antes de la primera silaba. Medido en la primera muestra del audio:

        sin cebar   -0,017969      <- el salto
        1 fotograma -0,000002
        3 fotogramas +0,000000

    Con uno basta y cuesta 34 ms, que se pagan una vez por sintesis y no por
    fotograma. No es un problema del sesgo de las convoluciones: se comprobo
    que decodificar un latente NULO en frio da silencio exacto (rms 0,0000).

    Se detecta la primera llamada porque la cache llega vacia; generate() crea
    una nueva en cada sintesis, asi que no hace falta avisar desde fuera.
    """
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    if tok is None or getattr(tok, "_cebado", False):
        return
    decodificar = tok.decode

    def decode_cebado(latents, cache=None, sample_indices=None, use_cache=False, **kw):
        if use_cache and cache is not None and not getattr(cache, "cache", True):
            decodificar(torch.zeros_like(latents), cache=cache,
                        sample_indices=sample_indices, use_cache=True, **kw)
        return decodificar(latents, cache=cache, sample_indices=sample_indices,
                           use_cache=use_cache, **kw)

    tok.decode = decode_cebado
    tok._cebado = True
    print("[arranque] decoder cebado con silencio (quita el chasquido inicial)",
          flush=True)


def frenar_guia(modelo, freno: float = None) -> None:
    """Reescala la guia de la difusion para que una locucion larga no se
    desboque de volumen (el "CFG rescale" de Lin et al. 2023, aplicado aqui).

    EL FALLO QUE ARREGLA, medido (2026-08-06, 8 frases encadenadas, 42 s,
    voz sp-Spk1_man, semilla 11, 6 pasos): con cfg 4.5 el RMS sube de
    -22 dB a -7 dB en los primeros 20 segundos de locucion y se queda ahi,
    con hasta un 4,8 % de muestras recortadas en el peor tramo de 5 s -- eso
    es la voz "creciendo hasta distorsionarse" que se oia en el asistente.
    No es del motor (openvino 1,85 % de recorte, torch-mps 0,30 %, la misma
    rampa en ambos) ni de las sesiones (una peticion larga por /tts/stream da
    EXACTAMENTE el mismo audio); es del modelo al encadenar contexto largo:
    los latentes que genera vuelven a entrar como contexto, con cfg > 1 la
    extrapolacion uncond + cfg*(cond - uncond) los saca un poco mas de rango
    en cada vuelta, y el bucle se realimenta hasta que el decodificador
    satura. Por frases sueltas no se ve porque cada peticion arranca de cero
    y en ~5 s la deriva no da tiempo a nada (0,00 % de recorte en 8 frases).
    Con cfg 1.5 tampoco (RMS plano en -25 dB), pero cfg bajo cuesta
    fidelidad: WER peor 85,7 % (ver PeticionTTS.cfg_scale).

    EL ARREGLO: tras extrapolar, el eps guiado se reescala para que su
    desviacion tipica vuelva a ser la de la rama condicional -- la energia
    que el modelo aprendio en entrenamiento -- y se mezcla con el original
    segun `freno` (0 = todo extrapolado, como antes; 1 = todo reescalado).
    Eso corta la realimentacion sin renunciar a la direccion de la guia.

    MEDIDO con el mismo banco (misma semilla, mismas 8 frases, cfg 4.5,
    torch-mps): freno 0 -> rampa de -22,5 a -9,3 dB y 0,30 % de recorte;
    freno 0.75 -> RMS estable en -25 +-1,5 dB los 44 s enteros, 0,00 % de
    recorte, pico 0,71, y el WER de la locucion entera pasa de 9 % a 8 %.
    En frases sueltas no empeora: WER medio 10 % frente a 11 % sin freno.
    EL DEFECTO SE QUEDA EN 0.75. Se probo 0.85 en la VM y CUESTA FIDELIDAD:
    9 clips con cfg 3.5 dieron WER medio 14,8 % y peor caso 50,0 %, frente a
    9,7 % y 11,1 % con 0.75. Aplana la rampa de energia, si, pero a un precio
    que no compensa. Queda como palanca por si alguien prefiere el volumen
    plano a la precision.

    El detalle de por que 0.75 no basta en OpenVINO: la deriva se corta del todo
    en torch (pendiente +0,03) pero en OpenVINO, donde la cabeza de
    prediccion va en int8, queda residuo -- pendiente +0,26, la norma del
    latente de 8,2 a 9,1 y +3,8 dB de RMS por locucion. Eso es lo que se oia
    como voz que se enturbia segun avanza. Con 0.85 la pendiente baja a
    -0,05 y el RMS queda plano.

    OJO CON SUBIRLO MAS: 1.0 tambien aplana la curva pero SOBREFRENA y la voz
    colapsa (WER 71,6 %). El margen util es estrecho y esta cerca.

    VIBEVOICE_FRENO_GUIA=0 lo desactiva y deja el
    comportamiento anterior bit a bit.

    Se sustituye sample_speech_tokens ENTERO en vez de envolverlo porque el
    reescalado va DENTRO del bucle de pasos, entre la extrapolacion y el
    solver: desde fuera no hay donde engancharse. Copia fiel de upstream
    (modeling_vibevoice_streaming_inference.py, sample_speech_tokens) mas
    las tres lineas del freno; si Microsoft cambia esa funcion, esto hay
    que re-copiarlo.
    """
    import types

    fi = FRENO_GUIA if freno is None else freno
    if fi <= 0:
        return

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(
            self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0],
                             self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined),
                condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            # ---- el freno: la energia del eps guiado vuelve a la de la
            # rama condicional, y se mezcla segun `fi` ----
            std_cond = cond_eps.std(dim=-1, keepdim=True)
            std_guiado = half_eps.std(dim=-1, keepdim=True)
            half_eps = (fi * (half_eps * (std_cond / (std_guiado + 1e-8)))
                        + (1.0 - fi) * half_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]

    modelo.sample_speech_tokens = types.MethodType(sample_speech_tokens, modelo)
    print(f"[arranque] freno de guia {fi} (una locucion larga ya no se "
          f"desboca de volumen)", flush=True)


# La guia reforzada del ARRANQUE. El primer sonido de una locucion se decide
# con un contexto minimo: generate() lee la primera ventana de 5 tokens y
# genera 6 latentes (0,8 s de audio) antes de leer nada mas -- se comprobo
# sintetizando el mismo arranque con dos continuaciones distintas y la primera
# muestra que diverge es EXACTAMENTE la 0,800 s. En esa ventana la difusion
# va mas suelta de la cuenta y la primera palabra sale a veces masticada:
# "El backup..." con semilla 7 se oye "El Gacop", con la 42 "Escode" -- y sin
# semilla es una loteria por peticion, que es justo "al principio no vocaliza
# bien la mayoria de las veces". Medido con 3 textos dificiles x 8 semillas
# (cfg 3.5, sp-Spk1_man): 4,2 % de WER en las TRES PRIMERAS palabras frente a
# 0,0 % en todo el resto. El fallo es del arranque, no del texto.
#
# La palanca es la guia: con cfg 4.0-4.5 esos mismos arranques salen bien
# (la difusion se cine mas al texto), pero cfg alto TODO el rato aplana la
# melodia y cuesta fidelidad (medido en PeticionTTS.cfg_scale y narrador.py).
# Asi que se refuerza SOLO el arranque: los primeros fotogramas salen con la
# guia en CFG_ARRANQUE y una rampa lineal la devuelve al cfg pedido en
# CFG_ARRANQUE_FOTOGRAMAS fotogramas. Cuesta cero latencia y cero computo
# (la guia no anade pasadas; ver PeticionTTS.cfg_scale).
CFG_ARRANQUE = float(os.environ.get("VIBEVOICE_CFG_ARRANQUE", "4.5"))
CFG_ARRANQUE_FOTOGRAMAS = int(os.environ.get("VIBEVOICE_CFG_ARRANQUE_FOTOGRAMAS", "6"))
# En que fotograma de SU arranque va la generate() que tiene el candado. Es un
# dict de modulo y no una variable local de reforzar_guia_arranque a proposito:
# una sesion que se para a esperar texto suelta el candado, y la generate() que
# entra mientras tanto lo pone a cero (generate_reiniciado). Si el contador
# viviera en la closure, la sesion reanudaria con el de la otra -- y si esa
# habia hecho menos de CFG_ARRANQUE_FOTOGRAMAS, con la rampa de 4.5 en mitad
# de una frase. Al estar aqui, foto_generacion/reponer_generacion se lo llevan
# y lo traen con el resto del estado de la sesion.
_ARRANQUE = {"frame": 0}


def reforzar_guia_arranque(modelo) -> None:
    """Envuelve sample_speech_tokens para subir la guia en los primeros
    fotogramas de CADA generate(). Se aplica DESPUES de frenar_guia y
    funciona igual con el freno apagado: envuelve lo que haya.

    El contador (_ARRANQUE) se reinicia envolviendo generate(): cada peticion
    suelta es un arranque, y en una sesion larga lo son la primera generate()
    y las que encadena el tope de caché -- que arrancan igual de frias desde
    el prefijo pristino, asi que tambien lo necesitan. Una sesion que se para
    a mitad lo fotografia antes de soltar el candado y lo repone al volver
    (SesionViva._pausar/_reanudar), asi que la generate() que se cuele en la
    pausa no le cambia en que punto de su arranque iba."""
    if CFG_ARRANQUE <= 0 or CFG_ARRANQUE_FOTOGRAMAS <= 0:
        return
    muestrear = modelo.sample_speech_tokens
    generar = modelo.generate

    def generate_reiniciado(*a, **kw):
        _ARRANQUE["frame"] = 0
        return generar(*a, **kw)

    def muestrear_reforzado(condition, neg_condition, cfg_scale=3.0):
        k = _ARRANQUE["frame"]
        _ARRANQUE["frame"] = k + 1
        if k < CFG_ARRANQUE_FOTOGRAMAS and CFG_ARRANQUE > cfg_scale:
            # Rampa lineal: fotograma 0 con CFG_ARRANQUE, y de vuelta al
            # pedido al agotar la ventana. Sin escalon: el freno de guia ya
            # normaliza la energia, pero la prosodia agradece la suavidad.
            peso = (CFG_ARRANQUE_FOTOGRAMAS - k) / CFG_ARRANQUE_FOTOGRAMAS
            cfg_scale = cfg_scale + (CFG_ARRANQUE - cfg_scale) * peso
        return muestrear(condition, neg_condition, cfg_scale)

    modelo.generate = generate_reiniciado
    modelo.sample_speech_tokens = muestrear_reforzado
    print(f"[arranque] guia reforzada al empezar: cfg {CFG_ARRANQUE} con rampa "
          f"de {CFG_ARRANQUE_FOTOGRAMAS} fotogramas (la primera palabra ya no "
          f"se mastica)", flush=True)


def demorar_eos(modelo) -> None:
    """Aplaza UN bloque el "se acabo" del clasificador cuando el modelo quiere
    parar en mitad del decaimiento de la ultima palabra y no queda ventana de
    donde sacar la cola. El porque y las medidas, en el bloque COLA INSISTIR.

    Se envuelve `forward` del clasificador y no el objeto entero porque
    tts_eos_classifier es un submodulo registrado: asignarle una funcion suelta
    lo rechaza nn.Module. Envolver el forward de la instancia lo tapa igual
    -- __call__ va a self.forward -- y deja el modulo donde estaba.

    NO SE INTENTA ADIVINAR QUE LLAMADA ES LA QUE DECIDE. Upstream consulta el
    clasificador tres veces por fotograma y solo la tercera manda; distinguirlas
    desde aqui seria atarse a un orden que no promete nadie. En vez de eso, el
    streamer ARMA el fotograma (RemateEOS.deja_pasar, que corre una vez por
    fotograma y antes que las tres llamadas) y mientras esta armado se dice que
    no hay EOS a QUIEN PREGUNTE. El presupuesto solo se gasta cuando alguna de
    esas llamadas traia de verdad un EOS: si no, no se ha aplazado nada.
    """
    if COLA_INSISTIR <= 0:
        return
    if SOLAPAR_DECODER:
        # Ver el bloque COLA INSISTIR: con el decodificador en otro hilo el
        # pico del "ultimo fotograma" puede ir por detras, y decidir con el
        # fotograma equivocado es peor que no decidir.
        print("[arranque] cola insistente desactivada: el decodificador va "
              "solapado y el pico del ultimo fotograma llegaria tarde",
              flush=True)
        return
    clasificador = getattr(modelo, "tts_eos_classifier", None)
    if clasificador is None:
        print("[aviso] no hay tts_eos_classifier: la cola insistente no se "
              "instala (revisa si upstream lo renombro)", flush=True)
        return
    real = clasificador.forward

    def forward(*a, **kw):
        logits = real(*a, **kw)
        if not _REMATE["armado"]:
            return logits
        if float(torch.sigmoid(logits.detach()).reshape(-1)[0]) > 0.5:
            # Esta si traia EOS: el presupuesto se gasta aqui y no antes.
            # OJO: `armado` NO se baja aqui. Las tres llamadas del fotograma
            # tienen que ver lo mismo, y la que decide es la ultima: bajarlo
            # en la primera dejaba pasar el EOS de verdad por la tercera. Lo
            # vuelve a calcular deja_pasar en el fotograma siguiente, y ahi ya
            # se encuentra el presupuesto gastado.
            _REMATE["aplazados"] += 1
            print(f"[sesion] EOS en el ultimo latente del bloque con el "
                  f"fotograma aun sonando (pico {_REMATE['pico']:.3f}): se "
                  f"pide un bloque mas para la cola", flush=True)
        # Muy negativo: sigmoid(-10) = 4,5e-5, o sea "no hay EOS". El bloque
        # siguiente vuelve a preguntar y ese ya pasa.
        return torch.full_like(logits, -10.0)

    clasificador.forward = forward
    print(f"[arranque] cola insistente: hasta {COLA_INSISTIR} bloque(s) de mas "
          f"si el EOS cae con la voz por encima de {COLA_FINAL_PICO} "
          f"(la ultima palabra ya no se corta a mitad)", flush=True)


# ------------------------------------------- la rama incondicional, agrupada --
# UNA pasada del backbone cada N latentes en la rama negativa del CFG.
#
# EL HALLAZGO: el bucle de Microsoft llama a tts_language_model DOS VECES por
# fotograma acustico. Una con el contexto real y otra con la secuencia
# negativa -- que arranca de un solo <|image_pad|> y NUNCA recibe texto, solo
# los latentes ya generados. Medido con GET /crono en la VM, dentro de una
# tanda real de 12 clips (684 fotogramas):
#
#   backbone (tts_lm)     1445 llamadas   45,7 ms cada una   77 % del reloj
#   cabeza de difusion    4104 llamadas    2,7 ms            13 %
#   decodificador          684 llamadas   93,6 ms      SOLAPADO, fuera del
#                                                      camino critico
#   contrapresion del decodificador: 0,02 % -> el bucle NUNCA le espera
#
# 1445/684 = 2,11 pasadas de backbone por fotograma. Una de cada dos es la
# incondicional, y cuesta lo mismo que la buena.
#
# LA APUESTA ERA que agrupar N fotogramas en UNA pasada de N tokens saldria
# casi gratis, porque el coste del modelo serian sus 156 MB de pesos y no la
# longitud de la secuencia. MEDIDO, NO SE CUMPLE: una pasada de dos tokens
# cuesta 35,3 ms frente a los 20,0 de una de uno, o sea 1,77x. Este backbone
# esta limitado por COMPUTO, no por memoria, y por eso agrupar apenas ahorra.
#
#   neg_cada   RTF     WER medio   audio generado
#      1       0,940     12,7 %      referencia
#      2       0,914     15,2 %      +7,9 % mas largo
#      3       0,911       --        +7,6 % mas largo
#
# Un 2,6 % de RTF a cambio de 2,5 puntos de WER y locuciones un 8 % mas largas
# -- la condicion negativa desfasada retrasa el fin de frase. NO COMPENSA, y
# por eso el defecto es 1. Se deja la palanca porque el diagnostico (dos
# pasadas de backbone por fotograma) vale mas que el resultado, y porque en
# una CPU con VNNI, donde el computo dejaria de mandar, la cuenta podria
# salir distinta.
#
# 1 = comportamiento original bit a bit.
NEG_CADA = int(os.environ.get("VIBEVOICE_NEG_CADA", "1"))


def marcar_rama_condicional(modelo) -> None:
    """Le pone la marca `_positivo` a la cache de la rama CONDICIONAL.

    Sin esto TtsLmOV no puede distinguir las dos ramas: a ambas les llega una
    pasada de un token con su propia cache. La marca sale del unico dato que
    las separa de verdad -- `tts_text_masks` con algun 1, o sea tokens de
    texto, que la rama negativa no ve jamas.

    La primera llamada de cada generate() con texto va ANTES que cualquier
    llamada negativa (ver el bucle de generate en
    modeling_vibevoice_streaming_inference.py), asi que para cuando la negativa
    aparece la condicional ya esta marcada y no hay ambiguedad.

    Se instala SIEMPRE, tambien con NEG_CADA=1: cuesta un `.max()` sobre un
    tensor de un elemento por pasada y es lo que permite mover el agrupado por
    peticion (PeticionTTS.neg_cada) sin reiniciar el servicio. Medir cambiando
    variables de entorno ya salio caro una vez en este proyecto -- el modulo de
    Nix las fija en el servicio y pisa las del gestor.
    """
    original = modelo.forward_tts_lm

    def forward_tts_lm_marcado(*a, **kw):
        mascara = kw.get("tts_text_masks")
        salida = original(*a, **kw)
        if mascara is not None and bool(mascara.max()):
            cache = getattr(salida, "past_key_values", None)
            if cache is not None:
                try:
                    cache._positivo = True
                except AttributeError:
                    pass            # cache de torch: el agrupado no aplica
        return salida

    modelo.forward_tts_lm = forward_tts_lm_marcado
    if NEG_CADA > 1:
        print(f"[arranque] rama incondicional agrupada: una pasada del backbone "
              f"cada {NEG_CADA} fotogramas", flush=True)


# ---------------------------------------------------- latentes a examen --
# Mirilla directa al bucle de realimentacion: el latente que devuelve
# sample_speech_tokens vuelve al modelo como contexto del paso siguiente
# (acoustic_connector), y la sospecha de "el audio se degrada segun avanza
# la locucion" apunta justo ahi. Medir el audio de rebote (RMS, WER por
# tramos) ya fallo dos veces; esto mira la causa: una fila por fotograma
# con la norma, la media y la desviacion del propio latente.
RUTA_LATENTES = os.environ.get("VIBEVOICE_LATENTES_CSV", "")


def instrumentar_latentes(modelo) -> None:
    """Si VIBEVOICE_LATENTES_CSV apunta a un fichero, cada latente acustico
    deja una fila CSV: generacion, fotograma, norma L2, media, desviacion.
    Sin la variable no se toca nada -- cero coste en produccion.

    ENVUELVE en vez de editar: sample_speech_tokens puede ser la de upstream
    O la copia con freno de frenar_guia (con VIBEVOICE_FRENO_GUIA=0 no se
    sustituye), y asi se miden las dos tal cual son. El contador de
    generaciones se lleva envolviendo generate(): cada peticion suelta es una
    generacion nueva, y una sesion larga es UNA (o varias si salta el tope de
    caché), que es exactamente la distincion que interesa comparar."""
    if not RUTA_LATENTES:
        return
    estado = {"gen": 0, "frame": 0}
    muestrear = modelo.sample_speech_tokens
    generar = modelo.generate

    def generate_contado(*a, **kw):
        estado["gen"] += 1
        estado["frame"] = 0
        return generar(*a, **kw)

    def muestrear_medido(condition, neg_condition, cfg_scale=3.0):
        lat = muestrear(condition, neg_condition, cfg_scale)
        v = lat.detach().float()
        estado["frame"] += 1
        with open(RUTA_LATENTES, "a") as f:
            f.write(f"{estado['gen']},{estado['frame']},"
                    f"{float(v.norm()):.6f},{float(v.mean()):.6f},"
                    f"{float(v.std()):.6f}\n")
        return lat

    modelo.generate = generate_contado
    modelo.sample_speech_tokens = muestrear_medido
    print(f"[arranque] latentes a {RUTA_LATENTES} (norma, media, desviacion "
          f"por fotograma)", flush=True)


class ConvDepthwiseRapida(torch.nn.Module):
    """Depthwise Conv1d reescrita como suma de K desplazamientos.

    PyTorch no trae kernel optimizado de convolucion depthwise cuando falta
    oneDNN -- que es el caso en ARM y en cualquier maquina sin MKLDNN -- y cae
    a la implementacion de referencia procesando GRUPO POR GRUPO. Con
    groups=2048 eso son 2048 convoluciones diminutas donde deberia haber una.

    Medido con el perfilador: aten::_slow_conv2d_forward se lleva el 75 % del
    tiempo del decodificador, con 22.434 llamadas por cada decode.

    Pero una depthwise no es mas que, para cada desplazamiento del kernel,
    multiplicar por un escalar por canal y sumar. Vectorizado sobre todos los
    canales a la vez:

        Conv1d depthwise (torch)   36,85 ms
        suma de 7 desplazamientos   0,35 ms      -> 106x

    Y la salida es la MISMA: diferencia maxima 4,77e-07, que es el redondeo de
    coma flotante al reordenar las sumas.

    Solo se aplica a las depthwise puras y sin dilatacion (stride 1, padding 0,
    groups == canales), que son las 26 del decodificador acustico. Cualquier
    otra forma se deja intacta.
    """

    def __init__(self, conv: torch.nn.Conv1d):
        super().__init__()
        c, k = conv.out_channels, conv.kernel_size[0]
        self.k = k
        # (K,1,C,1): asi self.w[j] ya sale con la forma que necesita el
        # broadcast, sin un view por cada paso del bucle.
        w = conv.weight.detach().reshape(c, k).t().contiguous().view(k, 1, c, 1)
        self.register_buffer("w", w)
        self.tiene_sesgo = conv.bias is not None
        if self.tiene_sesgo:
            self.register_buffer("b", conv.bias.detach().reshape(1, c, 1).clone())

    def forward(self, x):
        largo = x.shape[2] - self.k + 1
        salida = x[:, :, :largo] * self.w[0]
        for j in range(1, self.k):
            salida = salida + x[:, :, j:j + largo] * self.w[j]
        return salida + self.b if self.tiene_sesgo else salida


def acelerar_convoluciones_depthwise(modelo) -> int:
    """Sustituye las depthwise del decodificador. Devuelve cuantas cambio."""
    if os.environ.get("VIBEVOICE_SIN_DEPTHWISE_RAPIDA", "").strip() not in ("", "0"):
        return 0
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    dec = getattr(tok, "decoder", None)
    if dec is None:
        return 0
    cambiadas = 0
    for padre in dec.modules():
        for nombre, hijo in list(padre.named_children()):
            if (isinstance(hijo, torch.nn.Conv1d)
                    and hijo.groups > 1
                    and hijo.groups == hijo.in_channels == hijo.out_channels
                    and hijo.stride[0] == 1
                    and hijo.dilation[0] == 1
                    and hijo.padding[0] == 0):
                setattr(padre, nombre, ConvDepthwiseRapida(hijo))
                cambiadas += 1
    if cambiadas:
        devolver_memoria()
        print(f"[arranque] {cambiadas} convoluciones depthwise reescritas "
              f"(medido 106x mas rapido cada una)", flush=True)
    return cambiadas


def soltar_encoder_acustico(modelo) -> None:
    """Tira el codificador acustico, que en texto->voz es peso muerto.

    Son 1311 MB en fp32 -- MEDIDO -- y ademas no se cuantizan: quantize_dynamic
    solo toca nn.Linear y el tokenizador acustico es casi todo convolucion, asi
    que sobreviven enteros en memoria. Es la mayor partida residente de todo el
    servicio.

    Y no hacen falta, por dos razones independientes:

      1. NO ESTAN EN EL CHECKPOINT. transformers lo avisa al cargar: "Some
         weights ... are newly initialized: ['model.acoustic_tokenizer.
         encoder...']". Son pesos ALEATORIOS. Si el camino de sintesis los
         usara, el audio saldria a ruido.

      2. NADIE LOS LLAMA. En modeling_vibevoice_streaming_inference.py la
         unica referencia al tokenizador acustico es .decode() (linea 784).
         El encoder solo haria falta para el camino inverso -- sacar el
         prefijo de una voz a partir de un audio -- y aqui los prefijos ya
         vienen precalculados en los .pt.

    Verificado generando con el encoder eliminado: 2,40 s de audio, pico 0,464.
    Identico a con el.

    VIBEVOICE_CONSERVAR_ENCODER=1 lo deja en su sitio, por si algun dia se
    quiere calcular prefijos de voz desde audio en este mismo proceso.
    """
    if os.environ.get("VIBEVOICE_CONSERVAR_ENCODER", "").strip() not in ("", "0"):
        return
    tok = getattr(getattr(modelo, "model", None), "acoustic_tokenizer", None)
    enc = getattr(tok, "encoder", None)
    if enc is None or isinstance(enc, torch.nn.Identity):
        return
    mb = (sum(p.numel() * p.element_size() for p in enc.parameters())
          + sum(b.numel() * b.element_size() for b in enc.buffers())) / 1048576
    # Identity y no del: hay codigo que consulta el atributo aunque no lo use.
    tok.encoder = torch.nn.Identity()
    devolver_memoria()
    print(f"[arranque] codificador acustico liberado ({mb:.0f} MB)", flush=True)


def compartir_embeddings_muertos(modelo) -> None:
    """Deja de tener DOS tablas de embeddings cuando solo se usa una.

    tts_language_model trae su propia embed_tokens de 151936x896 -- 136 M
    parametros -- que NO SE USA NUNCA. No es deduccion mia, lo dice el codigo
    de Microsoft en modeling_vibevoice_streaming.py:

        # We only need the Transformer layers here. Note that embed_tokens
        # in tts_language_model is unused

    forward_tts_lm siempre recibe inputs_embeds ya calculados, y el unico
    lookup que hay usa el embedding del OTRO modelo de lenguaje.

    Duele mas de lo que parece porque quantize_dynamic solo toca nn.Linear:
    los embeddings sobreviven en fp32 enteros. Son ~520 MB de tabla muerta.

    Se APUNTA a la del otro modelo en vez de borrarla: misma forma y mismo
    vocabulario, asi que si algun camino la consultara daria exactamente lo
    mismo que el lookup bueno. Borrarla dejaria un None que explota raro.
    """
    m = getattr(modelo, "model", None)
    lm = getattr(m, "language_model", None)
    tts = getattr(m, "tts_language_model", None)
    if lm is None or tts is None:
        return
    buena = getattr(lm, "embed_tokens", None)
    muerta = getattr(tts, "embed_tokens", None)
    if buena is None or muerta is None or muerta is buena:
        return
    if buena.weight.shape != muerta.weight.shape:
        return  # otra variante del modelo: no tocar nada
    mb = muerta.weight.numel() * muerta.weight.element_size() / 1048576
    tts.embed_tokens = buena
    devolver_memoria()
    print(f"[arranque] tabla de embeddings duplicada liberada ({mb:.0f} MB)", flush=True)


def elegir_motor_cuantizacion() -> str | None:
    """Motor de int8 valido PARA ESTA CPU, o None si no hay ninguno.

    NO vale coger el primero de supported_engines. La lista llega con qnnpack
    delante incluso en x86, y qnnpack es el backend de ARM: usarlo en un
    Intel o AMD aborta al empaquetar la primera capa con

        RuntimeError: unknown architecure

    (la errata es de PyTorch, no mia). Y no avisa antes: falla en el momento
    de convertir, con el modelo ya cargado.

    En macOS pasa lo contrario y por eso hace falta elegir: qnnpack esta
    disponible pero el motor activo llega como "none", y entonces revienta con
    "Didn't find engine ... NoQEngine".
    """
    disponibles = [m for m in torch.backends.quantized.supported_engines
                   if m != "none"]
    if not disponibles:
        return None
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686"):
        preferencia = ["x86", "fbgemm", "onednn", "qnnpack"]
    else:
        preferencia = ["qnnpack", "onednn", "x86", "fbgemm"]
    for motor in preferencia:
        if motor in disponibles:
            return motor
    return disponibles[0]


def a_dispositivo(obj):
    """Lleva a la GPU una estructura anidada de tensores, ajustando el tipo.

    Hace falta porque las voces (.pt) se guardaron en bfloat16 y desde CPU: en
    GPU el modelo va en fp16 y el primer matmul aborta si los tipos no
    coinciden. (Aqui ponia fp32; se comprobo cargando un .pt y son bf16. El
    codigo hacia lo correcto igualmente, porque convierte a TIPO sea cual sea
    el de origen, pero la justificacion escrita estaba mal.)
    Los tensores enteros (mascaras, indices) se mueven pero NO se convierten:
    volverlos fp16 los corrompe.

    En CPU no se llama siquiera, asi que ese camino queda exactamente como
    estaba medido.
    """
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.to(device=DISPOSITIVO, dtype=TIPO)
        return obj.to(device=DISPOSITIVO)
    # La cache de atencion (DynamicCache) no es dict, no es secuencia y NO
    # tiene .to(). Sin tratarla aparte se quedaria entera en CPU y la primera
    # capa aborta con "Passed CPU tensor to MPS op" -- o su equivalente en
    # cuda. Se copia el objeto y se le ponen listas nuevas, para no tocar el
    # prefijo pristino que esta cacheado.
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        copia = copy.copy(obj)
        copia.key_cache = [a_dispositivo(t) for t in obj.key_cache]
        copia.value_cache = [a_dispositivo(t) for t in obj.value_cache]
        return copia
    # transformers >= 4.56 guarda el DynamicCache por capas (.layers, cada una
    # con .keys/.values) y ya sin key_cache. Los .pt oficiales de Microsoft
    # vienen del formato viejo y entran por la rama de arriba; los que fabrica
    # scripts/clonar_voz.py con la libreria de hoy entran por esta. Sin ella el
    # prefijo clonado se queda en CPU EN SILENCIO y la primera generacion en
    # GPU aborta con "Passed CPU tensor to MPS op".
    if hasattr(obj, "layers") and all(
            hasattr(c, "keys") and hasattr(c, "values") for c in obj.layers):
        copia = copy.copy(obj)
        copia.layers = []
        for capa in obj.layers:
            capa_copia = copy.copy(capa)
            capa_copia.keys = a_dispositivo(capa.keys)
            capa_copia.values = a_dispositivo(capa.values)
            copia.layers.append(capa_copia)
        return copia
    if isinstance(obj, dict):
        # copy() y no dict(): el prefijo es un BaseModelOutputWithPast y
        # generate() accede a sus atributos, no lo trata como dict pelado.
        copia = copy.copy(obj)
        for clave, valor in obj.items():
            copia[clave] = a_dispositivo(valor)
        return copia
    if isinstance(obj, (list, tuple)):
        return type(obj)(a_dispositivo(v) for v in obj)
    return obj


def prefijo_voz(nombre: str):
    """Prefijo pristino, cacheado. NUNCA se entrega tal cual: generate() lo muta,
    asi que los consumidores hacen deepcopy."""
    if nombre not in _estado["prefijos"]:
        ruta = VOCES_DIR / f"{nombre}.pt"
        if not ruta.exists():
            disponibles = sorted(p.stem for p in VOCES_DIR.glob("*.pt"))
            raise HTTPException(404, f"voz '{nombre}' no existe; hay: {disponibles}")
        # weights_only=False: el .pt guarda un BaseModelOutputWithPast, subclase
        # de OrderedDict, y viene del repo oficial de Microsoft.
        prefijo = torch.load(ruta, map_location="cpu", weights_only=False)
        if EN_GPU:
            prefijo = a_dispositivo(prefijo)
        _estado["prefijos"][nombre] = prefijo
    return _estado["prefijos"][nombre]


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    _estado["prefijos"] = {}
    ini = time.perf_counter()
    _estado["procesador"], _estado["modelo"] = cargar_modelo()

    # Calentamiento: la primera generate() de un proceso paga asignaciones
    # unicas. Mejor pagarlas al arrancar que en la primera peticion real.
    #
    # Va en try porque es una OPTIMIZACION, no un requisito, y es el punto
    # que MAS memoria pide de todo el arranque: al modelo ya cargado se le
    # suman las activaciones de una sintesis entera. En una maquina justa
    # (Docker Desktop en un Mac, por ejemplo) se muere justo aqui, tras
    # haber cargado bien, y sin calentamiento el servicio funciona
    # perfectamente: solo paga ese coste en la primera peticion real.
    #
    # VIBEVOICE_SIN_CALENTAMIENTO=1 lo salta sin intentarlo siquiera.
    if os.environ.get("VIBEVOICE_SIN_CALENTAMIENTO", "").strip() not in ("", "0"):
        print("[arranque] calentamiento omitido por configuracion", flush=True)
    else:
        try:
            _sintetizar("Hola.", VOZ_DEFECTO, 1.5, streamer=None)
        except Exception as e:
            print(
                f"[aviso] fallo el calentamiento ({type(e).__name__}: {e}). "
                f"El servicio arranca igual; la primera peticion sera mas "
                f"lenta. Si se repite, prueba VIBEVOICE_SIN_CALENTAMIENTO=1.",
                flush=True,
            )
    devolver_memoria()
    print(
        f"[arranque] modelo listo en {time.perf_counter() - ini:.1f} s"
        f" ({_rss_mb():.0f} MB residentes)",
        flush=True,
    )
    yield
    for s in list(_SESIONES.values()):
        s.cerrar()
    _SESIONES.clear()
    _estado.clear()


def _rss_mb() -> float:
    """Memoria residente en MB, o nan si no hay forma de saberlo.

    /proc/self/statm es lo preciso, pero es de Linux: en macOS no existe /proc
    y el mensaje de arranque salia con un feo "nan MB residentes".

    El repliegue es getrusage, con dos salvedades que conviene tener presentes
    al comparar numeros entre maquinas: da el PICO y no el valor actual, y
    ru_maxrss viene en KB en Linux pero en BYTES en macOS.
    """
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, IndexError, ValueError):
        pass
    try:
        import resource
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return pico / 1048576 if sys.platform == "darwin" else pico / 1024
    except Exception:
        return float("nan")


app = FastAPI(title="VibeVoice streaming", version="1.0.0", lifespan=ciclo_vida)

# La consola de voz-api vive en otro puerto, asi que sus llamadas aqui son
# de otro origen y el navegador las bloquearia. Se permite cualquier origen
# porque el servicio ya exige bearer token y no usa cookies: sin
# allow_credentials, un origen ajeno no puede robar sesion ninguna.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VOZ_CORS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["authorization", "content-type"],
    expose_headers=["X-RTF-Esperado", "X-Ritmo-Hz"],
)


class RemateEOS:
    """El trocito de estado que conserva la COLA de la locucion.

    Lo usan por composicion los dos streamers -- el de /tts/stream y el de las
    sesiones --, porque el fallo es el mismo en los dos y el arreglo tambien:
    el clasificador de EOS dispara DESPUES de emitir su fotograma y el bucle de
    6 latentes sigue dando audio que se estaba tirando. Ver el bloque COLA
    FINAL de arriba, con la medida.

    Dos reglas y ninguna mas:
      - el `end(indices)` del clasificador NO cierra: solo levanta la bandera;
        el que cierra es el `end()` sin indices que generate() hace siempre al
        salir del bucle (y, si algo revienta, el `finally` de quien sintetiza).
      - tras el EOS pasan como mucho `cuantos` fotogramas, y se para en cuanto
        uno de ellos ya es suelo de sala: ese entra -- para aterrizar en
        silencio y no en mitad del decaimiento -- y los siguientes no.
    """

    __slots__ = ("cuantos", "pico", "visto", "emitidos", "aterrizado")

    def __init__(self, cuantos: int = None, pico: float = None):
        self.cuantos = max(0, COLA_FINAL if cuantos is None else int(cuantos))
        self.pico = COLA_FINAL_PICO if pico is None else float(pico)
        self.visto = False        # el clasificador ya dijo "se acabo"
        self.emitidos = 0         # fotogramas de cola emitidos desde entonces
        self.aterrizado = False   # ya se llego a suelo de sala: no queda cola

    def retener_cierre(self, indices) -> bool:
        """True si este end() es el del clasificador y hay que aguantarlo."""
        if indices is None or self.visto or self.cuantos <= 0:
            return False
        self.visto = True
        return True

    def deja_pasar(self, trozo) -> bool:
        """True si este fotograma todavia debe emitirse. Antes del EOS, todos.

        De paso lleva la cuenta de fotogramas y apunta el pico de este en
        _REMATE, y ARMA el aplazamiento del EOS cuando toca (ver el bloque
        COLA INSISTIR): esto corre una vez por fotograma acustico y justo
        antes de las llamadas al clasificador de ese fotograma, que es lo que
        lo hace el sitio bueno para decidirlo. Cuesta un abs().max() sobre
        3200 muestras, nada al lado de los 133 ms que cuesta generarlo."""
        pico = float(trozo.detach().abs().max())
        _REMATE["pico"] = pico
        _REMATE["fotogramas"] += 1
        # Ultimo latente del bloque de 6 y la voz todavia sonando: si el modelo
        # decide parar AQUI no queda ni un fotograma de la ventana para la
        # cola. Se arma para este fotograma y solo para este.
        _REMATE["armado"] = (
            COLA_INSISTIR > 0
            and _REMATE["aplazados"] < COLA_INSISTIR
            and _REMATE["fotogramas"] % LATENTES_VENTANA == 0
            and pico >= COLA_FINAL_PICO)
        if not self.visto:
            return True
        if self.aterrizado or self.emitidos >= self.cuantos:
            return False
        self.emitidos += 1
        if pico < self.pico:
            # Suelo de sala: la palabra ya termino de apagarse. Este entra
            # (es el silencio en el que aterriza la locucion) y se cierra.
            self.aterrizado = True
        return True


# EL AIRE DE ANTES DE LA PRIMERA PALABRA: 0,33 s EN UN RELLENO DE 1,2 s
#
# El modelo no empieza a hablar en el primer fotograma: genera unos cuantos de
# suelo de sala antes del primer sonido, y hasta ahora se emitian tal cual.
# MEDIDO el 11-09-2026 en la VM (openvino), pico >= RESPIRO_PICO como umbral:
#
#   ruta                                    duracion   silencio delante
#   rellenos del asistente (/tts/stream)     1,22 s     0,33 s (max 0,58)
#   frases largas (/tts/stream)              2,84 s     0,19 s (max 0,44)
#   parrafo de 6 frases (sesion)            20,50 s     0,70 s
#
# En un relleno de una palabra eso es UN CUARTO del clip, y los rellenos existen
# precisamente para tapar la espera: el asistente los suelta para que no haya
# silencio, y llegaban con un tercio de segundo de silencio dentro. En la sesion
# son 0,70 s que se suman a la latencia hasta la primera palabra, que es la que
# se nota (el streaming entero se monto para bajarla de 23,21 s a 0,20).
#
# SE TIRAN FOTOGRAMAS ENTEROS, NO SE CORTA DENTRO DE UNO, y no es pereza:
#   - El fotograma que trae el ataque se emite COMPLETO. Cortar dentro exigiria
#     el mismo cuidado que ya costo dos intentos en el respiro (el ataque de la
#     palabra empieza 20-30 ms antes de la primera muestra fuerte), y aqui no
#     hace falta: ese fotograma dura 133 ms y el ataque va dentro.
#   - La rejilla de fotogramas se conserva. El audio que sale es EXACTAMENTE el
#     de antes menos N fotogramas de cabeza, asi que sigue valiendo la
#     reimplementacion del respiro con la que ws_fidelidad.py comprueba que el
#     servidor hace lo que dice (aplicar_respiro parte el flujo en fotogramas de
#     3200 muestras; un corte a mitad se la habria descuadrado).
#
# El umbral es el mismo RESPIRO_PICO ya medido, que separa "suelo de sala"
# (pico <= 0,0185) de "aqui dentro hay voz" (pico >= 0,0406) sin zona gris.
#
# VIBEVOICE_RECORTE_ENTRADA=0 lo apaga y devuelve el audio de antes, fotograma
# a fotograma.
RECORTE_ENTRADA = os.environ.get("VIBEVOICE_RECORTE_ENTRADA", "1") not in ("0", "no")


class RecorteEntrada:
    """Se come los fotogramas callados de ANTES de la primera palabra.

    Es la pareja de RemateEOS: uno cuida el principio de la locucion y el otro
    el final. Los dos van por composicion en los dos streamers, porque el
    problema y el arreglo son los mismos por las dos vias.

    `rescate()` es la red de seguridad: si la locucion entera resulto ser
    silencio -- un texto que no produce habla --, hay que devolver algo o el
    cliente se queda con un WAV vacio y sin saber por que.
    """

    __slots__ = ("activo", "arrancado", "tirados", "_ultimo")

    def __init__(self, activo: bool = None):
        # Con RESPIRO_PICO desactivado no hay umbral con el que decidir, asi
        # que no se recorta nada: mejor el aire de antes que comerse una
        # palabra.
        self.activo = ((RECORTE_ENTRADA if activo is None else bool(activo))
                       and RESPIRO_PICO > 0)
        self.arrancado = False
        self.tirados = 0
        self._ultimo = None

    def deja_pasar(self, trozo) -> bool:
        """False si este fotograma es aire de antes de empezar a hablar."""
        if self.arrancado or not self.activo:
            return True
        if float(trozo.detach().abs().max()) >= RESPIRO_PICO:
            self.arrancado = True
            return True
        self._ultimo = trozo
        self.tirados += 1
        return False

    def rescate(self):
        """El ultimo fotograma tirado, si NADA llego a sonar. None si sono."""
        return None if self.arrancado else self._ultimo


class StreamerCancelable:
    """Envuelve AsyncAudioStreamer anadiendo cancelacion cooperativa.

    put() corre en el HILO de generate(). Si el consumidor HTTP marco
    `cancelado`, lanzar aqui desmonta la pila de generate() y libera los 6
    nucleos al instante. Es el unico punto de corte que ofrece la API: el
    generate() del modelo no mira ningun flag externo, asi que sin esto un
    cliente que se va dejaria la CPU 20 s generando audio para nadie.
    """

    def __init__(self, cola_final: int = None, recorte_entrada: bool = None):
        from vibevoice.modular import AsyncAudioStreamer
        self.interno = AsyncAudioStreamer(batch_size=1, stop_signal=None)
        self.cancelado = False
        # generate() ya cerro el flujo por su cuenta. Lo que llegue despues
        # sobra, pero NO es que el cliente se haya ido.
        self.terminado = False
        # La cola de la ultima palabra (ver el bloque COLA FINAL). Ojo: ahora
        # `terminado` se pone unos fotogramas MAS TARDE que el EOS, que es
        # justo de lo que se trata.
        self.remate = RemateEOS(cola_final)
        # Y el aire de antes de la primera palabra, que no se emite.
        self.entrada = RecorteEntrada(recorte_entrada)

    def put(self, trozos, indices):
        # Solo aborta si el que se fue es el CLIENTE.
        #
        # Cuando el clasificador predice EOS a mitad de una ventana acustica,
        # generate() llama a end() pero NO sale del bucle de 6 latentes: sigue
        # llamando a put() con los que quedan. Para entonces el consumidor ya
        # vio el fin del flujo y su `finally` puso cancelado=True, asi que
        # abortabamos la generacion en las ultimas milesimas -- justo antes de
        # guardar el estado de la sesion, que por eso nunca se guardaba.
        #
        # Solo se salvaba el caso de que el numero de latentes fuera multiplo
        # de 6, que es 1 de cada 6. De ahi que pareciera aleatorio.
        if self.cancelado and not self.terminado:
            raise GeneracionCancelada()
        if self.terminado:
            return
        # batch_size 1: un solo trozo por llamada, y su indice es siempre 0.
        if not self.remate.deja_pasar(trozos[0]):
            return
        if not self.entrada.deja_pasar(trozos[0]):
            return
        self.interno.put(trozos, indices)

    def end(self, indices=None):
        # El EOS del clasificador no cierra: quedan por bajar los fotogramas en
        # los que se apaga la ultima palabra. Cierra el end() sin indices que
        # generate() hace al salir del bucle -- o el `finally` de _sintetizar,
        # que tambien llama sin indices.
        if self.remate.retener_cierre(indices):
            return
        self.terminado = True
        # Si NADA sono, el recorte se lo habria comido todo: se devuelve un
        # fotograma para no cerrar con un WAV vacio.
        rescate = self.entrada.rescate()
        if rescate is not None:
            self.interno.put([rescate], [0])
        self.interno.end(indices)

    def flujo(self):
        return self.interno.get_stream(0)


def _ajustar_pasos(pasos: Optional[int]) -> None:
    """Fija los pasos de difusion del modelo. Solo con el candado del modelo
    tomado: es estado GLOBAL del modelo, no un parametro de la llamada."""
    quiere = PASOS if pasos is None else pasos
    if quiere != _estado.get("pasos_ahora", PASOS):
        _estado["modelo"].set_ddpm_inference_steps(quiere)
    _estado["pasos_ahora"] = quiere


def _ajustar_neg_cada(cada: Optional[int]) -> None:
    """Agrupado de la rama incondicional, por peticion. Con el candado tomado.

    Se toca el atributo del backbone y no una variable de entorno A PROPOSITO:
    el modulo de Nix fija las variables en la unidad de systemd y pisa las que
    ponga quien lance el banco, asi que medir cambiando el entorno da cuatro
    audios identicos y una tarde perdida. Esto es un parametro de la peticion,
    y el md5 lo delata si no ha hecho efecto."""
    lm = getattr(getattr(_estado.get("modelo"), "model", None),
                 "tts_language_model", None)
    if lm is None or not hasattr(lm, "neg_cada"):
        return
    lm.neg_cada = max(1, NEG_CADA if cada is None else cada)


def _sintetizar(texto, voz, cfg_scale, streamer, semilla=None, pasos=None,
                neg_cada=None):
    """Cuerpo sincrono de la sintesis; corre en un hilo del executor."""
    procesador = _estado["procesador"]
    solapado = _estado.get("solapado")
    if solapado is not None and streamer is not None:
        streamer = StreamerSolapado(solapado, streamer)
    try:
        with _candado_modelo:
            _ajustar_pasos(pasos)
            _ajustar_neg_cada(neg_cada)
            # Antes de generar, no despues: el ruido se sortea dentro de
            # generate().
            if semilla is not None:
                torch.manual_seed(semilla)
            base = prefijo_voz(voz)
            # deepcopy DOBLE: ni el procesador ni generate() tocan el pristino.
            entradas = procesador.process_input_with_cached_prompt(
                text=texto, cached_prompt=copy.deepcopy(base),
                padding=True, return_tensors="pt", return_attention_mask=True,
            )
            if EN_GPU:
                entradas = a_dispositivo(entradas)
            reloj = time.perf_counter()
            # El estado del remate es por generate(): en que latente del bloque
            # va, como sono el ultimo fotograma y si ya se aplazo el EOS.
            remate_cero()
            with torch.no_grad():
                _estado["modelo"].generate(
                    **entradas,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=procesador.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    # El streamer ya entrega cada trozo segun sale; sin esto
                    # generate() ADEMAS acumula la sintesis entera y la
                    # concatena al final, para devolver algo que aqui se ignora.
                    return_speech=False,
                    all_prefilled_outputs=copy.deepcopy(base),
                    audio_streamer=streamer,
                )
            CRONO_TUBERIA["generate"] += time.perf_counter() - reloj
            CRONO_TUBERIA["generaciones"] += 1
    except GeneracionCancelada:
        # Se avisa: cuando esto salta por error, callarlo cuesta horas.
        print("[aviso] generacion cancelada por el cliente", flush=True)
    except Exception:
        # El futuro del executor no lo espera nadie, asi que sin esto un fallo
        # aqui desaparece sin dejar rastro y el cliente recibe silencio.
        import traceback
        print("[error] generacion fallida:", flush=True)
        traceback.print_exc()
        raise
    finally:
        # Pase lo que pase, cierra la cola: sin esto un fallo dentro de
        # generate() dejaria al consumidor esperando un trozo que no llega.
        # end() es idempotente. StreamerSolapado.end() ademas drena el worker,
        # asi que al salir de aqui no queda audio a medio decodificar.
        if streamer is not None:
            streamer.end()
        elif solapado is not None:
            # Calentamiento: no hay streamer que drene por su cuenta, pero el
            # worker si tiene trabajo encolado y la siguiente sintesis no debe
            # encontrarselo a medias.
            with contextlib.suppress(Exception):
                solapado.drenar()


# --------------------------------------------------------------------------
# SESIONES: UNA generate() VIVA por sesion, alimentada con texto segun llega
#
# EL PROBLEMA
# Sin esto, cada peticion arranca con deepcopy(pristino): el modelo empieza
# SIEMPRE desde el mismo estado acustico, asi que al narrar por frases suena
# como una lista de frases sueltas y no como alguien hablando seguido.
#
# LO QUE SE PROBO ANTES Y NO FUNCIONA: TRASPLANTAR LA CACHÉ KV
# Guardar el estado al final de una llamada y arrancar la siguiente desde ahi.
# La maquinaria era fiel -- recortar la caché a k posiciones daba exactamente el
# mismo audio que haber parado la generacion en el latente k --, pero el estado
# transportado era el equivocado. Cada llamada termina cuando el clasificador de
# EOS dice que la locucion se acabo, asi que lo que se guardaba era el estado de
# "ya he terminado de hablar". Al reanudar desde ahi con texto nuevo, el modelo
# vuelve a disparar EOS en la primera ventana: la frase sale MUDA y su texto,
# que quedo pendiente, se cuela al principio de la SIGUIENTE. De ahi
# transcripciones como "El tren llega a Manana por la tarde y vemos al parque".
#
# Medido entonces con 6 frases x 4 semillas, voz sp-Spk3_man, 6 pasos, cfg 1,5:
#
#                                   WER    no dicho   frases mudas
#   sueltas                        25,0 %    21,5 %       0 / 24
#   encadenadas (trasplante)       50,5 %    42,5 %       7 / 24
#   ... recortando la cola muda    62,5 %    56,5 %       7 / 24
#   ... recortando 6 latentes mas  54,7 %    28,0 %       1 / 18
#
# Ninguno de los dos arreglos vale: el segundo quita las frases mudas pero
# entonces REPITE el final de la anterior. Y las dos sospechas que habia estan
# descartadas por experimento: ni el last_hidden_state fabricado se lee (audio
# identico bit a bit rellenandolo de ruido), ni la rama negativa es la culpable
# (con cfg_scale=1,0 no interviene y encadenar sigue destrozando el audio).
#
# LO QUE SI FUNCIONA
# generate() YA sabe encadenar: consume `tts_text_ids` en ventanas de 5 tokens
# intercaladas con 6 latentes acusticos, y con varias frases en UNA sola llamada
# el audio sale perfecto. Nunca dispara EOS a mitad porque EL TEXTO LE LLEGA POR
# DELANTE DEL HABLA. Asi que en vez de partir la generacion en trozos, se deja
# UNA sola viva en su hilo y se le va metiendo texto por debajo.
#
# El unico obstaculo era que `tts_text_ids` es un tensor fijo. Resulta que
# generate() lo toca en tres sitios y nada mas -- .to(), .shape[1] y dos cortes
# [:, a:b] --, asi que basta con pasarle un objeto que se haga pasar por tensor
# y que RELEA su contenido en cada vuelta: TextoEnCurso. No hay que parchear ni
# una linea de Microsoft, ni reimplementar el bucle.
#
# Y como la pausa no cambia ningun calculo -- el estado se queda quieto mientras
# el hilo espera --, alimentar por trozos da EXACTAMENTE el mismo audio que
# haber pasado todo el texto de golpe. Eso no hay que creerselo, se comprueba
# por md5 (scripts/sesiones_fidelidad.py).
#
# MEDIDO ASI, con 6 frases x 6 semillas (11, 7, 3, 23, 42, 101), voz
# sp-Spk3_man, 6 pasos, cfg 1,5, transcrito con whisper.cpp. "sesion" alimenta
# frase a frase esperando a que el modelo se quede PARADO sin texto antes de
# meter la siguiente, que es el caso dificil:
#
#                        WER    peor    mudas    repiten
#   sueltas             0,8 %  16,7 %   0 / 36      0
#   junta (1 llamada)   2,8 %  50,0 %   0 / 36      0
#   sesion de golpe     2,8 %  50,0 %   0 / 36      0
#   sesion frase a frase 2,8 % 50,0 %   0 / 36      0
#
#   audio identico bit a bit a 'junta': 6/6 semillas, en los dos modos de sesion
#
# Las tres ultimas filas son la MISMA fila: no es que se parezcan, es que el
# audio es el mismo. Encadenar cuesta 2 puntos de WER frente a decir las frases
# sueltas -- el peor caso es un "Tienes" que whisper oye "quiénes" --, y ese
# coste es del modelo al encadenar, no de las sesiones: sale igual en 'junta',
# que es el camino bueno de Microsoft. A cambio no hay ni una frase muda ni una
# que repita a la anterior, que era justo lo que hundia al trasplante de caché.
VENTANA_TEXTO = 5      # TTS_TEXT_WINDOW_SIZE de modeling_..._streaming_inference
LATENTES_VENTANA = 6   # TTS_SPEECH_WINDOW_SIZE, idem

SESIONES_ACTIVAS = os.environ.get("VIBEVOICE_SESIONES", "1") not in ("0", "no")
# Tope de posiciones de la caché. Al pasarse, la locucion se cierra bien y la
# sesion sigue con una generate() nueva desde el prefijo pristino: se pierde la
# continuidad en esa costura, pero no la voz. Recortar por delante no vale,
# porque lo que hay al principio es justo el prefijo que DEFINE la voz.
# Cada ventana son 5 tokens de texto + 6 latentes = 11 posiciones ~ 0,8 s de
# audio, asi que 3000 son unos tres minutos seguidos. El limite duro del modelo
# son 8192 (decoder_config.max_position_embeddings) y ahi corta a lo bruto, a
# mitad de palabra; por eso se para antes.
TOPE_CACHE = int(os.environ.get("VIBEVOICE_TOPE_CACHE", "3000"))
CADUCIDAD_SESION = float(os.environ.get("VIBEVOICE_CADUCIDAD_SESION", "300"))
# Cuanto espera el bucle, parado, a que llegue mas texto antes de dar la
# locucion por terminada. Es el margen que tiene el LLM de arriba para producir
# la frase siguiente sin que se cierre la locucion.
ESPERA_TEXTO = float(os.environ.get("VIBEVOICE_ESPERA_TEXTO", "20"))
# Cuantas posiciones puede seguir pidiendo generate() MAS ALLA del final del
# texto sellado antes de darlo por descarrilado (EOS que no llega). El margen
# es MARGEN_EOS + MARGEN_EOS_FACTOR * (tokens sellados), y la parte
# proporcional NO es prudencia de mas: es la fisica del bucle.
#
# generate() LEE texto a ritmo fijo -- 5 tokens por ventana de 6 latentes,
# o sea 6,25 tokens por segundo de audio -- pero la voz DICE unos 4,7-5
# tokens por segundo (medido: 291 tokens en 57,3-63,2 s segun semilla). Si el
# texto entra mas deprisa de lo que se habla (un LLM rapido, o todo de golpe),
# la lectura se adelanta y al agotarse el texto quedan por DECIR unos
# 0,25-0,36 tokens de cada uno: con 291 tokens, el EOS legitimo llego con
# exceso 67-104 (6 semillas). ESO ERA LO QUE EL MARGEN FIJO DE 100 NO SABIA:
# guillotinaba locuciones legitimas a mitad de la ultima frase en cuanto el
# texto pasaba de ~300 tokens o la voz iba algo lenta -- la semilla 1 lo
# reproduce determinista, cortada en "...hasta la" con el final sin decir, y
# las transcripciones rellenan el corte con una palabra inventada ("hasta la
# proxima"), que es justo el sintoma que se achacaba al modelo. Un descarrile
# de verdad es OTRA escala: 766 posiciones de mas (2,3x el texto) y seguia
# (2026-08-05). Con margen 30 + 1,0x texto, esa locucion de 330 tokens se
# corta en el exceso 360 en vez de en el 3000 del tope, y ninguna legitima
# (maximo medido 0,36x + lookahead) se acerca al umbral.
MARGEN_EOS = int(os.environ.get("VIBEVOICE_MARGEN_EOS", "30"))
MARGEN_EOS_FACTOR = float(os.environ.get("VIBEVOICE_MARGEN_EOS_FACTOR", "1.0"))
# Desde donde se RETIENE el audio en vez de emitirlo (ver ColaAudioSesion):
# pasado este exceso, un EOS legitimo es ya poco probable (medido: llegan con
# exceso <= 0,36x el texto) y lo que se genere solo se suelta si el EOS acaba
# llegando. Por debajo NO se retiene nada: retener desde que el texto se agota
# seria estrangular el remate legitimo -- esos 0,3x tokens aun por decir son
# 10-15 s de locucion normal y el oyente los esta escuchando en directo.
RETENER_FACTOR = float(os.environ.get("VIBEVOICE_RETENER_FACTOR", "0.5"))

# Tramo FINAL de puntuacion de un trozo. Es lo que el pretokenizador de Qwen2
# puede fundir con un "\n" posterior (".\n" es un token), asi que se retiene
# hasta saber que viene detras; ver SesionViva.alimentar(). \w en vez de
# \p{L}\p{N} deja fuera el "_", que en texto hablado no aparece.
_COLA_PUNTUACION = re.compile(r"[^\s\w]+$")

# EL RESPIRO: PAUSA DE VERDAD EN CADA PUNTO, Y SOLO EN LOS PUNTOS
#
# Con los trozos cosidos con espacio el texto es fiel, pero el modelo lee los
# puntos como si fueran comas: medido (Mac, mps, sp-Spk1_man, cfg 3.5, texto de
# 10 frases, 3 semillas), la pausa tras punto es 0,34-0,36 s de media y tras
# coma 0,27-0,34 s -- LA MISMA. No hay jerarquia y la locucion suena sin aire.
# Y no es el volumen ni el tono: el RMS por sextos es plano (+-0,9 dB) y el F0
# no cae. Es solo la estructura de pausas.
#
# Se probo alargar la pausa del punto CON UN MARCADOR DE TEXTO y no hay ninguno
# que salga barato. Medido en la VM (i7-8700T, openvino, 6 frases x 3 semillas,
# sp-Spk3_man, cfg 4.5, 6 pasos), cambiando SOLO el separador entre frases:
#
#   separador        genera  callado   se oye   RTF que ve el reproductor
#   " "              14,80 s   2,09 s  14,51 s   1,043
#   "  "             15,24 s   2,09 s  14,98 s   1,108
#   " — "            14,76 s   1,91 s  14,53 s   1,108
#   "\t"             15,16 s   2,27 s  14,89 s   1,096
#   "… "             15,16 s   1,91 s  14,89 s   1,085
#   "; "             14,71 s   2,04 s  14,58 s   1,096
#   "\n"             20,27 s   6,36 s  16,22 s   1,311
#   "\n\n"           20,13 s   6,27 s  15,87 s   1,334
#
# Solo "\n" y "\n\n" pausan. Y pausan porque el modelo ejecuta ahi su parada de
# FIN DE LOCUCION -- ".\n" (token 624) es exactamente con lo que termina toda
# locucion --, de duracion loca (0,2-3,2 s) y a precio completo: un fotograma
# callado cuesta lo mismo de generar que uno de habla. El resto de separadores
# se comportan como el espacio: el modelo los ignora.
#
# ESA ERA LA VERSION ANTERIOR DE ESTE BLOQUE, Y ERA UN MAL NEGOCIO
# Cosia "\n\n" tras cada punto y luego RECORTABA el silencio sobrante. El texto
# quedaba bien y la pausa caia en su sitio, pero se pagaban 4,2 s de CPU por
# 4,2 s de silencio que acto seguido se tiraban a la basura. Medido el 7 de
# agosto en la VM, mismo texto, misma semilla, mismo motor:
#
#                                    audio que sale   pared    RTF
#   /tts/stream " ".join                   15,20 s   15,18 s  0,999
#   sesion, costura de espacio             15,20 s   15,86 s  1,044
#   sesion, "\n\n" + recorte (lo viejo)    15,61 s   22,86 s  1,464
#
# La sesion NO era mas lenta: generaba los mismos 21,9 s de audio que
# /tts/stream con "\n\n".join (22,94 s de pared, RTF 1,049) y entregaba 15,61.
# El +40 % de RTF era exactamente el audio descartado, y por eso el reproductor
# se quedaba seco y se oian microcortes.
#
# EL ARREGLO: LA PAUSA NO SE GENERA, SE INSERTA
# El modelo YA pausa en los finales de frase con costura de espacio; lo que no
# hace es pausar LO BASTANTE, ni siempre. Y se comprobo DONDE pausa: se parte
# el audio por las rachas de silencio y se transcribe cada trozo (3 semillas,
# 6 frases). NI UNA de las pausas cae dentro de una frase -- todas caen en un
# final de frase; lo que falla es que se salta algunas (3-4 de 5). Asi que
# alargar la pausa que el modelo YA hizo no puede meter aire donde no toca,
# que era el fallo historico (0,86 s tras coma cuando el "\n" iba en cada
# costura de trozo, cayera donde cayera).
#
#   1. TEXTO: costura con ESPACIO, siempre. Los ids son entonces identicos a
#      los de " ".join, asi que el HABLA es bit a bit la de /tts/stream. Eso no
#      es un parecido: es la garantia de calidad mas fuerte que hay aqui, y la
#      comprueba scripts/ws_fidelidad.py por md5.
#   2. AUDIO: cuando una racha de fotogramas callados llega a
#      RESPIRO_FOTOGRAMAS -- el modelo esta pausando --, se anaden
#      RESPIRO_ALARGA fotogramas mas del MISMO suelo de sala.
#
# EL UMBRAL DE DETECCION Y EL TOPE SON DOS NUMEROS DISTINTOS, y antes eran uno
# solo. Con "\n\n" las pausas eran de 5 a 24 fotogramas y el mismo 3 servia
# para "esto es una pausa" y para "de aqui en adelante se recorta". Con costura
# de espacio las pausas del modelo son de 2 a 6 fotogramas: un umbral de 3 no
# dispara en las frases cortas -- medido con las 3 frases de ws_fidelidad, el
# modelo pausa 2 fotogramas y no se insertaba NADA -- y un tope de 3 recorta
# aire legitimo que ya se ha pagado. Asi que:
#
#   RESPIRO_FOTOGRAMAS = 2   el modelo esta pausando: aqui se inserta el aire
#   RESPIRO_TOPE       = 8   maximo de callados que se emiten: aqui se recorta
#
# Y de paso sale una jerarquia que antes no habia: la pausa que el modelo hace
# corta (una coma, 2 fotogramas) queda en 4, y la larga (un punto, 4-6) queda
# en 6-8. El modelo decide DONDE y cuanto; esto solo suma.
#
# EL TOPE ALTO ADEMAS MEJORA LA CALIDAD. El recorte solo puede quitar audio, y
# el fotograma de la costura lleva dentro el ataque de la palabra siguiente
# (ver mas abajo): cuanto menos se recorte, menos ocasiones de comerselo. Con
# el tope en 8 no se llega a recortar en una locucion normal.
#
# EL RELLENO ES EL PROPIO FOTOGRAMA, EN ESPEJO. Repetirlo tal cual metia un
# escalon en el empalme y una periodicidad de 7,5 Hz; darle la vuelta no: el
# fotograma invertido EMPIEZA por la ultima muestra del original, asi que el
# empalme es continuo por construccion, y alternando invertido/original la
# cadena entera lo es. Medido sobre el WAV: el salto maximo entre muestras
# consecutivas es 0,1477 con alargue y 0,1477 sin el -- el mismo numero, o sea
# que no se anade ni una discontinuidad.
#
# MEDIDO (misma locucion de 15,18 s de pared, transcrita con whisper.cpp):
#   alargue   se oye    RTF     racha en el punto   WER
#     0       15,07 s  1,008    3 fot (0,40 s)      6,5 %
#     1       15,60 s  0,973    4 fot (0,53 s)      4,3 %
#     2       16,13 s  0,941    5 fot (0,67 s)      4,3 %
#     3       16,67 s  0,911    6 fot (0,80 s)      6,5 %
# (el WER se mueve por ruido de whisper: el habla es la MISMA en las cuatro).
#
# Y el aire no solo es gratis: DA MARGEN. Cada fotograma insertado son 133 ms
# que el reproductor gana para rellenar el bufer, que es justo lo que hace un
# humano al respirar entre frases. De ahi que el defecto sea 2 y no 1.
#
# LO QUE ESTO NO ARREGLA: si el modelo decide no pausar en una frontera, ahi
# no hay donde insertar. Se busco un ancla independiente y no la hay -- el
# clasificador de EOS es plano (p < 0,001) en todas las fronteras internas, y
# la posicion de LECTURA del texto no sirve porque va por delante del habla en
# una cantidad variable (~0,3x el texto). La alternativa era cortar la locucion
# en cada frase para saber donde cae la frontera, y sale igual de cara: medido,
# una generate() por frase cuesta RTF 1,172 frente a 1,060 con una sola
# (1,2 s de pared por cada corte, que es lo mismo que costaba el "\n\n").
#
# La referencia historica: 7,27 s callado con "\n" en toda costura (el fallo),
# 2,51 s cosiendo con espacio (sin aire). El punto medio es esto, y ahora es
# gratis.
#
# LO QUE NO SE PUEDE RECORTAR ES EL FOTOGRAMA ENTERO (2026-08-06)
# La primera version tiraba el fotograma completo cuando su RMS medio bajaba
# del umbral, con la idea de que el silencio y el habla no se solapan. NO ES
# CIERTO, y se midio: 29 locuciones, 6 voces en espanol, 3 semillas, 2 textos,
# 400 s de audio, fotograma a fotograma.
#
#   suelo de sala   RMS  p50 0,00073   p99 0,00503   MAX 0,00554
#   habla           RMS  MIN 0,00185   p1  0,00504   p50 0,055
#
# Es decir: el suelo LLEGA a 0,0055 y hay habla en 0,0018. Con la media de
# 133 ms las dos poblaciones se pisan y NINGUN umbral las separa (el mejor
# posible deja 13 fotogramas mal clasificados). El motivo es de bulto: el
# fotograma de la COSTURA es medio silencio y medio palabra -- el ataque de
# la primera palabra tras la pausa entra a los 70-120 ms de sus 133 --, asi
# que su media queda en 0,003-0,0045 y el recorte se llevaba el ataque
# entero. Medido en las mismas 29 locuciones: 9 de las 54 costuras (17 %)
# perdian el arranque de la palabra, y la muestra mas fuerte que se tiraba
# llegaba a |x| = 0,145 -- a -17 dBFS eso no es suelo de sala, es la voz.
# El WER no lo ve (2,5 % con recorte contra 3,0 % sin el: whisper rellena la
# consonante que falta), pero se oye como que "el audio se corta un poco".
#
# EL ARREGLO: decidir con el PICO, y recortar solo la CABEZA callada.
# La amplitud de pico del fotograma si separa las dos cosas en la costura,
# con un hueco vacio entre medias:
#
#   costuras que son silencio de verdad   pico |x| <= 0,0185
#   costuras que llevan el ataque dentro  pico |x| >= 0,0406
#
# RESPIRO_PICO = 0,03 cae en ese hueco. Un fotograma por debajo se tira
# entero, como antes; uno por encima NO se tira: se emite desde justo antes
# de su primera muestra fuerte (con RESPIRO_PRERROLLO de margen para no
# cortar la rampa del ataque) y se recorta solo el silencio de delante,
# ~84 ms de media. El contador de callados NO se rearma al rescatar: la
# pausa sigue midiendo lo mismo, que es justo lo que este bloque vino a
# arreglar. En habla floja continuada (RMS bajo el umbral pero con picos)
# la primera muestra fuerte llega antes del prerrollo, asi que el fotograma
# sale entero y no se recorta nada.
#
# MEDIDO en el mismo corpus, antes y despues:
#   recorte total        34,27 s -> 33,67 s (de 400 s: el aire no cambia)
#   pico max descartado  0,1452 -> 0,0290  (deja de tirarse voz)
#   costuras con ataque perdido  9/54 -> 0/54
#
# EL AIRE NO ERA SUELO DE SALA: ERA EL ATAQUE DE LA PALABRA SIGUIENTE, Y AL
# REVES (2026-08-15)
#
# EL SINTOMA, textual: "la respiracion entre cada frase es algo que no es muy
# realista, me da un poco de risa porque lo lee muy raro".
#
# Y no era una impresion. El fotograma que se repetia en espejo NO es suelo de
# sala: es el fotograma de la COSTURA, y este bloque ya sabia desde el 6 de
# agosto que ahi dentro esta el ataque de la palabra siguiente -- por eso el
# recorte del tope decide con el PICO y no con la media. Lo que no se hizo fue
# aplicar esa misma prueba al ALARGUE. Medido sobre la locucion de 6 frases con
# la que se vio el fallo (sp-Spk1_man, semilla 11, 3 pausas), fotograma a
# fotograma:
#
#   fotograma copiado   rms      pico     centro temporal de la energia
#   pausa 1 (fot 21)    0,00507  0,03522  0,929
#   pausa 2 (fot 42)    0,00504  0,03223  0,915
#   pausa 3 (fot 62)    0,00175  0,00739  0,352
#   suelo de sala de verdad (interior de una pausa que el modelo GENERA con
#   "\n\n")             0,00188  0,00674  ~0,5
#
# Centro temporal 0,93 quiere decir que casi toda la energia del fotograma esta
# en su ULTIMO decimo. Y la envolvente por bloques de 10 ms lo remata: los
# primeros 110 ms van entre -56 y -64 dBFS (eso si es suelo) y los ultimos 20
# suben a -47,9 y -37,3. Eso es el arranque de una palabra, no una respiracion.
# Su pico llega a 0,0352, por ENCIMA del RESPIRO_PICO = 0,03 con el que este
# mismo bloque decide "aqui dentro hay voz".
#
# Asi que lo que se emitia en cada final de frase era, literalmente:
#   1. el fotograma entero -> el arranque de la palabra, 20 ms
#   2. ese mismo fotograma INVERTIDO -> el arranque, del reves (empieza a
#      -37,7 dBFS y se apaga en 10 ms: un golpe seco que se traga)
#   3. el fotograma otra vez -> el arranque, por segunda vez
#   4. y ahora si, la palabra
# Tres golpes de energia separados 133 ms = una modulacion de 7,5 Hz. La misma
# periodicidad que el espejo venia a evitar, pero con el ataque dentro. Por eso
# "lo lee muy raro": es un tartamudeo con un chasquido al reves en medio.
#
# EL ESPEJO NO ERA EL CULPABLE, LO ERA EL MATERIAL. Sobre ruido plano el espejo
# no tiene nada que invertir; sobre una rampa de 25 dB, si. Y la comprobacion
# que daba luz verde -- "el salto maximo entre muestras no cambia" -- media
# continuidad de MUESTRA, que el espejo garantiza por construccion, y no
# continuidad de ENVOLVENTE, que es lo que oye el oido.
#
# EL ARREGLO: EL SUELO DE VERDAD, Y DELANTE DEL ATAQUE
#   1. El fotograma se parte por el PIE del ataque (_partir_pausa): la cabeza
#      callada por un lado y el ataque con su rampa por el otro.
#   2. El aire se hace SOLO con la cabeza (_aire_de_pausa), en espejo alternado
#      como siempre -- ahora sobre ruido plano, donde el espejo es inocuo.
#   3. Y se mete ENTRE las dos: cabeza, aire, ataque. La palabra ya no se parte.
#
# MEDIDO sobre la misma locucion, con el aire insertado bajo la lupa
# (scripts/deriva/pausas/): golpes = arranques de energia por encima de
# -45 dBFS dentro del aire; rango = recorrido de la envolvente dentro del aire;
# eco = cuanto suenan los 30 ms de DELANTE del aire por encima del aire (o sea,
# ataque abandonado al otro lado de la pausa); mod = modulacion a 7,5 Hz.
#
#   variante                          golpes  rango   eco    mod   salto
#   sin aire (referencia)                  -      -     -      -   0,0000
#   espejo del fotograma entero, detras    4   30,1  +6,3   3,27   0,0022
#   suelo de verdad, detras                0    8,2  +19,4  2,58   0,0236
#   suelo de verdad, delante  <- ESTO      0   11,8   -2,0  1,06   0,0015
#   silencio digital, delante              0  186,5  +6,4  16,17   0,0049
#
# El habla no cambia en ninguna: quitando los fotogramas de suelo, lo que queda
# es muestra a muestra el de /tts/stream con " ".join.
#
# EL SILENCIO DIGITAL SE DESCARTO POR MEDIDA, no por gusto: dejar 267 ms a cero
# entre frases abre un agujero en el suelo de sala (recorrido de 186 dB, y una
# modulacion de 16 dB a 7,5 Hz) que suena a puerta de ruido abriendo y cerrando.
# La grabacion de una persona en una sala no calla a cero; el modelo tampoco.
#
# VIBEVOICE_RESPIRO=0 lo apaga (ni alargue ni recorte: el audio sale bit a bit
# como el de /tts/stream con " ".join), y cada sesion puede pedirlo o
# rechazarlo con el campo `respiro`. El umbral y el tope tienen mando por si
# otra voz tiene un suelo de ruido distinto.
RESPIRO_ACTIVO = os.environ.get("VIBEVOICE_RESPIRO", "1") not in ("0", "no")
# Callados SEGUIDOS a partir de los cuales se da por hecho que el modelo esta
# pausando y se le mete el aire. 2 y no 1 porque un solo fotograma flojo lo da
# tambien una oclusiva (la /p/ de "pendientes" deja 133 ms casi mudos) y ahi
# no hay pausa ninguna: alargarlo partiria la palabra.
RESPIRO_FOTOGRAMAS = int(os.environ.get("VIBEVOICE_RESPIRO_FOTOGRAMAS", "2"))
# Fotogramas de suelo de sala que se INSERTAN ahi: el aire de la frase, que ya
# no se le pide al modelo. 0 deja solo el recorte. Cada uno son 133 ms de pausa
# y 133 ms de margen para el bufer del reproductor; ver la tabla de arriba.
RESPIRO_ALARGA = int(os.environ.get("VIBEVOICE_RESPIRO_ALARGA", "2"))
# Callados seguidos que se emiten como mucho. Es la red de seguridad contra una
# pausa desbocada -- un cliente que meta "\n\n" en su propio texto se lleva las
# de 24 fotogramas del modelo --, no el regulador de la pausa normal: con
# costura de espacio no se llega.
RESPIRO_TOPE = int(os.environ.get("VIBEVOICE_RESPIRO_TOPE", "8"))
RESPIRO_UMBRAL = float(os.environ.get("VIBEVOICE_RESPIRO_UMBRAL", "0.006"))
# Cada cuanto mira el websocket si el modelo se quedo sin texto por delante.
# Es tiempo que el modelo pasa PARADO esperando que se le pida mas; ver
# sesion_ws.vigilar() para la medida.
PERIODO_VIGIA = float(os.environ.get("VIBEVOICE_PERIODO_VIGIA", "0.01"))
# Amplitud de pico por encima de la cual un fotograma NO se tira aunque su
# media diga "callado": lleva senal dentro. Con 0 se vuelve al recorte de
# fotograma entero de la primera version (y a comerse los ataques).
RESPIRO_PICO = float(os.environ.get("VIBEVOICE_RESPIRO_PICO", "0.03"))
# Muestras que se dejan DELANTE de esa primera muestra fuerte, para no cortar
# la rampa del ataque ni meter un escalon audible en la costura. 240 = 10 ms
# a 24 kHz; ahi la senal esta todavia en el suelo, asi que el empalme no suena.
RESPIRO_PRERROLLO = int(os.environ.get("VIBEVOICE_RESPIRO_PRERROLLO", "240"))
# Lo mismo, pero para PARTIR el fotograma donde se mete el aire, que es otra
# cosa: ahi no basta con no cortar la rampa, hay que dejarla ENTERA al otro
# lado de la pausa. La primera muestra que pasa de RESPIRO_PICO es el ataque ya
# a -30 dBFS; su pie cae 20-30 ms antes. 960 = 40 ms, en medio de la meseta
# medida (ver _partir_pausa). Con 240 el arranque de la palabra se quedaba
# delante del aire y se oia dos veces.
RESPIRO_PIE = int(os.environ.get("VIBEVOICE_RESPIRO_PIE", "960"))

# LA COLA DE LA LOCUCION: EL EOS LLEGA UN FOTOGRAMA ANTES DE QUE LA VOZ CALLE
#
# EL SINTOMA: "todos los audios terminan abruptamente y la ultima palabra no se
# entiende bien como termina". Y NO era el respiro: medido con las mismas tres
# frases y dos semillas, el audio con `respiro` y sin el termina EXACTAMENTE
# igual -- los ultimos 12 fotogramas tienen el mismo RMS muestra a muestra y la
# ultima muestra fuerte cae en el mismo sitio. El respiro alarga pausas de
# ENTRE frases; en el final no toca nada, porque el recorte solo entra pasados
# RESPIRO_TOPE callados seguidos y la locucion se acaba antes.
#
# LA CAUSA ESTA EN EL BUCLE DE MICROSOFT
# (modeling_vibevoice_streaming_inference.py, lineas 770-853). Por cada ventana
# de texto se generan TTS_SPEECH_WINDOW_SIZE = 6 latentes, y el orden dentro de
# la vuelta es:
#
#     speech_latent = sample_speech_tokens(...)     # se genera el fotograma
#     audio_chunk   = acoustic_tokenizer.decode(...)
#     audio_streamer.put(audio_chunk, ...)          # se emite
#     ...
#     if tts_eos_logits[0] > 0.5:                   # <- el EOS se mira DESPUES
#         finished_tags[...] = True
#         audio_streamer.end(diffusion_indices)
#
# El clasificador de EOS mira el estado DESPUES de emitir el fotograma, y el
# `for` de los 6 latentes NO se rompe: sigue generando y llamando a put() con
# los que queden (upstream solo deja de guardarlos en `audio_chunks`). O sea
# que tras el EOS se generan, se pagan y se DECODIFICAN hasta 5 fotogramas mas
# -- 0,67 s -- que el streamer tiraba a la basura: AudioStreamer.put() los
# ignora por `finished_flags`, y ColaAudioSesion.put() por `self.cerrado`.
#
# Y en esos fotogramas esta la caida de la ultima palabra. El decodificador
# acustico es CAUSAL: las muestras del fotograma k salen de los latentes hasta
# k, asi que la cola de una consonante que se apaga a caballo de la frontera
# vive en el fotograma k+1. Tirarlo es cortar la palabra en su decaimiento, que
# es exactamente "no se entiende bien como termina".
#
# MEDIDO en la VM (openvino, 6 pasos, cfg 3,5, sp-Spk1_man, 3 frases), antes:
#
#   semilla   ultima muestra >= 0,03   cola tras ella   pico de los ultimos 5 ms
#     11              a 4,6 ms del fin      4,6 ms        0,0322  (-29,8 dBFS)
#
# Es decir: el fichero se acaba con la voz a -30 dBFS, tres veces por encima
# del suelo de sala. Eso no es un final, es un corte.
#
# EL ARREGLO: NO CERRAR EN EL EOS, SINO UNOS FOTOGRAMAS DESPUES
# El streamer se queda el EOS del clasificador (`end(indices)`) sin cerrar,
# emite hasta COLA_FINAL fotogramas mas y cierra de verdad con el `end()` sin
# indices que generate() hace siempre al salir del bucle. Cuesta CERO: esos
# fotogramas ya se generaban y se decodificaban igual, solo que se tiraban.
#
# Y no se emiten a ciegas: en cuanto uno de ellos es suelo de sala de verdad
# (pico por debajo de COLA_FINAL_PICO) se cierra ahi mismo, con ese fotograma
# dentro. Asi la locucion acaba SIEMPRE aterrizando en silencio -- la cola
# natural entera y ni un fotograma mas de relleno.
COLA_FINAL = int(os.environ.get("VIBEVOICE_COLA_FINAL", "5"))
# Pico por debajo del cual un fotograma de la cola ya es suelo de sala y no hay
# nada mas que esperar. El mismo umbral que separa "silencio" de "aqui dentro
# hay voz" en el respiro, por el mismo motivo y con la misma medida.
COLA_FINAL_PICO = float(os.environ.get("VIBEVOICE_COLA_FINAL_PICO",
                                       str(RESPIRO_PICO)))

# LO QUE LA VENTANA NO DA: CUANDO EL EOS CAE EN EL ULTIMO LATENTE
# La cola de arriba es gratis porque se aprovecha lo que la ventana de 6 ya
# genero. Pero si el EOS cae en el latente numero 6 no queda NADA que
# aprovechar, y ahi el final sigue en seco. Medido en 20 locuciones (4 textos
# x 5 semillas): 13 se llevaron su fotograma de cola y 7 no, y los 3 finales
# que seguian cortando con la voz por encima de -35 dBFS estaban entre esos 7.
# Es 1 de cada 6 por pura aritmetica de la ventana.
#
# EL ARREGLO: pedirle UN bloque mas, y solo cuando de verdad hace falta. El
# clasificador de EOS se envuelve para APLAZAR una sola vez su "se acabo", y
# solo si se dan las dos condiciones a la vez:
#
#   - el EOS cae en el ultimo latente del bloque (no queda cola que heredar), y
#   - el fotograma que se acaba de emitir todavia SUENA (pico >= COLA_FINAL_PICO):
#     el modelo quiere parar en mitad del decaimiento de una palabra.
#
# Aplazado el EOS, generate() abre otro bloque de 6 -- ya sin texto que leer --
# y de ahi sale la cola de verdad, que el streamer corta en cuanto aterriza en
# suelo de sala. Solo se aplaza UNA vez por generate(): pase lo que pase, el
# modelo se para en el bloque siguiente y el freno de MARGEN_EOS ni se entera
# (6 posiciones contra un margen de 30 + 1,0x el texto).
#
# SE MIDIO Y NO VALE: VIENE APAGADO. La idea era buena y el mecanismo funciona
# -- salta cuando tiene que saltar --, pero lo que el modelo hace con el bloque
# de mas no es siempre apagar la palabra. Banco de 20 locuciones (4 textos x 5
# semillas) con el aplazamiento puesto, frente al original:
#
#   19 de 20   igual o mejor (la cola mediana sube de 97 a 252 ms)
#    1 de 20   PEOR, y de la mala manera: 4,27 s -> 6,13 s de audio. Al no
#              dejarle parar, el modelo no remató la palabra: EMPEZO OTRA. Son
#              1,9 s de habla que el texto no pedia, y encima esa locucion
#              termino cortada a -17,9 dBFS, que es peor que el fallo original.
#
# Inventarse contenido es un precio que un arreglo de la cola no puede pagar:
# el fallo que se venia a corregir se oye mal, pero al menos dice lo que ponia.
# Asi que el defecto es 0 y esto queda como palanca para volver a medirlo
# (VIBEVOICE_COLA_INSISTIR=1) si algun dia hay forma de distinguir "esta
# apagando la palabra" de "esta arrancando otra" antes de emitirlo.
#
# Lo que SI se queda es la cola gratis de arriba (COLA_FINAL), que no puede
# inventar nada porque no le pide al modelo ni un fotograma de mas.
#
# CON EL DECODIFICADOR SOLAPADO NO SE ACTIVA NUNCA. La condicion mira el pico
# del ultimo fotograma emitido, y con VIBEVOICE_SOLAPAR_DECODER=1 ese pico lo
# rellena otro hilo que puede ir uno o dos fotogramas por detras: se estaria
# decidiendo con el fotograma equivocado. Mejor no hacer nada que hacerlo mal.
COLA_INSISTIR = int(os.environ.get("VIBEVOICE_COLA_INSISTIR", "0"))

# Lo que el streamer sabe y el clasificador de EOS necesita: como sono el
# ultimo fotograma y en que latente del bloque va. Un dict de modulo basta
# porque _candado_modelo garantiza UNA generacion a la vez; lo pone a cero
# quien arranca cada generate().
#
# QUIEN CUENTA LOS FOTOGRAMAS ES EL STREAMER, NO EL CLASIFICADOR. La primera
# version contaba llamadas al clasificador y estaba MAL: upstream lo llama
# TRES veces por fotograma -- una dentro del forward_tts_lm de la rama buena
# (linea 466, su `logits`), otra en el de la rama negativa, y la que de verdad
# decide en la linea 848 --, asi que "una llamada = un fotograma" no se
# cumple. Se vio en el banco: el aplazamiento saltaba 18 veces de 40 y el
# audio salia BIT A BIT EL MISMO, porque caia en una llamada cuyo resultado
# generate() ni mira. put() del streamer, en cambio, es exactamente uno por
# fotograma acustico y ocurre ANTES de las tres llamadas de ese fotograma.
_REMATE = {"pico": 0.0, "fotogramas": 0, "aplazados": 0, "armado": False}


def remate_cero() -> None:
    _REMATE.update(pico=0.0, fotogramas=0, aplazados=0, armado=False)


# EL ESTADO QUE UNA generate() ARRASTRA Y QUE OTRA LE PUEDE PISAR
#
# Una sesion suelta _candado_modelo mientras espera texto, y la generate() que
# entra en ese hueco -- otra sesion, o /tts/stream -- toca estado que es del
# PROCESO y no de la llamada. Esta es la lista entera, y vive en un solo sitio
# para que anadir un estado nuevo obligue a decidir si va aqui:
#
#   - el RNG global de torch: el ruido de la difusion (el unico sorteo)
#   - _ARRANQUE: en que fotograma de su arranque va la rampa de cfg
#   - _REMATE: pico del ultimo fotograma y cuenta de fotogramas del bloque
#   - los pasos de difusion: set_ddpm_inference_steps es del modelo, y
#     sample_speech_tokens lo lee en CADA latente
#   - neg_cada: atributo del backbone (solo con motor openvino)
#
# Los dos ultimos no se fotografian: los conoce la sesion (son sus ajustes) y
# se vuelven a fijar, que es mas barato y no depende de que estuvieran bien
# puestos al pausar.
#
# MEDIDO ANTES DEL ARREGLO (scripts/ws_fidelidad.py, prueba `pausa`): la
# sesion A abre con una frase corta y se queda parada ANTES de su primer
# fotograma, esperando la ventana de adelanto; la sesion B entra con pasos+4
# y otra semilla y habla entera; A reanuda. Sin esto, A salia con los pasos de
# B y sin su rampa de arranque, y B remataba con los pasos de A: ninguna de
# las dos daba el md5 de la misma sesion a solas. El RNG ya se llevaba y traia
# (era el primer estado que se descubrio, ver el bloque de _candado_modelo);
# esto generaliza aquello a todo lo demas.
def foto_generacion() -> dict:
    """Lo que hay que llevarse al soltar el candado. ANTES de soltarlo."""
    return {"rng": torch.get_rng_state(),
            "arranque": _ARRANQUE["frame"],
            "remate": dict(_REMATE)}


def reponer_generacion(foto: dict, pasos, neg_cada) -> None:
    """Deja el proceso como lo dejo la sesion. DESPUES de recuperar el candado."""
    torch.set_rng_state(foto["rng"])
    _ajustar_pasos(pasos)
    _ajustar_neg_cada(neg_cada)
    _ARRANQUE["frame"] = foto["arranque"]
    _REMATE.update(foto["remate"])


_SESIONES: dict = {}
_FIN = object()   # centinela: se acabo el audio de la sesion


class TextoEnCurso:
    """Se hace pasar por el tensor `tts_text_ids`, pero CRECE mientras generate()
    lo consume, y BLOQUEA el bucle cuando se queda sin texto por delante.

    CONTRATO CON generate() (modeling_vibevoice_streaming_inference.py). De todo
    el tensor, generate() usa exactamente esto y nada mas:

        625  tts_text_ids = tts_text_ids.to(self.device)
        672  ... if tts_text_ids.shape[1] >= TTS_TEXT_WINDOW_SIZE else ...
        727  cur  = tts_text_ids[:, i*VENTANA:(i+1)*VENTANA]
        728  next = tts_text_ids[:, (i+1)*VENTANA:(i+2)*VENTANA].shape[1]

    SI MICROSOFT CAMBIA ESO, ESTO SE ROMPE. En concreto:
      - Si materializan el tensor antes del bucle (`ids = tts_text_ids.clone()`),
        deja de releerse y las sesiones se quedan mudas tras la primera frase.
      - Si dejan de leer exactamente dos cortes por vuelta, se descuadra el
        reparto cur/lookahead y `restante()` devolveria texto ya dicho (se
        repetiria) o se comeria texto sin decir. Hay un aviso por consola.
      - Si cambian VENTANA_TEXTO, hay que cambiarlo aqui tambien.

    POR QUE NO SE PUEDE ENTREGAR UNA VENTANA A MEDIAS
    Los cortes son ABSOLUTOS sobre el buffer: la ventana i son los tokens
    [5i, 5i+5). Si en la vuelta i solo hay 3 tokens y se entregan, el bucle pasa
    a la ventana i+1 = [5i+5, 5i+10) y los tokens 5i+3 y 5i+4 que lleguen
    despues NO SE DICEN NUNCA. Por eso solo se sirve una ventana completa, y si
    no lo esta, se espera.

    Y POR QUE HAY QUE MIRAR UNA VENTANA MAS ALLA
    `next_text_window_size` no es informativo: con el se alarga por adelantado
    la mascara de atencion y el cache_position de la vuelta SIGUIENTE. Lo que se
    promete ahi hay que cumplirlo token a token. Como el buffer solo crece y las
    ventanas solo se sirven completas, la promesa se cumple sola... salvo al
    sellar. Por eso sellar CONGELA el buffer: a partir de ahi ya no entra texto,
    se sirve lo que quede -- ultima ventana corta incluida -- y el modelo cierra
    la locucion con su EOS de siempre.
    """

    def __init__(self, ids, dispositivo, espera_max=ESPERA_TEXTO,
                 al_pausar=None, al_reanudar=None,
                 posicion_inicial=0, tope=TOPE_CACHE):
        self._ids = list(ids)
        self._dispositivo = dispositivo
        self._espera_max = espera_max
        self._al_pausar = al_pausar or (lambda: None)
        self._al_reanudar = al_reanudar or (lambda: None)
        self._cond = threading.Condition()
        self._sellado = False
        self._corte = None         # tope de caché: texto que ya no cabe aqui
        self._toca_cur = True      # los cortes llegan alternados: cur, lookahead
        self.posicion_inicial = posicion_inicial
        self.tope = tope
        self.consumidos = 0        # tokens que el bucle ya ha metido en el modelo
        self.ventanas = 0
        self.esperado = 0.0        # segundos que el bucle paso quieto
        self.esperando = False     # ahora mismo, parado esperando texto
        self.sellado_por_espera = False
        self.sellado_por_tope = False
        self.descuadre = False

    # ---- lado del que alimenta (hilos de HTTP) ----
    def alimentar(self, ids) -> bool:
        """Anade texto. False si ya estaba sellado (hay que abrir otra)."""
        with self._cond:
            if self._sellado:
                return False
            self._ids.extend(ids)
            self._cond.notify_all()
            return True

    def sellar(self) -> None:
        """Se acabo el texto: que diga lo que le queda y cierre la locucion."""
        with self._cond:
            self._sellado = True
            self._cond.notify_all()

    @property
    def sellado(self) -> bool:
        with self._cond:
            return self._sellado

    def restante(self) -> list:
        """Texto que entro pero que el modelo no llego a decir.

        El tope con _disponible() importa tras un sellado por tope de caché:
        mientras el modelo remata con su EOS sigue pidiendo ventanas y
        `consumidos` avanza en vacio MAS ALLA del corte, asi que contar desde
        `consumidos` a secas se comia esos tokens -- hasta MARGEN_EOS por cada
        costura de tope -- y la generate() siguiente arrancaba sin ellos."""
        with self._cond:
            return self._ids[min(self.consumidos, self._disponible()):]

    def en_prorroga(self) -> bool:
        """Sellado, texto servido entero y ya MAS ALLA del exceso donde los
        EOS legitimos llegan (medido: <= 0,36x el texto; se retiene desde
        RETENER_FACTOR = 0,5x). El audio generado a partir de aqui es
        sospechoso: solo se suelta si el EOS acaba llegando. Es la senal con
        la que ColaAudioSesion retiene en vez de emitir.

        OJO, no es "texto agotado": entre agotar el texto y la prorroga hay
        un remate LEGITIMO de ~0,3x tokens aun por decir (el texto se lee a
        6,25 tokens/s de audio pero se habla a ~4,7-5), y ese remate debe
        seguir goteando en directo. Una vez cierta no vuelve a ser falsa:
        sellar congela el buffer y `consumidos` solo crece."""
        with self._cond:
            disponible = self._disponible()
            return (self._sellado and self.consumidos >= disponible
                    and self.consumidos - disponible
                        > MARGEN_EOS + RETENER_FACTOR * disponible)

    def _disponible(self) -> int:
        """Cuanto texto puede ver el bucle. Solo es menos que todo cuando el
        tope de caché obliga a dejar el resto para la generate() siguiente."""
        return len(self._ids) if self._corte is None else self._corte

    def posicion(self) -> int:
        """Posiciones ocupadas en la caché del tts_lm, contadas por fuera."""
        return self.posicion_inicial + self.consumidos + LATENTES_VENTANA * self.ventanas

    # ---- lado de generate() (hilo del modelo) ----
    def to(self, dispositivo):
        self._dispositivo = dispositivo
        return self

    @property
    def shape(self):
        # Solo se consulta para decidir el tamano de la PRIMERA ventana, que
        # tiene que coincidir con el primer corte. Esperar aqui a tener una
        # ventana entera es lo que garantiza que coincidan.
        self._esperar(VENTANA_TEXTO)
        with self._cond:
            return (1, self._disponible())

    def __getitem__(self, clave):
        _, corte = clave
        ini, fin = corte.start, corte.stop
        self._esperar(fin)
        with self._cond:
            trozo = self._ids[ini:min(fin, self._disponible())]
        if self._toca_cur:
            if ini != self.ventanas * VENTANA_TEXTO and not self.descuadre:
                # No es fatal, pero significa que el reparto cur/lookahead ya no
                # es el que este codigo supone. Se avisa una vez.
                self.descuadre = True
                print(f"[aviso] tts_text_ids: corte inesperado {ini}:{fin} en la "
                      f"ventana {self.ventanas}; revisa si generate() cambio de "
                      f"forma de leer el texto", flush=True)
            self.consumidos = ini + len(trozo)
            self.ventanas += 1
            with self._cond:
                # EL FRENO DE VERDAD contra un EOS que no llega. Tras sellar,
                # las lecturas mas alla del texto devuelven vacio, el modelo
                # sigue DICIENDO lo que lleva leido de adelanto -- que es un
                # remate legitimo de hasta ~0,36x el texto, ver MARGEN_EOS --
                # y el contrato es que cierre con su EOS al acabarselo. Si
                # sigue pidiendo ventanas mucho mas alla de eso, esta
                # generando parloteo sin contenido detras y no va a parar
                # solo: se desmonta desde aqui. Sellar otra vez (que es lo
                # unico que hacia el tope) no frena nada, porque las lecturas
                # YA devolvian vacio.
                #
                # Contra _disponible() y no contra len(_ids) a proposito: tras
                # un sellado por tope de caché lo servible acaba en _corte, y
                # medir contra el buffer entero dejaba al guardia ciego justo
                # ahi -- con 500 tokens pendientes para la generate() siguiente
                # habrian hecho falta 500 posiciones de descarrile antes de
                # saltar. Sin tope, _disponible() ES len(_ids) y no cambia nada.
                disponible = self._disponible()
                exceso = self.consumidos - disponible
                margen = MARGEN_EOS + MARGEN_EOS_FACTOR * disponible
                if self._sellado and exceso > margen:
                    print(f"[sesion] locucion descarrilada: {exceso} posiciones "
                          f"pedidas tras el final del texto ({disponible} "
                          f"tokens, margen {margen:.0f}) sin EOS; se corta",
                          flush=True)
                    raise LocucionDescarrilada(
                        f"el modelo agoto el texto ({disponible} tokens) "
                        f"y no cerro con su EOS tras {exceso} posiciones de mas "
                        f"(margen {margen:.0f}); se corta la locucion")
                if self.posicion() > self.tope and self._corte is None:
                    # No cabe mas en esta locucion. Se corta AQUI, en el borde
                    # de una ventana ya servida: el siguiente vistazo devuelve 0
                    # -- que es lo que se promete para la vuelta siguiente -- y
                    # el modelo cierra con su EOS. Lo que queda se dice en la
                    # generate() siguiente, con la voz intacta.
                    self._corte = self.consumidos
                    self._sellado = True
                    self.sellado_por_tope = True
                    self._cond.notify_all()
                    print(f"[sesion] tope de caché ({self.tope}) en la posicion "
                          f"{self.posicion()}: se cierra la locucion y sigue en "
                          f"otra ({len(self._ids) - self.consumidos} tokens "
                          f"pendientes)", flush=True)
        self._toca_cur = not self._toca_cur
        return torch.tensor([trozo], dtype=torch.long, device=self._dispositivo)

    def _esperar(self, hasta: int) -> None:
        with self._cond:
            if self._sellado or self._disponible() >= hasta:
                return
        # A partir de aqui el bucle se queda QUIETO. Se suelta el candado del
        # modelo para que otra peticion pueda usarlo mientras esta sesion calla:
        # una sesion esperando texto no debe secuestrar la CPU de nadie.
        marca = time.monotonic()
        self.esperando = True
        self._al_pausar()
        try:
            with self._cond:
                queda = self._espera_max
                while not self._sellado and self._disponible() < hasta and queda > 0:
                    t = time.monotonic()
                    self._cond.wait(queda)
                    queda -= time.monotonic() - t
                if not self._sellado and self._disponible() < hasta:
                    # Se acabo la paciencia: mejor cerrar bien la locucion que
                    # dejar al oyente con una frase colgada para siempre.
                    self._sellado = True
                    self.sellado_por_espera = True
        finally:
            self._al_reanudar()
            self.esperando = False
            self.esperado += time.monotonic() - marca


def _recortar_callado(trozo):
    """De un fotograma que el tope del respiro manda recortar, lo que hay que
    EMITIR. None si esta callado de verdad y se puede tirar entero.

    El fotograma dura 133 ms y el ataque de la palabra siguiente cae DENTRO de
    el, asi que la media no vale para decidir (ver el bloque RESPIRO). Se mira
    el pico: si no llega a RESPIRO_PICO es suelo de sala y se va entero; si
    llega, se devuelve desde RESPIRO_PRERROLLO muestras antes de la primera
    fuerte -- se tira solo la cabeza callada y el ataque se salva.

    Se aplana a 1-D porque a partir de aqui el trozo puede ser mas corto que
    un fotograma y lo unico que se hace con el es concatenarlo (a_pcm16 ya
    hace reshape(-1), y `numel` no cambia de sentido)."""
    plano = trozo.reshape(-1)
    if RESPIRO_PICO <= 0:
        return None
    # nonzero() y no argmax(): argmax no promete devolver la PRIMERA de varias
    # posiciones maximas, y aqui la primera es justo lo que se busca.
    fuertes = torch.nonzero(plano.abs() >= RESPIRO_PICO)
    if fuertes.numel() == 0:
        return None
    primera = int(fuertes[0])
    return plano[max(0, primera - RESPIRO_PRERROLLO):]


def _partir_pausa(trozo):
    """(cabeza callada, resto) del fotograma donde se va a meter el aire.

    La primera muestra que pasa de RESPIRO_PICO no es el principio de la
    palabra: es donde el ataque YA esta a -30 dBFS. El PIE del ataque cae
    20-30 ms antes (medido: la envolvente del fotograma de la costura sube de
    -56 a -48 a -37 dBFS en bloques de 10 ms). Cortar en la muestra fuerte deja
    esa rampa al otro lado de la pausa, y entonces se oye el arranque de la
    palabra, luego el aire, y luego la palabra otra vez.

    Asi que se corta RESPIRO_PIE muestras antes de la primera fuerte, y no
    RESPIRO_PRERROLLO. Se probo tambien un retroceso adaptativo -- ir hacia
    atras por bloques de 5 ms mientras sigan por encima del suelo del propio
    fotograma -- y da EL MISMO corte (32 ms antes en los dos fotogramas de la
    medida), asi que se queda el numero fijo: no depende de como redondee la
    coma flotante, y ws_fidelidad.py lo reproduce sin margen de duda.

    MEDIDO barriendo el retroceso (locucion de 6 frases, 3 pausas), con el eco
    = cuanto suenan los 30 ms de DELANTE del aire por encima del aire:

        pie      eco     golpes en el aire   recorrido de la envolvente
        10 ms   +3,8 dB        3                  23,0 dB
        20 ms   -1,6 dB        0                  11,8 dB
        40 ms   -2,2 dB        0                  11,8 dB
        60 ms   -2,0 dB        0                  11,9 dB

    A partir de 20 ms el ataque queda entero al otro lado y la meseta es plana;
    40 ms cae en medio de ella.

    Sin ataque dentro, resto sale vacio y el aire va detras del fotograma
    entero, que es donde iba siempre."""
    plano = trozo.reshape(-1)
    if RESPIRO_PICO <= 0:
        return plano, plano[:0]
    fuertes = torch.nonzero(plano.abs() >= RESPIRO_PICO)
    if fuertes.numel() == 0:
        return plano, plano[:0]
    corte = max(0, int(fuertes[0]) - RESPIRO_PIE)
    return plano[:corte], plano[corte:]


def _aire_de_pausa(suelo, muestras: int) -> list:
    """`muestras` de aire hechas con `suelo`, en espejo alternado.

    El espejo es el mismo truco de siempre -- la copia invertida EMPIEZA por la
    ultima muestra del original, asi que cada junta es continua por
    construccion --, pero ahora se aplica al SUELO DE SALA y no al fotograma
    entero. Sobre ruido plano el espejo no tiene nada que invertir; sobre un
    ataque de palabra si, y eso es lo que sonaba al reves.

    El ultimo trozo se recorta para dar exactamente `muestras` y no romper la
    cuenta de fotogramas: el corte cae en ruido de -55 dBFS, y el salto que
    deja se midio en 0,0015 (el habla llega a 0,20).

    Se devuelven copias y no vistas: el suelo puede venir del trozo que ya va
    camino de la cola, y nadie debe compartir memoria con lo ya emitido."""
    plano = suelo.reshape(-1)
    if muestras <= 0 or plano.numel() == 0:
        return []
    fuera, puestas, k = [], 0, 0
    while puestas < muestras:
        pieza = (torch.flip(plano, [0]) if k % 2 == 0 else plano.clone())
        if puestas + pieza.numel() > muestras:
            pieza = pieza[:muestras - puestas].clone()
        fuera.append(pieza)
        puestas += pieza.numel()
        k += 1
    return fuera


class ColaAudioSesion:
    """El `audio_streamer` que espera generate(), volcado a una cola asincrona.

    No cierra la cola de la sesion al terminar: una sesion larga puede encadenar
    varias generate() (al llegar al tope de caché) sobre el MISMO flujo de audio.

    LA PRORROGA SE RETIENE, Y ES EL RECORTE DE VERDAD
    El guardia de MARGEN_EOS corta un descarrile, pero esto es streaming: para
    cuando salta, lo emitido ya no se puede desenviar. Asi
    que el audio generado en la PRORROGA (TextoEnCurso.en_prorroga: pasado el
    exceso donde los EOS legitimos llegan) no se emite: se retiene aqui.

      - EOS limpio: generate() llama a end() y lo retenido se suelta entero,
        en orden. No se pierde ni un trozo; el precio es que ese ultimo tramo
        llega de golpe al final en vez de gotear.
      - Descarrile: LocucionDescarrilada salta desde la LECTURA de texto y
        desmonta la pila de generate() SIN pasar por end() -- upstream no lo
        llama en ningun finally, se comprobo --, asi que lo retenido muere
        con este objeto y el oyente no lo oye.

    Y el coste durante la locucion es CERO a proposito: mientras quede texto
    por delante, put() emite exactamente igual que antes, y el remate
    legitimo tras agotarse el texto -- ~0,3x tokens aun por decir, 10-15 s en
    una locucion larga alimentada deprisa -- sigue goteando en directo, que
    para eso el oyente lo esta escuchando. En una locucion normal la
    prorroga NI EMPIEZA: el EOS llega antes (exceso medido <= 0,36x frente al
    umbral de 0,5x) y no se retiene ni un trozo. Retener desde que el texto
    se agota, que fue el primer diseno, estrangulaba justo ese remate: 10-15 s
    de silencio en mitad de la escucha y el final a chorro.

    Lo que NO promete: en un descarrile de verdad, el parloteo generado ENTRE
    el final del contenido real y el umbral de prorroga (del orden de 15 s en
    un texto de 300 tokens) si llega al oyente; lo que se corta es el resto,
    que con el margen antiguo eran minutos. Para afinar mas haria falta saber
    POR DONDE VA hablando el modelo, y eso el bucle de generate() no lo
    cuenta: solo expone el EOS, que es justo lo que falla en un descarrile.
    """

    def __init__(self, lazo, cola, texto=None, respiro=False, cola_final=None,
                 recorte_entrada=None):
        self.lazo, self.cola = lazo, cola
        # El TextoEnCurso de ESTA generate(): la fuente de la senal de
        # prorroga. Sin el (None) no se retiene nunca, put() como siempre.
        self.texto = texto
        self.cerrado = False
        self.trozos = 0
        self.retenidos = []   # el audio de la prorroga, a la espera del EOS
        # Lo pone SesionViva.abortar() cuando el cliente se larga. Por la via
        # HTTP nadie lo toca nunca, asi que ahi el comportamiento no cambia.
        self.cancelado = False
        # El aire de la frase (ver el bloque RESPIRO): al llegar a
        # RESPIRO_FOTOGRAMAS callados se alarga la pausa con suelo de sala
        # propio, y pasado RESPIRO_TOPE se recorta lo que el modelo siga
        # callando. Los contadores arrancan de cero en cada generate() porque
        # _generar() monta una cola nueva: en la costura de un tope de caché
        # eso da un poco de aire de mas al empezar, que es el lado prudente --
        # nunca se come audio.
        self.respiro = respiro
        self._callado_seguido = 0
        self._sonado = False      # ya salio algun fotograma con voz dentro
        # Ultimo fotograma que era suelo de sala ENTERO. Es el material de
        # repuesto para el aire cuando el fotograma de la costura casi no tiene
        # cabeza callada; ver put().
        self._suelo = None
        # La cola de la ultima palabra (ver el bloque COLA FINAL). Aqui vive el
        # arreglo del "final en seco": el EOS del clasificador ya no cierra la
        # cola, la cierra el end() sin indices del final de generate().
        self.remate = RemateEOS(cola_final)
        # Y su pareja al principio: los fotogramas callados de antes de la
        # primera palabra no se emiten (0,70 s medidos en un parrafo de 6
        # frases). Ver RecorteEntrada.
        self.entrada = RecorteEntrada(recorte_entrada)

    def put(self, trozos, indices):
        # Tras el cierre de verdad lo que llegue sobra. OJO: el cierre ya NO es
        # el EOS del clasificador -- generate() sigue dando hasta 5 fotogramas
        # mas del bucle de 6 latentes, y ahi esta el decaimiento de la ultima
        # palabra; los deja pasar `remate` (bloque COLA FINAL).
        if self.cerrado:
            return
        # UNICO punto de corte que ofrece generate(): no mira ningun flag
        # externo, asi que abortar es lanzar desde aqui y dejar que la excepcion
        # desmonte su pila. Es lo mismo que hace StreamerCancelable en
        # /tts/stream, y por eso el corte tarda como mucho lo que dure un trozo
        # (~133 ms de audio) en notarse.
        if self.cancelado:
            raise GeneracionCancelada()
        for i, idx in enumerate(indices):
            if int(idx) != 0:
                continue
            if not self.remate.deja_pasar(trozos[i]):
                continue
            if not self.entrada.deja_pasar(trozos[i]):
                continue
            trozo = trozos[i].detach().float().cpu()
            if not self.respiro:
                self._emitir(trozo)
                continue
            # Un fotograma "callado" es el suelo de sala de una pausa del
            # modelo, y una pausa del modelo es SIEMPRE un final de frase
            # (comprobado transcribiendo los trozos entre pausa y pausa; ver
            # el bloque RESPIRO). Asi que aqui se hacen las dos mitades del
            # respiro: en cuanto se confirma la pausa se ALARGA con suelo de
            # sala propio -- el aire, gratis --, y pasado RESPIRO_TOPE se
            # RECORTA lo que el modelo siga callando.
            if float(trozo.pow(2).mean().sqrt()) >= RESPIRO_UMBRAL:
                self._callado_seguido = 0
                self._sonado = True
                self._emitir(trozo)
                continue
            self._callado_seguido += 1
            if self._callado_seguido > RESPIRO_TOPE:
                # El recorte NO es a ciegas: el fotograma de la costura lleva
                # dentro el ataque de la palabra siguiente y tirarlo entero se
                # lo comia. Solo se tira lo que de verdad esta callado; lo que
                # lleva senal se emite desde justo antes de ella.
                trozo = _recortar_callado(trozo)
                if trozo is not None:
                    self._emitir(trozo)
                continue
            # El aire va ENTRE frases: ni delante de la primera ni detras de la
            # ultima. Delante solo retrasaria el primer sonido, que es la
            # latencia que mas se nota (de ahi `_sonado`); detras -- ya con el
            # EOS visto, o sea dentro de la cola final -- son 267 ms de silencio
            # pegados al final que nadie oye como pausa y que solo retrasan el
            # turno del que escucha. Medido: sin esta guarda, una locucion corta
            # se llevaba 400 ms de cola (1 fotograma de aterrizaje + 2 de aire
            # insertado) donde bastan 133.
            if (self._callado_seguido != RESPIRO_FOTOGRAMAS or not self._sonado
                    or self.remate.visto or RESPIRO_ALARGA <= 0):
                self._emitir(trozo)
                # Suelo de sala de repuesto para el aire: solo vale el
                # fotograma que esta callado ENTERO (pico por debajo del
                # umbral), no el de la costura. En una pausa siempre acaba de
                # pasar uno, porque el aire se mete en el SEGUNDO callado.
                if float(trozo.abs().max()) < RESPIRO_PICO:
                    self._suelo = trozo.reshape(-1).clone()
                continue
            # AQUI VA EL AIRE, y va DENTRO del fotograma, no detras.
            #
            # Este fotograma es el de la costura: sus primeros 110 ms son suelo
            # de sala y los ultimos 20 son el arranque de la palabra siguiente
            # (medido; ver el bloque RESPIRO). Meter el aire detras del
            # fotograma entero dejaba ese arranque al otro lado de la pausa --
            # se oia el principio de la palabra, luego 267 ms de pausa, y luego
            # la palabra otra vez -- y ademas lo repetia, porque el material que
            # se copiaba era ESE fotograma.
            #
            # Asi que se parte por el pie del ataque y se emite cabeza, aire,
            # ataque. El aire se hace solo con la cabeza, que es suelo de sala
            # de verdad; si no queda cabeza suficiente (un ataque que empieza
            # muy pronto) se usa el ultimo fotograma que era suelo entero, que
            # en una pausa siempre acaba de pasar.
            cabeza, ataque = _partir_pausa(trozo)
            fuente = cabeza if cabeza.numel() >= trozo.numel() // 4 else self._suelo
            if fuente is None or fuente.numel() == 0:
                fuente = trozo.reshape(-1)
            if cabeza.numel():
                self._emitir(cabeza)
            # El aire mide RESPIRO_ALARGA fotogramas EXACTOS, se haga con lo que
            # se haga: asi la pausa sigue creciendo lo que se anuncia y la
            # cuenta de fotogramas del flujo no se descuadra.
            for extra in _aire_de_pausa(fuente, RESPIRO_ALARGA * trozo.numel()):
                self._emitir(extra)
            if ataque.numel():
                self._emitir(ataque)

    def _emitir(self, trozo) -> None:
        """Un fotograma a la cola -- o a la reserva, si la locucion ya esta en
        prorroga y hay que esperar al EOS para soltarlo."""
        self.trozos += 1
        if self.texto is not None and self.texto.en_prorroga():
            self.retenidos.append(trozo)
        else:
            self.lazo.call_soon_threadsafe(self.cola.put_nowait, trozo)

    def end(self, indices=None):
        # Aqui SOLO se llega con un cierre legitimo (el EOS del modelo, o el
        # fin del bucle de generate()): lo retenido era el remate de verdad y
        # se suelta entero. Mismo hilo y misma via que put(), asi que el orden
        # con lo ya emitido se conserva. Un descarrile no pasa por aqui.
        #
        # El EOS del clasificador (el unico end() que llega CON indices) no
        # cierra: se aguanta para que baje la cola de la ultima palabra, y
        # cierra el end() sin indices con el que generate() sale del bucle.
        if self.remate.retener_cierre(indices):
            return
        # Si NADA sono en toda la generate(), el recorte de entrada se lo comio
        # todo: se devuelve un fotograma para no cerrar en vacio.
        rescate = self.entrada.rescate()
        if rescate is not None:
            self._emitir(rescate.detach().float().cpu())
        self.cerrado = True
        if self.retenidos:
            # Que quede en el log: un EOS que llego DESPUES del umbral de
            # prorroga es raro (los medidos llegan antes) y merece verse.
            print(f"[sesion] EOS en plena prorroga: se sueltan "
                  f"{len(self.retenidos)} trozos retenidos "
                  f"(~{self.segundos_retenidos():.1f} s)", flush=True)
        for trozo in self.retenidos:
            self.lazo.call_soon_threadsafe(self.cola.put_nowait, trozo)
        self.retenidos = []

    def segundos_retenidos(self) -> float:
        return sum(t.numel() for t in self.retenidos) / RITMO


class SesionViva:
    """Una generate() viva en su hilo, con una cola de texto por delante."""

    def __init__(self, nombre, voz, cfg_scale, semilla, pasos, lazo,
                 respiro=True, cola_final=None, neg_cada=None,
                 recorte_entrada=None):
        self.nombre = nombre
        self.voz = voz
        self.cfg_scale = cfg_scale
        self.semilla = semilla
        self.pasos = pasos
        # Agrupado de la rama incondicional (ver NEG_CADA). None = el del
        # servicio. Se fija al arrancar CADA generate() y se vuelve a fijar
        # al reanudar tras una pausa: antes las sesiones no lo tocaban nunca
        # y heredaban el que hubiera dejado el ultimo /tts/stream.
        self.neg_cada = neg_cada
        # Aire de entrada: como la voz y el respiro, solo se mira al crearla.
        self.recorte_entrada = recorte_entrada
        # El respiro (pausa de verdad en cada punto; ver el bloque RESPIRO) es
        # por sesion Y por servicio: el campo `respiro` de la peticion manda,
        # pero VIBEVOICE_RESPIRO=0 lo apaga globalmente.
        self.respiro = bool(respiro) and RESPIRO_ACTIVO
        # Fotogramas de cola tras el EOS (bloque COLA FINAL). None = el defecto
        # del servicio; va por sesion para poder medir el antes y el despues
        # sin reiniciar nada, que es como se midio.
        self.cola_final = cola_final
        self.lazo = lazo
        self.cola = asyncio.Queue()
        self.visto = time.time()
        self.cerrada = False
        self.terminada = False
        self.escuchando = False
        self.generaciones = 0
        self.error = None
        self.eos_temprano = 0     # veces que el modelo callo con texto pendiente
        self.abortada = False     # el cliente se fue: cortar sin miramientos
        self._hablado = False     # ya entro texto: los siguientes llevan costura
        self._cola_punt = ""      # puntuacion final retenida; ver alimentar()
        self._pendiente = []
        self._alimentador = None
        self._audio = None        # el ColaAudioSesion de la generate() en curso
        self._arrancado = False
        self._entre_locuciones = False   # parado, pero no dentro de generate()
        self._foto = None         # foto_generacion() mientras esta parada
        self._cond = threading.Condition()
        self._hilo = threading.Thread(target=self._correr, daemon=True,
                                      name=f"sesion-{nombre}")

    # ---- API ----
    def arrancar(self):
        self._arrancado = True
        self._hilo.start()

    def alimentar(self, texto: str) -> int:
        """Encola texto. Va al alimentador vivo si lo hay; si no, a la reserva
        para la generate() siguiente."""
        # EL SEPARADOR ENTRE TROZOS ES PROSODIA, no un detalle de formato.
        # Antes cada trozo se tokenizaba como texto.strip() + "\n", que es lo
        # que hace el procesador con una peticion suelta. Pero el modelo trata
        # el salto de linea como frontera de PARRAFO y mete una pausa larga y
        # ademas ERRATICA. Medido (Mac, torch-mps, misma semilla 11, 6 trozos
        # de un texto seguido, silencio = tramos bajo el 2 % del pico):
        #
        #   "\n" en cada costura      30,9 s · 7,3 s callado · pausas de
        #                             0,9 s (tras una COMA), 2,4 s y 1,6 s
        #   "\n" solo tras .!?        31,7 s · 5,9 s callado · aun una de 2,7 s
        #   espacio en toda costura   28,1 s · 2,5 s callado · la mas larga
        #                             0,37 s, y el WER identico (10 %)
        #
        # La pausa de 0,9 s tras la coma era el caso mas grave: el troceador
        # de arriba (scripts/narrador.py) corta por comas o por espacios para
        # arrancar pronto, y el "\n" acababa incrustado en mitad de una frase
        # del LLM. Asi que la BASE es coser los trozos con espacio -- el LLM
        # separa sus frases con ". ", no con saltos de linea -- y el unico
        # "\n" garantizado es el del final de la locucion, que lo pone
        # cerrar() igual que el procesador en una peticion suelta.
        #
        # Y el espacio se queda EN TODA costura, tambien con `respiro`. Hubo
        # una version (6 de agosto) que ponia "\n\n" tras cada punto para que
        # el modelo pausara de verdad; funcionaba, pero le costaba al modelo
        # generar 4,2 s de silencio que el tope de ColaAudioSesion tiraba acto
        # seguido -- RTF 1,46 en sesion frente a 1,04 con espacio, y el
        # reproductor seco. El aire ahora se INSERTA en el audio y no se le
        # pide al modelo; las medidas, en el bloque RESPIRO de arriba. Un
        # cliente que QUIERA una pausa de parrafo puede seguir mandando el
        # "\n" dentro de su propio texto: ese no se toca.
        #
        # El espacio va como PREFIJO del trozo siguiente, no como sufijo del
        # anterior, porque el BPE funde " y" en un token: sufijo daria un
        # token de espacio suelto y OTRO texto. VERIFICADO con el tokenizador:
        # encode(trozo) + encode(" resto") == encode("trozo resto") token a
        # token, asi que alimentar por trozos sigue dando EXACTAMENTE los
        # mismos ids -- y por tanto el mismo audio bit a bit -- que el texto
        # continuo equivalente en una sola llamada (se comprueba por md5 en
        # scripts/ws_fidelidad.py, ahora contra " ".join).
        #
        # LA PUNTUACION FINAL SE RETIENE hasta saber que viene detras. El
        # pretokenizador de Qwen2 deja que un tramo de puntuacion absorba los
        # saltos de linea que le sigan (" ?[^\s\p{L}\p{N}]+[\r\n]*"), asi que
        # "texto." + "\n" del cierre tokeniza DISTINTO segun se codifique
        # junto (un token ".\n") o por separado ("." y "\n") -- se midio: era
        # el unico token de 120 que divergia de la peticion unica, y con el
        # su audio. Retener el tramo final de puntuacion y soltarlo pegado a
        # lo siguiente (el trozo que viene, o el "\n" del cierre) restaura la
        # igualdad exacta. Cortar delante de la puntuacion es seguro: letras
        # y digitos no absorben nada, encode("texto")+encode(".") ==
        # encode("texto."). El retardo es de un token y el modelo de todas
        # formas no habla hasta tener dos ventanas por delante.
        self.visto = time.time()
        with self._cond:
            if self.cerrada:
                raise HTTPException(409, f"sesion '{self.nombre}' ya cerrada")
            # Sin tocar la puntuacion: `respiro` ya no cambia NI UN TOKEN, solo
            # el audio (ver el bloque RESPIRO). Por eso una sesion con respiro
            # y otra sin el generan el mismo habla, y las dos la misma que
            # /tts/stream con " ".join.
            pieza = (self._cola_punt + (" " if self._hablado else "")
                     + texto.strip())
            m = _COLA_PUNTUACION.search(pieza)
            self._cola_punt = m.group(0) if m else ""
            if m:
                pieza = pieza[:m.start()]
            ids = _estado["procesador"].tokenizer.encode(
                pieza, add_special_tokens=False)
            self._hablado = True
            al = self._alimentador
            if al is not None and al.alimentar(ids):
                return len(ids)
            self._pendiente.extend(ids)
            self._cond.notify_all()
        return len(ids)

    def cerrar(self) -> None:
        """Termina la locucion limpiamente: el modelo dice lo que le queda y
        cierra con su EOS."""
        with self._cond:
            if self.cerrada:
                return
            # El texto sellado termina en "\n", igual que el de una peticion
            # normal (text.strip() + "\n"): es la unica frontera de parrafo
            # legitima -- el final -- y con ella el EOS sale como siempre. Se
            # codifica PEGADO a la puntuacion retenida (".\n" es UN token para
            # el BPE); ver el bloque de alimentar().
            if self._hablado or self._cola_punt:
                ids = _estado["procesador"].tokenizer.encode(
                    self._cola_punt + "\n", add_special_tokens=False)
                self._cola_punt = ""
                al = self._alimentador
                if al is None or not al.alimentar(ids):
                    self._pendiente.extend(ids)
            self.cerrada = True
            al = self._alimentador
            self._cond.notify_all()
        if al is not None:
            al.sellar()

    def abortar(self) -> None:
        """Corta YA la generacion en curso: el cliente se fue.

        cerrar() es lo educado -- el modelo dice lo que le queda y cierra con su
        EOS --, y es lo correcto por HTTP, donde el POST /fin lo manda alguien
        que sigue escuchando. Pero cuando lo que se cae es el websocket no queda
        nadie al otro lado: seguir seria medio minuto de CPU al 100 % con el
        candado del modelo tomado, generando audio para el vacio.

        Se tira de los dos hilos a la vez porque el bucle puede estar en
        cualquiera de los dos sitios: se sella el texto (por si esta parado
        esperando mas, dentro de TextoEnCurso._esperar) y se marca la cola de
        audio (por si esta dentro de generate(), donde put() es el unico punto
        por el que se puede desmontar la pila).
        """
        with self._cond:
            self.cerrada = True
            self.abortada = True
            self._pendiente = []
            al, audio = self._alimentador, self._audio
            self._cond.notify_all()
        if audio is not None:
            audio.cancelado = True
        if al is not None:
            al.sellar()

    def esperar_fin(self, segundos: float) -> bool:
        """Espera a que muera el hilo. False si sigue vivo al agotarse el plazo.

        Hace falta esperarlo de verdad: mientras viva tiene tomado
        _candado_modelo, y devolver el control al cliente antes de eso dejaria
        la siguiente peticion bloqueada contra un hilo fantasma.
        """
        if not self._arrancado:
            return True
        self._hilo.join(segundos)
        return not self._hilo.is_alive()

    def estado(self) -> dict:
        al = self._alimentador
        return {
            "sesion": self.nombre, "voz": self.voz,
            "viva": not self.terminada, "cerrada": self.cerrada,
            "abortada": self.abortada,
            "escuchando": self.escuchando, "generaciones": self.generaciones,
            "pendientes": len(self._pendiente) + (len(al.restante()) if al else 0),
            "posicion": al.posicion() if al else 0,
            # Lo que mira el cliente para saber si puede mandar la frase
            # siguiente sin que se le cuele un silencio: el modelo esta parado
            # porque se ha quedado sin texto por delante.
            "esperando": bool(al and al.esperando) or self._entre_locuciones,
            "esperado_s": round(al.esperado, 2) if al else 0.0,
            "eos_temprano": self.eos_temprano,
            "error": self.error,
        }

    # ---- hilo ----
    def _correr(self):
        esperar_mas = False
        try:
            while True:
                with self._cond:
                    # Tras un corte por tope de caché la sesion SIGUE viva: el
                    # modelo se ha puesto al dia con el texto, no es que se haya
                    # acabado. Sin esta espera, la sesion moria justo aqui -- y
                    # como el corte pasa cuando ya no queda nada pendiente, moria
                    # SIEMPRE que se llegaba al tope, dejando al oyente colgado.
                    fin_espera = time.monotonic() + ESPERA_TEXTO
                    self._entre_locuciones = esperar_mas
                    while (esperar_mas and not self._pendiente
                           and not self.cerrada
                           and time.monotonic() < fin_espera):
                        self._cond.wait(fin_espera - time.monotonic())
                    self._entre_locuciones = False
                    if not self._pendiente:
                        break
                    ids, self._pendiente = self._pendiente, []
                    al = TextoEnCurso(
                        ids, DISPOSITIVO, ESPERA_TEXTO,
                        al_pausar=self._pausar,
                        al_reanudar=self._reanudar,
                        tope=TOPE_CACHE,
                    )
                    if self.cerrada:
                        al.sellar()
                    self._alimentador = al
                self._generar(al)
                with self._cond:
                    # Lo que el modelo no llego a decir vuelve a la reserva y
                    # abre la generate() siguiente.
                    self._pendiente = al.restante() + self._pendiente
                    self._alimentador = None
                    if al.restante() and not al.sellado_por_tope:
                        # El modelo cerro la locucion teniendo texto sin decir:
                        # es justo el fallo que este diseno viene a evitar.
                        self.eos_temprano += 1
                        print(f"[aviso] sesion {self.nombre}: EOS con "
                              f"{len(al.restante())} tokens sin decir", flush=True)
                    if not al.consumidos:
                        # No dijo NADA: seguir seria un bucle infinito diciendo
                        # nada. Mejor terminar y que se vea.
                        break
                    # Cerrada pero con texto sin decir (solo pasa tras un EOS
                    # prematuro): se abre otra y se dice, en vez de tragarselo.
                    if self.cerrada and not self._pendiente:
                        break
                    if al.sellado_por_espera and not self._pendiente:
                        break
                    esperar_mas = True
        except GeneracionCancelada:
            # El cliente se fue y abortar() corto la generate() desde dentro.
            # NO es un fallo: no se guarda en self.error ni se imprime traza.
            print(f"[sesion] {self.nombre}: generacion abortada, el cliente se fue",
                  flush=True)
        except LocucionDescarrilada as e:
            # EOS que no llego: ya esta contado en el print del guardia. Se
            # registra como error para que el websocket lo cuente al cliente
            # antes del 'hecho', sin traza -- no hay pila que investigar.
            self.error = str(e)
        except Exception as e:
            import traceback
            self.error = f"{type(e).__name__}: {e}"
            print(f"[error] sesion {self.nombre}:", flush=True)
            traceback.print_exc()
        finally:
            self.terminada = True
            self.cerrada = True
            self.lazo.call_soon_threadsafe(self.cola.put_nowait, _FIN)
            devolver_memoria()

    # ---- el estado de ESTA sesion, y de ninguna otra ----
    # Lo que se le pasa a TextoEnCurso como al_pausar/al_reanudar. Ademas de
    # soltar y recuperar el candado del modelo, se llevan y traen TODO lo que
    # una generate() arrastra fuera de si misma: el RNG, el contador de la
    # rampa de arranque, el remate, los pasos y neg_cada. La lista, y por que
    # cada cosa, esta en foto_generacion/reponer_generacion.
    #
    # POR QUE HACE FALTA
    # Ese estado es del PROCESO y el candado se suelta en cada pausa, asi que la
    # generate() que se cuela en medio -- que hace su torch.manual_seed(), pone
    # el contador de arranque a cero, fija SUS pasos -- dejaba a la primera
    # reanudando con un ruido, una guia y un solver que no eran los suyos. Ver
    # el bloque de _candado_modelo, con la medida del RNG, y el de
    # foto_generacion con la del resto.
    #
    # POR QUE ASI Y NO CON UN torch.Generator PROPIO
    # Un generador por sesion habria que METERLO donde se sortea, y ahi solo se
    # llega parcheando a Microsoft: sample_speech_tokens() llama a torch.randn()
    # sin admitir `generator`. Fotografiar y reponer el estado global consigue lo
    # mismo -- un hilo de ruido por sesion -- sin tocar upstream, y ademas cubre
    # CUALQUIER punto que sortee, no solo el unico que hoy se conoce. Lo mismo
    # vale para los pasos: son un atributo del modelo que sample_speech_tokens
    # lee en cada latente, no un parametro de la llamada.
    #
    # EL ORDEN IMPORTA EN LOS DOS SENTIDOS
    # La foto ANTES de soltar (si no, otra sesion podria avanzar el RNG antes de
    # que se mire) y la reposicion DESPUES de recuperar el candado (si no, se
    # pisaria con la que todavia esta generando).
    #
    # SOLO EL RNG DE CPU
    # El ruido de la difusion sale de un torch.randn(...) SIN device en
    # sample_speech_tokens() y se mueve despues con .to(condition), asi que quien
    # lo sortea es el generador de CPU aunque el modelo corra en mps o cuda.
    # COMPROBADO: tras un torch.randn(4, 64).to("mps") cambia el estado de CPU y
    # NO el de MPS. Si algun dia upstream crea el ruido ya en el dispositivo,
    # aqui hay que guardar tambien torch.mps/cuda.get_rng_state().
    #
    # Cuesta 1,5 us por pausa (5056 bytes de estado del RNG; el resto son
    # cuatro escalares), y las pausas son una por frase: al lado de los
    # segundos que dura una locucion, nada.
    def _pausar(self) -> None:
        self._foto = foto_generacion()
        _candado_modelo.release()

    def _reanudar(self) -> None:
        _candado_modelo.acquire()
        if self._foto is not None:
            reponer_generacion(self._foto, self.pasos, self.neg_cada)

    def _generar(self, al: TextoEnCurso):
        procesador = _estado["procesador"]
        base = prefijo_voz(self.voz)
        # Con el alimentador puesto: es quien le dice a la cola cuando la
        # locucion entra en prorroga y hay que retener (ver ColaAudioSesion).
        audio = ColaAudioSesion(self.lazo, self.cola, texto=al,
                                respiro=self.respiro,
                                cola_final=self.cola_final,
                                recorte_entrada=self.recorte_entrada)
        with self._cond:
            # Bajo el candado y comprobando abortada: si el cliente se fue entre
            # que se armo la cola y que se registra, abortar() no la habria
            # visto y la generate() arrancaria ya sin nadie que la oiga.
            if self.abortada:
                audio.cancelado = True
            self._audio = audio
        # Fuera del try: el `finally` lo mira, y si se resolviera dentro podria
        # no estar definido cuando algo falle antes de llegar a esa linea.
        solapado = _estado.get("solapado")
        _candado_modelo.acquire()
        try:
            _ajustar_pasos(self.pasos)
            _ajustar_neg_cada(self.neg_cada)
            if self.semilla is not None:
                # Aqui EMPIEZA el hilo de ruido de esta locucion; de conservarlo
                # a traves de las pausas se encargan _pausar/_reanudar. Se
                # siembra en cada generate() y no una sola vez por sesion a
                # proposito: una sesion larga puede encadenar varias -- al llegar
                # al tope de caché -- y asi cada una arranca igual que si fuera
                # la primera, que es lo que hace comparable el audio.
                torch.manual_seed(self.semilla)
            # text="" porque el texto ya no viene de aqui: lo pone el
            # alimentador. Lo unico que se aprovecha son los input_ids falsos
            # y las mascaras, que salen de la longitud del prefijo de voz.
            entradas = procesador.process_input_with_cached_prompt(
                text="", cached_prompt=copy.deepcopy(base),
                padding=True, return_tensors="pt", return_attention_mask=True,
            )
            if EN_GPU:
                entradas = a_dispositivo(entradas)
            entradas.pop("tts_text_ids")
            al.posicion_inicial = int(entradas["tts_lm_input_ids"].shape[1])
            self.generaciones += 1
            # El envoltorio va SOLO a generate(); self._audio sigue siendo la
            # cola de verdad, que es a la que abortar() le pone `cancelado` y
            # de la que se leen los `retenidos` al salir.
            destino = StreamerSolapado(solapado, audio) if solapado else audio
            remate_cero()   # por generate(), igual que en _sintetizar
            with torch.no_grad():
                _estado["modelo"].generate(
                    **entradas,
                    tts_text_ids=al,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=procesador.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    show_progress_bar=False,
                    return_speech=False,
                    all_prefilled_outputs=copy.deepcopy(base),
                    audio_streamer=destino,
                )
        finally:
            # ANTES de soltar el candado: si generate() salio por excepcion --
            # descarrile o aborto -- nadie llamo a end(), y el worker podria
            # seguir decodificando trozos de ESTA locucion mientras otra sesion
            # entra y le mueve el estado del decodificador por debajo. Ademas,
            # `retenidos` no esta completo hasta que el worker termina de emitir.
            if solapado is not None:
                with contextlib.suppress(Exception):
                    solapado.drenar()
            _candado_modelo.release()
            if audio.retenidos:
                # Se llega aqui con retenidos solo cuando NO hubo end(): un
                # descarrile (o un aborto) desmonto la pila de generate(). Es
                # EL recorte funcionando -- este audio se genero sin texto
                # detras y el oyente no lo oye --, y el numero es la medida
                # de cuanto parloteo se le ahorro.
                print(f"[sesion] {self.nombre}: se descartan "
                      f"{len(audio.retenidos)} trozos retenidos "
                      f"(~{audio.segundos_retenidos():.1f} s) generados en la "
                      f"prorroga sin EOS", flush=True)
                audio.retenidos = []
            with self._cond:
                self._audio = None


def _caducar_sesiones() -> None:
    ahora = time.time()
    for nombre, s in list(_SESIONES.items()):
        if s.terminada and not s.escuchando and ahora - s.visto > 5:
            _SESIONES.pop(nombre, None)
        elif ahora - s.visto > CADUCIDAD_SESION:
            s.cerrar()
            _SESIONES.pop(nombre, None)


def cabecera_wav_flujo(ritmo: int = RITMO) -> bytes:
    """Cabecera RIFF/WAVE de longitud desconocida (tamanos 0xFFFFFFFF), que es
    la convencion para flujos. ffplay, mpv, Chrome y Firefox la aceptan, asi
    que el cliente reproduce mientras descarga."""
    return b"".join([
        b"RIFF", struct.pack("<I", 0xFFFFFFFF), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, ritmo, ritmo * 2, 2, 16),
        b"data", struct.pack("<I", 0xFFFFFFFF),
    ])


def a_pcm16(trozo: torch.Tensor) -> bytes:
    audio = trozo.detach().float().cpu().numpy().reshape(-1)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class PeticionTTS(BaseModel):
    texto: str = Field(..., min_length=1, max_length=8000)
    voz: str = VOZ_DEFECTO
    # 3.0 y no 1.5: MEDIDO con el banco de fidelidad (scripts/fidelidad.py),
    # que cierra el circuito texto -> voz -> whisper -> texto sobre 6 frases
    # x 3 repeticiones.
    #
    #   cfg 1,5   WER medio 13,6 %   peor 85,7 %   3/6 frases inestables
    #   cfg 3,0   WER medio  3,6 %   peor 14,3 %   1/6
    #
    # El peor caso pasa de 85,7 % a 14,3 %. Y es GRATIS en tiempo: la difusion
    # evalua la rama positiva y la negativa en un lote de 2 pase lo que pase
    # (se midio que doblar el lote cuesta un 5 % mas, no el doble), asi que
    # subir la guia no anade una sola pasada.
    #
    # Ademas 3.0 es el defecto del propio upstream en sample_speech_tokens:
    # ibamos por debajo de lo que el modelo espera.
    cfg_scale: float = Field(3.0, gt=0.5, lt=5.0)

    # MISMO TEXTO, AUDIO DISTINTO CADA VEZ
    # sample_speech_tokens() arranca cada latente con torch.randn() sin
    # semilla, una vez por fotograma acustico. `do_sample=False` no lo toca:
    # eso solo fija que token elige el modelo de lenguaje, no el ruido del
    # que parte la difusion. Medido pidiendo la misma frase cuatro veces:
    # duraciones 3,47 / 3,20 / 3,20 / 3,47 s y correlacion entre pasadas de
    # 0,019 -- es decir, audio sin ningun parecido forma a forma.
    #
    # Casi siempre suena bien, pero de vez en cuando el sorteo cae mal y sale
    # un clip que ni whisper entiende. Con semilla fija eso deja de ser una
    # loteria: la misma peticion da exactamente el mismo audio.
    #
    # LA SEMILLA NO ES LO UNICO QUE HAY QUE FIJAR PARA QUE ESO SE CUMPLA
    # El ruido es el unico sorteo, pero no el unico estado que arrastra una
    # sintesis. El decodificador acustico es causal y en streaming, asi que el
    # audio depende TAMBIEN de las colas de sus convoluciones al empezar. En
    # torch eso no da problema -- generate() crea una cache nueva por llamada --,
    # pero con el motor openvino ese estado vive en el IR compilado, que es uno
    # para todo el proceso, y se colaba de una peticion a la siguiente: misma
    # semilla y md5 distinto. Lo arregla AcusticoOV en pkgs/vibevoice-ov/motor.py
    # haciendo que el estado siga al objeto cache de cada generate().
    #
    # El defecto es el del servicio (VIBEVOICE_SEMILLA; None si no esta puesta,
    # que es el sorteo de siempre). Un "semilla": null EXPLICITO en la peticion
    # sigue sorteando: pydantic conserva el None que manda el cliente.
    semilla: Optional[int] = Field(SEMILLA_DEFECTO, ge=0, lt=2**31,
                                   description="fija el ruido de la difusion; "
                                               "misma semilla = mismo audio")

    # Pasos del solver de difusion por latente. Medido en la VM con int8:
    # 20 -> RTF 2,75 · 8 -> 2,18 · 6 -> 2,18 · 4 -> 2,11. Por debajo de 4 el
    # solver multistep se degrada; por encima de 8 se paga RTF sin ganar nada
    # audible. Por defecto manda VIBEVOICE_PASOS.
    pasos: Optional[int] = Field(None, ge=4, le=20)

    # VELOCIDAD SIN TOCAR EL MODELO
    # VibeVoice no tiene ningun parametro de duracion ni length_scale: el ritmo
    # sale de las 6 ventanas acusticas por cada 5 tokens de texto y no se
    # expone. Lo que si se puede es DECLARAR otro ritmo de muestreo en la
    # cabecera WAV: el audio no se toca, se reproduce mas o menos deprisa.
    # Cuesta cero CPU. El precio es que el tono sube o baja con la velocidad,
    # asi que el margen util es estrecho: a +-10% no se nota, mas alla suena a
    # ardilla o a resaca. De ahi el rango cerrado.
    velocidad: float = Field(1.0, ge=0.85, le=1.20)

    # Agrupado de la rama incondicional del CFG: una pasada del backbone cada
    # N fotogramas en vez de una por fotograma. Ver NEG_CADA y TtsLmOV. None =
    # lo que diga VIBEVOICE_NEG_CADA. Va por peticion para poder medirlo sin
    # reiniciar el servicio ni pelearse con las variables de la unidad.
    neg_cada: Optional[int] = Field(None, ge=1, le=6)

    # Fotogramas de cola que se emiten tras el EOS del clasificador, donde se
    # apaga la ultima palabra (ver el bloque COLA FINAL). 0 = el corte en seco
    # de antes; None = lo que diga VIBEVOICE_COLA_FINAL. Va por peticion para
    # poder medir el antes y el despues con el MISMO binario.
    cola_final: Optional[int] = Field(None, ge=0, le=6)

    # Tirar o no los fotogramas callados de ANTES de la primera palabra (ver
    # RecorteEntrada). None = lo que diga VIBEVOICE_RECORTE_ENTRADA. Va por
    # peticion por lo mismo que cola_final: para medir el antes y el despues
    # con el mismo binario, que es como se midio.
    recorte_entrada: Optional[bool] = None


class PeticionSesion(BaseModel):
    """Texto que se le mete a una sesion viva. La voz y los ajustes solo se
    miran al CREARLA: cambiarlos a mitad exigiria empezar otra locucion."""
    texto: str = Field(..., min_length=1, max_length=8000)
    # voz OPCIONAL a proposito. Si tuviera valor por defecto, el cliente que
    # manda solo {"texto": ...} en las frases siguientes -- que es lo natural --
    # estaria pidiendo la voz por defecto sin saberlo y se llevaria un 409 por
    # "cambio de voz a mitad de sesion". Solo se comprueba si viene puesta.
    voz: Optional[str] = None
    cfg_scale: float = Field(3.0, gt=0.5, lt=5.0)
    semilla: Optional[int] = Field(SEMILLA_DEFECTO, ge=0, lt=2**31)
    pasos: Optional[int] = Field(None, ge=4, le=20)
    # Aire en cada final de frase (ver el bloque RESPIRO). Solo se mira al
    # CREAR la sesion, como la voz. respiro=False da el audio pelado del
    # modelo, bit a bit el de /tts/stream con " ".join: ni alargue ni recorte.
    respiro: bool = True
    # Cola de la ultima palabra; ver el bloque COLA FINAL. Como la voz, solo se
    # mira al CREAR la sesion.
    cola_final: Optional[int] = Field(None, ge=0, le=6)
    # Agrupado de la rama incondicional, como en PeticionTTS. Solo al CREAR.
    neg_cada: Optional[int] = Field(None, ge=1, le=6)
    # Recorte del aire de entrada, como en PeticionTTS. Solo al CREAR.
    recorte_entrada: Optional[bool] = None
    # Cerrar en la misma llamada que se manda la ultima frase, que es lo comun.
    fin: bool = False


@app.get("/", response_class=HTMLResponse)
def pagina_prueba() -> HTMLResponse:
    """Pagina de prueba, servida por el PROPIO servicio.

    Tiene que salir de aqui y no de un sitio externo: el navegador bloquearia
    la peticion por CORS, y ademas asi funciona desde el movil a traves del
    tunel sin configurar nada.
    """
    ruta = Path(__file__).with_name("prueba.html")
    if not ruta.exists():
        raise HTTPException(404, "pagina de prueba no incluida en esta version")
    return HTMLResponse(ruta.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {
        "estado": "ok",
        "motor": _estado.get("motor", MOTOR),
        "dispositivo": DISPOSITIVO,
        "pasos": PASOS,
        # None = cada peticion sortea su ruido; un numero = el servicio es
        # determinista salvo que el cliente mande "semilla": null.
        "semilla_defecto": SEMILLA_DEFECTO,
        "voz_defecto": VOZ_DEFECTO,
        "rtf_esperado": RTF_MEDIDO,
        "ocupado": _candado.locked() or _candado_modelo.locked(),
        "auth": "bearer" if TOKEN else "abierta",
        # Que IR y que ajustes hay puestos DE VERDAD. Sin esto, comprobar un
        # despliegue exige leer la unidad de systemd y creerse que nadie ha
        # dejado un drop-in por medio.
        "ir": {"lm": Path(IR_LM).name, "cabeza": Path(IR_CABEZA).name,
               "acustico": Path(IR_ACUSTICO).name} if MOTOR == "openvino" else {},
        "hilos": {"total": HILOS, "decoder": HILOS_DECODER,
                  "solapado": SOLAPAR_DECODER},
        "neg_cada": NEG_CADA,
        # La cola de la ultima palabra, que es global a las dos vias (ver el
        # bloque COLA FINAL). Fuera de "sesiones" porque /tts/stream la lleva
        # igual.
        "cola_final": {"fotogramas": COLA_FINAL, "umbral_pico": COLA_FINAL_PICO,
                       "insistir": COLA_INSISTIR and not SOLAPAR_DECODER},
        # El aire de ANTES de la primera palabra, que tampoco se emite. Igual
        # que cola_final, va por las dos vias.
        "recorte_entrada": RECORTE_ENTRADA,
        "sesiones": {"activas": SESIONES_ACTIVAS,
                     "abiertas": sorted(_SESIONES),
                     "espera_texto_s": ESPERA_TEXTO,
                     "tope_cache": TOPE_CACHE,
                     "respiro": {"activo": RESPIRO_ACTIVO,
                                 "fotogramas": RESPIRO_FOTOGRAMAS,
                                 "alarga": RESPIRO_ALARGA,
                                 "tope": RESPIRO_TOPE,
                                 "umbral_rms": RESPIRO_UMBRAL,
                                 "umbral_pico": RESPIRO_PICO,
                                 "prerrollo": RESPIRO_PRERROLLO,
                                 "pie": RESPIRO_PIE},
                     "websocket": "/tts/sesion/ws"},
    }


@app.get("/crono")
def crono(reset: bool = True) -> dict:
    """Reparto del tiempo de la ULTIMA tanda de generaciones, y lo pone a cero.

    Se lee despues de un banco, no durante: mezcla todas las generate() que
    hayan pasado desde el ultimo reset. `motor` son los tiempos por componente
    del camino OpenVINO (suman mas del 100 % del reloj de pared, porque el
    decodificador corre en otro hilo); `tuberia` dice quien espera a quien.
    """
    datos = {"tuberia": dict(CRONO_TUBERIA)}
    try:                                    # solo existe con motor openvino
        import motor as _motor
        datos["motor"] = {k: {"s": v[0], "n": v[1],
                              "ms": (v[0] / v[1] * 1000) if v[1] else 0.0}
                          for k, v in _motor.CRONO.items()}
        if reset:
            for v in _motor.CRONO.values():
                v[0], v[1] = 0.0, 0
    except Exception as e:                  # noqa: BLE001
        datos["motor"] = {"sin_datos": str(e)}
    m = datos.get("motor") or {}
    t = datos["tuberia"]
    if t["generaciones"] and isinstance(m.get("tts_lm"), dict):
        # Lo que NO es ninguna de las tres piezas compiladas: el LM de texto en
        # torch, el conector acustico, el clasificador de EOS y el despacho de
        # Python. Es la partida que nadie mira y en la que hay que buscar
        # cuando las tres piezas ya no dan mas.
        piezas = sum(m[k]["s"] for k in ("tts_lm", "cabeza", "acustico"))
        fot = m["acustico"]["n"] or 1
        datos["reparto_ms_por_fotograma"] = {
            "generate": 1000 * t["generate"] / fot,
            "tts_lm": 1000 * m["tts_lm"]["s"] / fot,
            "cabeza": 1000 * m["cabeza"]["s"] / fot,
            "acustico": 1000 * m["acustico"]["s"] / fot,
            "resto": 1000 * (t["generate"] - piezas) / fot,
            "fotogramas": fot,
        }
    if t["generaciones"]:
        datos["resumen"] = {
            "generate_s": t["generate"],
            "contrapresion_pct": 100 * t["contrapresion"] / t["generate"],
            "worker_ocupado_pct": 100 * t["trabajo_worker"] / t["generate"],
            "worker_hambre_pct": 100 * t["hambre_worker"] / t["generate"],
            "decode_ms": 1000 * t["trabajo_worker"] / max(1, t["decodes"]),
            "decodes": t["decodes"],
        }
    if reset:
        crono_cero()
    return datos


@app.get("/voces")
def voces(_=Depends(autorizar)):
    """Las voces instaladas. Sin esto el cliente tiene que adivinar nombres, y
    equivocarse solo se nota con un 404 a mitad de una peticion."""
    return {"voces": sorted(p.stem for p in VOCES_DIR.glob("*.pt")),
            "defecto": VOZ_DEFECTO}


@app.post("/tts/stream")
async def tts_stream(pet: PeticionTTS, _=Depends(autorizar)) -> StreamingResponse:
    prefijo_voz(pet.voz)  # valida ANTES de enviar cabeceras, para dar un 404 limpio
    # La velocidad NO se hace remuestreando. Remuestrear mueve el tono junto
    # con la duracion, y a +-15% lo que se oye es "mas agudo", no "mas rapido".
    # Aqui se estira el tiempo de verdad (WSOLA, ver estirar.py) y el ritmo de
    # salida no cambia nunca.
    ritmo = RITMO
    estirando = abs(pet.velocidad - 1.0) > 1e-3

    async def generador():
        # El candado se toma DENTRO del generador: si hay otra sintesis en
        # curso, esta espera su turno sin bloquear el bucle de eventos.
        async with _candado:
            streamer = StreamerCancelable(pet.cola_final, pet.recorte_entrada)
            lazo = asyncio.get_running_loop()
            # generate() es bloqueante -> hilo del executor.
            tarea = lazo.run_in_executor(
                None, _sintetizar, pet.texto, pet.voz, pet.cfg_scale, streamer,
                pet.semilla, pet.pasos, pet.neg_cada,
            )
            try:
                yield cabecera_wav_flujo(ritmo)
                if not estirando:
                    async for trozo in streamer.flujo():
                        yield a_pcm16(trozo)  # ~133 ms de audio por trozo
                else:
                    # A velocidad distinta de 1 se acumula la frase ENTERA y se
                    # estira de una vez. Estirar cada trozo de 133 ms por su
                    # cuenta dejaria una costura audible en cada empalme, y
                    # arrastrar el estado de WSOLA entre trozos es mas maquinaria
                    # de la que merece: una frase dura unos segundos, asi que lo
                    # unico que se pierde es la reproduccion progresiva DENTRO de
                    # la frase. Al narrar por frases encadenadas ni se nota.
                    trozos = []
                    async for trozo in streamer.flujo():
                        trozos.append(trozo.detach().float().cpu().numpy().reshape(-1))
                    if trozos:
                        entero = np.concatenate(trozos)
                        yield a_pcm16(torch.from_numpy(estirar(entero, pet.velocidad)))
            finally:
                # Cliente desconectado o flujo terminado: marcamos cancelado
                # (inofensivo si ya acabo) y esperamos al hilo, para no solapar
                # dos generaciones bajo el candado.
                streamer.cancelado = True
                await tarea
                # Cada sintesis deja cientos de MB de activaciones. Sin esto
                # el RSS crece peticion a peticion hasta que el OOM decide.
                devolver_memoria()

    return StreamingResponse(
        generador(),
        media_type="audio/wav",
        headers={
            # Sin Content-Length: uvicorn usa Transfer-Encoding: chunked.
            "Cache-Control": "no-store",
            "X-Ritmo-Hz": str(ritmo),
            "X-RTF-Esperado": str(RTF_MEDIDO),
            "Content-Disposition": 'inline; filename="voz.wav"',
        },
    )


# --------------------------------------------------------------------------
# API de sesiones. Tres verbos y un flujo de audio:
#
#   POST /tts/sesion/{id}          {"texto": "..."}   crea la sesion y encola
#   GET  /tts/sesion/{id}/audio                       WAV continuo, hasta el fin
#   POST /tts/sesion/{id}/fin                         cierra la locucion
#
# El texto va por un lado y el audio por otro A PROPOSITO. La locucion es UNA,
# continua, y el modelo va por detras del texto: cuando termina de decir la
# frase 2 ya se le metio la 3. Devolver "el audio de esta frase" en la respuesta
# de cada POST seria mentir, porque en ese instante todavia no existe.
#
# El orden es: POST con la primera frase, GET del audio, y a partir de ahi POST
# cuantos haga falta. Un POST no espera a nada: vuelve en cuanto encola.


def _sesion(nombre: str) -> "SesionViva":
    s = _SESIONES.get(nombre)
    if s is None:
        raise HTTPException(404, f"no hay sesion '{nombre}'")
    return s


@app.post("/tts/sesion/{nombre}")
async def sesion_texto(nombre: str, pet: PeticionSesion,
                       _=Depends(autorizar)) -> dict:
    """Encola texto. La primera llamada crea la sesion y arranca su generate().

    Si la sesion anterior con ese nombre ya habia terminado -- por inactividad o
    porque alguien la cerro --, esta llamada abre otra y la respuesta lo dice en
    `reabierta`. Ojo: el flujo de /audio de la anterior ya se cerro, asi que hay
    que volver a pedirlo.
    """
    if not SESIONES_ACTIVAS:
        raise HTTPException(503, "sesiones desactivadas (VIBEVOICE_SESIONES=0)")
    _caducar_sesiones()
    prefijo_voz(pet.voz or VOZ_DEFECTO)  # valida antes de montar nada
    s = _SESIONES.get(nombre)
    reabierta = s is not None and s.terminada
    if reabierta:
        _SESIONES.pop(nombre, None)
        s = None
    nueva = s is None
    if nueva:
        s = SesionViva(nombre, pet.voz or VOZ_DEFECTO, pet.cfg_scale,
                       pet.semilla, pet.pasos, asyncio.get_running_loop(),
                       respiro=pet.respiro, cola_final=pet.cola_final,
                       neg_cada=pet.neg_cada,
                       recorte_entrada=pet.recorte_entrada)
        _SESIONES[nombre] = s
    elif pet.voz is not None and pet.voz != s.voz:
        raise HTTPException(409, f"sesion '{nombre}' esta en voz '{s.voz}'; "
                                 f"cierrala para cambiar de voz")
    s.alimentar(pet.texto)
    if nueva:
        # Despues de alimentar: el hilo termina si arranca sin nada que decir.
        s.arrancar()
    if pet.fin:
        s.cerrar()
    return {**s.estado(), "reabierta": reabierta}


@app.get("/tts/sesion/{nombre}/audio")
async def sesion_audio(nombre: str, _=Depends(autorizar)) -> StreamingResponse:
    """El audio de la sesion entera, como un solo WAV que va llegando."""
    s = _sesion(nombre)
    if s.escuchando:
        raise HTTPException(409, f"ya hay un oyente en la sesion '{nombre}'")
    s.escuchando = True

    async def generador():
        try:
            yield cabecera_wav_flujo(RITMO)
            while True:
                trozo = await s.cola.get()
                if trozo is _FIN:
                    break
                yield a_pcm16(trozo)
        finally:
            s.escuchando = False
            # Si el que escuchaba se fue, la locucion no le sirve a nadie.
            s.cerrar()
            if s.terminada:
                _SESIONES.pop(nombre, None)

    return StreamingResponse(
        generador(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-Ritmo-Hz": str(RITMO),
            "X-RTF-Esperado": str(RTF_MEDIDO),
            "Content-Disposition": f'inline; filename="{nombre}.wav"',
        },
    )


@app.post("/tts/sesion/{nombre}/fin")
async def sesion_fin(nombre: str, _=Depends(autorizar)) -> dict:
    """Cierra la locucion: el modelo dice lo que le queda y termina."""
    s = _sesion(nombre)
    s.cerrar()
    return s.estado()


@app.get("/tts/sesion/{nombre}")
async def sesion_estado(nombre: str, _=Depends(autorizar)) -> dict:
    return _sesion(nombre).estado()


@app.delete("/tts/sesion/{nombre}")
async def sesion_borrar(nombre: str, _=Depends(autorizar)) -> dict:
    s = _sesion(nombre)
    s.cerrar()
    _SESIONES.pop(nombre, None)
    return {"sesion": nombre, "cerrada": True}


# --------------------------------------------------------------------------
# WEBSOCKET: la misma sesion, pero en una sola conexion
#
# POR QUE, SI YA FUNCIONA POR HTTP
# La interaccion es bidireccional y de larga duracion: entra texto mientras
# sale audio, durante minutos. HTTP obliga a partir eso en un POST por frase
# mas un GET de audio que dura toda la locucion, y a que el cliente sondee
# GET /tts/sesion/{id} para saber si el modelo se ha quedado sin texto por
# delante. Aqui es un solo socket: el texto sube, el audio y los avisos bajan,
# y "el modelo esta esperando" llega como evento en vez de por sondeo.
#
# LO QUE NO CAMBIA
# Por debajo es SesionViva, la misma clase, sin una rama especial. Asi que
# respeta _candado_modelo igual (lo toma _generar), suelta el candado mientras
# espera texto igual (TextoEnCurso._esperar, via al_pausar/al_reanudar) y
# tokeniza igual (trozos cosidos con espacio y un "\n" al final; el respiro no
# toca el texto, solo el audio -- ver alimentar() y el bloque RESPIRO). De
# ahi que el audio salga IDENTICO bit a bit al de la via HTTP con los mismos
# ajustes, que es lo que se comprueba por md5. El HTTP se queda intacto: es
# lo que corre en la VM y lo que se prueba con curl.
#
# PROTOCOLO
# Del cliente al servidor, mensajes de TEXTO con JSON:
#
#   {"accion":"abrir","voz":"sp-Spk3_man","cfg_scale":4.5,"pasos":6,"semilla":11}
#   {"accion":"texto","texto":"..."}
#   {"accion":"fin"}
#
# Del servidor al cliente, mensajes BINARIOS con el marco de
# scripts/asistente_web.py:
#
#   [tipo:1 byte][longitud:4 bytes big-endian][carga]
#   tipo 0 = PCM crudo, 16 bits con signo, 24000 Hz, mono
#   tipo 1 = evento JSON
#
# El websocket ya trae longitud propia, asi que el marco es redundante ahi. Se
# mantiene A PROPOSITO: el lector de asistente_web.py -- el bucle que acumula
# hasta tener el marco entero y reparte por tipo -- vale tal cual, sin tocar
# una linea, y el navegador puede alimentar el mismo camino de Web Audio venga
# el flujo de donde venga. Lo que cuesta son 5 bytes por trozo de 133 ms.
#
# Eventos (campo "tipo"): abierta, texto, esperando, fin_texto, sonando,
# hecho, error. Ver el detalle en sesion_ws().
#
# DEPENDENCIA QUE NO SE VE: uvicorn no habla websocket por si solo, necesita el
# paquete `websockets` (o wsproto). Aqui llega por uvicorn[standard], que es
# dependencia DIRECTA de vibevoice en el lock, asi que esta tanto en el venv del
# Mac como en la imagen Docker. Si algun dia se poda -- docker/Dockerfile.voz-
# stream ya desinstala uvloop, que viene del mismo extra --, esta ruta deja de
# existir y uvicorn responde 404 al apreton de manos SIN decir por que.
#
# VELOCIDAD: NO SE ADMITE DISTINTA DE 1, Y ES A PROPOSITO
# estirar() es WSOLA (ver estirar.py) y necesita la locucion ENTERA por tres
# razones que aqui no se pueden salvar:
#
#   1. Arrastra estado entre ventanas -- la `referencia` con la que empalma la
#      siguiente -- y mira 1024+384 muestras por delante. Estirar cada trozo de
#      133 ms por su cuenta deja una costura audible en CADA empalme, y aqui
#      los empalmes son ~7 por segundo durante toda la locucion.
#   2. La salida no dura lo mismo que la entrada, asi que los trozos dejarian de
#      encajar con los latentes y el cliente no podria alinear nada.
#   3. La locucion de una sesion NO TIENE FIN CONOCIDO: encadena generate() al
#      llegar a TOPE_CACHE y dura lo que el de arriba siga escribiendo.
#      "Acumularlo todo y estirar al final" es exactamente no hacer streaming:
#      seria no emitir un solo byte hasta el "fin", que es todo lo contrario de
#      para lo que existe este websocket.
#
# /tts/stream si lo hace -- acumula la frase y la estira de golpe -- porque alli
# la locucion es UNA frase de unos segundos y se sabe cuando acaba. Aqui se
# rechaza con un error claro en vez de fingir. El cliente que quiera ir mas
# deprisa tiene dos salidas honestas: (a) reproducir el PCM a otro ritmo
# (playbackRate en Web Audio), que es gratis pero mueve el tono -- justo lo que
# estirar.py existe para evitar --, o (b) acumular el audio de su lado y
# estirarlo el, ya sin restriccion de tiempo real. Ninguna de las dos es
# trabajo del servidor.

MARCO_PCM = 0
MARCO_EVENTO = 1


def marco(tipo: int, carga: bytes) -> bytes:
    """[tipo:1][longitud:4 BE][carga], igual que scripts/asistente_web.py."""
    return struct.pack(">BI", tipo, len(carga)) + carga


class AbrirSesionWS(BaseModel):
    """Ajustes de la locucion. Solo se miran AL ABRIR: cambiarlos a mitad
    obligaria a empezar otra locucion, que es justo lo que la sesion evita."""
    sesion: Optional[str] = Field(None, min_length=1, max_length=64)
    voz: str = VOZ_DEFECTO
    cfg_scale: float = Field(3.0, gt=0.5, lt=5.0)
    semilla: Optional[int] = Field(SEMILLA_DEFECTO, ge=0, lt=2**31)
    pasos: Optional[int] = Field(None, ge=4, le=20)
    # Pausa de verdad en cada punto (ver el bloque RESPIRO). False = la
    # locucion de antes, bit a bit.
    respiro: bool = True
    # Cola de la ultima palabra; ver el bloque COLA FINAL. 0 vuelve al corte en
    # seco de antes, que es contra lo que se midio.
    cola_final: Optional[int] = Field(None, ge=0, le=6)
    # Agrupado de la rama incondicional, como en PeticionTTS.
    neg_cada: Optional[int] = Field(None, ge=1, le=6)
    # Recorte del aire de entrada, como en PeticionTTS.
    recorte_entrada: Optional[bool] = None
    # Aceptado solo para poder dar un error claro; ver el bloque VELOCIDAD.
    velocidad: float = Field(1.0, ge=0.85, le=1.20)


def _autorizado_ws(ws: WebSocket, token: Optional[str]) -> bool:
    """El mismo bearer que autorizar(), mas un repliegue por query.

    OJO, ESTO DEJA EL TOKEN EN LOS REGISTROS: la URL completa -- con
    ?token=... -- aparece en el log de acceso de uvicorn, en el de cualquier
    proxy que haya delante y en el historial del navegador. Es aceptable en una
    red domestica y NO lo es en internet.

    Se admite igualmente porque la API de WebSocket del navegador no deja poner
    cabeceras: `new WebSocket(url)` no tiene donde meter Authorization, y el
    unico hueco del protocolo (Sec-WebSocket-Protocol) es un apano peor. Los
    clientes que SI pueden -- python, curl, cualquier cosa que no sea un
    navegador -- deben usar la cabecera, que es lo que se mira primero.
    """
    if not TOKEN:
        return True
    cabecera = ws.headers.get("authorization", "")
    if cabecera.lower().startswith("bearer "):
        dado = cabecera[7:].strip()
    else:
        dado = token or ""
    return secrets.compare_digest(dado, TOKEN)


@app.websocket("/tts/sesion/ws")
async def sesion_ws(ws: WebSocket, token: Optional[str] = Query(None)) -> None:
    """Una sesion viva por conexion. Ver el bloque WEBSOCKET de arriba.

    Eventos que emite, en el orden tipico:

      abierta     la sesion existe y tiene nombre (util para GET /tts/sesion/{id})
      texto       acuse de cada frase encolada, con los tokens que salieron
      sonando     primer trozo de PCM: a partir de aqui ya se oye algo
      esperando   el modelo se quedo sin texto por delante y esta PARADO
                  (esperando=true) o volvio a arrancar (esperando=false). Es el
                  mismo flag que la via HTTP publica en GET /tts/sesion/{id},
                  pero empujado en vez de sondeado: es la senal de "manda ya la
                  frase siguiente si no quieres un silencio".
      fin_texto   acuse de {"accion":"fin"}: no entra mas texto
      hecho       la locucion termino; lleva el estado final de la sesion
      error       cualquier cosa que salio mal, con texto explicativo
    """
    if not _autorizado_ws(ws, token):
        # Se rechaza ANTES del accept: Starlette contesta 403 al apreton de
        # manos y no llega a existir websocket ninguno. Asi un cliente sin token
        # no consume ni una sesion ni un hilo.
        await ws.close(code=1008)
        return
    await ws.accept()

    lazo = asyncio.get_running_loop()
    t0 = time.perf_counter()
    candado_envio = asyncio.Lock()
    s: Optional[SesionViva] = None
    nombre: Optional[str] = None
    arrancado = False
    tareas: list = []

    async def enviar(tipo: int, carga: bytes) -> None:
        # Dos tareas emiten a la vez (la bomba de audio y el vigia de
        # 'esperando') y send_bytes no es reentrante: sin este candado se
        # podrian intercalar dos marcos y el lector del cliente leeria basura.
        async with candado_envio:
            await ws.send_bytes(marco(tipo, carga))

    async def evento(**kw) -> None:
        await enviar(MARCO_EVENTO, json.dumps(kw, ensure_ascii=False).encode())

    def transcurrido() -> float:
        return round(time.perf_counter() - t0, 3)

    async def bombear() -> str:
        """Vuelca la cola de audio de la sesion al socket."""
        primero = True
        try:
            while True:
                trozo = await s.cola.get()
                if trozo is _FIN:
                    return "hecho"
                if primero:
                    primero = False
                    await evento(tipo="sonando", s=transcurrido())
                await enviar(MARCO_PCM, a_pcm16(trozo))
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            return "roto"

    async def vigilar() -> None:
        """Empuja los cambios del flag 'esperando'.

        Se SONDEA el estado en vez de colgar una devolucion de llamada dentro de
        TextoEnCurso porque 'esperando' tiene dos fuentes -- el bucle parado sin
        texto por delante y el hueco ENTRE dos generate() -- y solo estado() las
        junta.

        EL PERIODO SI SE NOTA, y estaba en 100 ms. Este evento es lo unico que
        le dice al cliente "manda ya la frase siguiente": mientras no salga, el
        modelo esta PARADO sin nada que decir. Sondear cada 100 ms le regala al
        modelo una espera de 0 a 100 ms en CADA frontera de frase, y el
        asistente alimenta frase a frase. Medido con 6 frases (4 pasadas): con
        100 ms la sesion va a RTF 0,985 alimentada frase a frase frente a 0,959
        de golpe; con 10 ms las dos van igual. No es gratis del todo -- son 100
        vueltas por segundo de un dict y un corte de lista -- pero al lado de
        los 133 ms que cuesta un fotograma no se mide.
        """
        antes = False
        try:
            while True:
                ahora = bool(s.estado()["esperando"])
                if ahora != antes:
                    antes = ahora
                    await evento(tipo="esperando", esperando=ahora,
                                 s=transcurrido())
                await asyncio.sleep(PERIODO_VIGIA)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            return

    async def leer() -> str:
        """Consume los mensajes del cliente hasta que se va.

        No termina con {"accion":"fin"}: despues de cerrar el texto aun queda
        por bajar todo el audio, y el que decide que se acabo es la bomba.
        """
        nonlocal arrancado
        while True:
            try:
                msg = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                return "desconectado"
            if msg["type"] == "websocket.disconnect":
                return "desconectado"
            crudo = msg.get("text")
            if crudo is None:
                await evento(tipo="error",
                             texto="se esperaba JSON de texto, llego binario")
                continue
            try:
                m = json.loads(crudo)
            except (json.JSONDecodeError, TypeError):
                await evento(tipo="error", texto="el mensaje no es JSON")
                continue
            accion = m.get("accion") if isinstance(m, dict) else None
            if accion == "texto":
                cuerpo = (m.get("texto") or "").strip()
                if not cuerpo:
                    await evento(tipo="error", texto="texto vacio")
                    continue
                try:
                    n = s.alimentar(cuerpo)
                except HTTPException as e:
                    await evento(tipo="error", texto=str(e.detail))
                    continue
                if not arrancado:
                    # Despues de alimentar, igual que por HTTP: el hilo se muere
                    # solo si arranca sin nada que decir.
                    s.arrancar()
                    arrancado = True
                await evento(tipo="texto", tokens=n, s=transcurrido())
            elif accion == "fin":
                s.cerrar()
                if not arrancado:
                    # Nunca hubo texto, asi que el hilo no arranco y nadie va a
                    # poner el centinela en la cola. Sin esto la bomba se
                    # quedaria esperando un fin que no llega.
                    s.terminada = True
                    s.cola.put_nowait(_FIN)
                await evento(tipo="fin_texto", s=transcurrido())
            elif accion == "abrir":
                await evento(tipo="error",
                             texto="la sesion ya esta abierta; abre otra conexion")
            else:
                await evento(tipo="error", texto=f"accion desconocida: {accion!r}")

    try:
        # ---- 1) primer mensaje: abrir ----
        try:
            crudo = await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError, KeyError):
            return
        try:
            pet = json.loads(crudo)
        except (json.JSONDecodeError, TypeError):
            await evento(tipo="error", texto="el primer mensaje no es JSON")
            return
        if not isinstance(pet, dict) or pet.get("accion") != "abrir":
            await evento(tipo="error",
                         texto='el primer mensaje tiene que ser {"accion":"abrir"}')
            return
        try:
            cfg = AbrirSesionWS(**{k: v for k, v in pet.items() if k != "accion"})
        except Exception as e:
            await evento(tipo="error",
                         texto=f"ajustes invalidos: {str(e).splitlines()[0]}")
            return
        if abs(cfg.velocidad - 1.0) > 1e-3:
            # Ver el bloque VELOCIDAD de arriba. Mejor un error claro que un
            # audio con una costura cada 133 ms o un socket que no emite nada
            # hasta el final.
            await evento(
                tipo="error",
                texto="velocidad != 1 no se admite por websocket: estirar() "
                      "necesita la locucion entera y aqui no tiene fin conocido. "
                      "Usa /tts/stream para una frase suelta, o cambia el ritmo "
                      "al reproducir.")
            return
        if not SESIONES_ACTIVAS:
            await evento(tipo="error",
                         texto="sesiones desactivadas (VIBEVOICE_SESIONES=0)")
            return
        _caducar_sesiones()
        try:
            prefijo_voz(cfg.voz)   # valida antes de montar nada
        except HTTPException as e:
            await evento(tipo="error", texto=str(e.detail))
            return
        nombre = cfg.sesion or f"ws-{uuid.uuid4().hex[:8]}"
        if nombre in _SESIONES:
            await evento(tipo="error", texto=f"la sesion '{nombre}' ya existe")
            nombre = None
            return
        s = SesionViva(nombre, cfg.voz, cfg.cfg_scale, cfg.semilla, cfg.pasos,
                       lazo, respiro=cfg.respiro, cola_final=cfg.cola_final,
                       neg_cada=cfg.neg_cada,
                       recorte_entrada=cfg.recorte_entrada)
        # El audio ya sale por aqui: que GET /tts/sesion/{id}/audio no lo robe.
        s.escuchando = True
        _SESIONES[nombre] = s
        await evento(tipo="abierta", sesion=nombre, voz=cfg.voz,
                     cfg_scale=cfg.cfg_scale, semilla=cfg.semilla,
                     pasos=cfg.pasos if cfg.pasos is not None else PASOS,
                     ritmo=RITMO, formato="pcm_s16le_mono",
                     rtf_esperado=RTF_MEDIDO, s=transcurrido())

        # ---- 2) texto para arriba, audio para abajo ----
        bomba = asyncio.create_task(bombear(), name=f"ws-audio-{nombre}")
        lector = asyncio.create_task(leer(), name=f"ws-texto-{nombre}")
        vigia = asyncio.create_task(vigilar(), name=f"ws-espera-{nombre}")
        tareas = [bomba, lector, vigia]
        # Termina lo que ocurra primero: o se acaba la locucion (bomba) o se va
        # el cliente (lector). El vigia no decide nada, solo acompana.
        await asyncio.wait({bomba, lector},
                           return_when=asyncio.FIRST_COMPLETED)
        if bomba.done() and bomba.result() == "hecho":
            est = s.estado()
            # En suppress porque la locucion puede acabar en el mismo instante
            # en que el cliente cierra: entonces el socket ya no admite nada y
            # eso no es un fallo que merezca una traza en el log.
            with contextlib.suppress(Exception):
                if est.get("error"):
                    await evento(tipo="error", texto=est["error"])
                await evento(tipo="hecho", s=transcurrido(), estado=est)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        with contextlib.suppress(Exception):
            await evento(tipo="error", texto=f"{type(e).__name__}: {e}")
    finally:
        for t in tareas:
            t.cancel()
        if tareas:
            await asyncio.gather(*tareas, return_exceptions=True)
        if s is not None:
            # Se saca del registro ANTES de esperar al hilo: GET /tts/sesion/{id}
            # tiene que dar 404 en cuanto se cae el websocket, no cuando el hilo
            # se entere. Y se ABORTA en vez de cerrar educadamente porque ya no
            # hay nadie escuchando (ver SesionViva.abortar).
            _SESIONES.pop(nombre, None)
            s.escuchando = False
            s.abortar()
            # join en un hilo del executor para no bloquear el bucle de eventos.
            # El plazo es por si generate() se atasca: mas vale un aviso en el
            # log que un handler colgado para siempre.
            if not await lazo.run_in_executor(None, s.esperar_fin, 15.0):
                print(f"[aviso] sesion {nombre}: el hilo sigue vivo 15 s despues "
                      f"de abortar; puede quedar reteniendo _candado_modelo",
                      flush=True)
        with contextlib.suppress(Exception):
            await ws.close()


def identificar_codigo() -> None:
    """Deja en el registro QUE FICHERO se esta ejecutando y su huella.

    NO ES DECORACION. El 6 de agosto se perdieron horas midiendo un fallo del
    respiro contra un servicio que NO tenia el respiro: un drop-in olvidado en
    /run/systemd/system/voz-stream.service.d/prueba.conf reescribia ExecStart
    para apuntar a una copia de trabajo en /var/lib/voz-prueba/voz_stream.py.
    Como /run es tmpfs, el drop-in sobrevive a daemon-reload y a
    `nixos-rebuild switch` -- la generacion nueva se instalaba, el servicio se
    reiniciaba y seguia arrancando el fichero viejo. El despliegue decia que si
    y el proceso corria codigo de cinco commits antes, con el freno de guia en
    un 0.80 que ya no estaba en ninguna configuracion.

    Nada en el arranque lo delataba: la unidad en /etc apuntaba al store, y
    solo el cmdline del PID contaba la verdad. Con esta linea, un
    `journalctl -u voz-stream | grep codigo` la cuenta sola.
    """
    try:
        ruta = Path(__file__).resolve()
        datos = ruta.read_bytes()
        huella = hashlib.sha256(datos).hexdigest()[:12]
        print(f"[arranque] codigo {ruta} sha256:{huella} "
              f"({len(datos.splitlines())} lineas)", flush=True)
    except Exception as e:
        # Nunca impedir el arranque por no poder identificarse.
        print(f"[arranque] no se pudo identificar el codigo: {e}", flush=True)


def main() -> None:
    identificar_codigo()
    # workers=1 SIEMPRE: cada worker cargaria su propio modelo (~2,3 GB) y la
    # VM de 5 GB no aguanta dos.
    uvicorn.run(
        app,
        host=os.environ.get("VOZ_STREAM_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOZ_STREAM_PUERTO", "8082")),
        workers=1,
        log_level=os.environ.get("VOZ_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
