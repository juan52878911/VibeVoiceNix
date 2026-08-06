"""Servidor de TTS con respuesta HTTP en streaming.

    POST /tts/stream  {"texto": "..."}  ->  audio/wav troceado

El cliente empieza a oír ~0,2 s después de pedirlo, mientras el resto se
genera. Verificado: los trozos emitidos son bit a bit identicos al audio
completo (mismo md5 que la generacion no-streaming).

Y para narrar algo que aun se esta escribiendo -- la salida de un LLM, por
ejemplo -- hay ademas SESIONES, que son una sola locucion continua a la que se
le va metiendo texto:

    POST /tts/sesion/{id}        {"texto": "..."}    encola texto
    GET  /tts/sesion/{id}/audio                      un WAV, toda la locucion
    POST /tts/sesion/{id}/fin                        cierra la locucion

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
import copy
import ctypes
import gc
import os
import platform
import sys
import struct
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estirar import estirar  # noqa: E402
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

MODELO_DIR = os.environ["VIBEVOICE_MODELO"]
VOCES_DIR = Path(os.environ["VIBEVOICE_VOCES"])
PASOS = int(os.environ.get("VIBEVOICE_PASOS", "6"))
VOZ_DEFECTO = os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man")
TOKEN = os.environ.get("VOZ_TOKEN", "").strip()
if not TOKEN:
    # Legitimo en una maquina aislada, pero tiene que VERSE: un .env mal
    # generado deja el servicio abierto sin que nadie lo note.
    print(
        "[AVISO] VOZ_TOKEN vacio: el servicio queda ABIERTO a cualquiera "
        "que alcance este puerto.",
        flush=True,
    )
HILOS = int(os.environ.get("OMP_NUM_THREADS", "6"))

# Motor de inferencia: "torch" (RTF 2,19) u "openvino" (RTF 1,09).
MOTOR = os.environ.get("VIBEVOICE_MOTOR", "torch")
# "auto" (por defecto), "cpu", "cuda" o "mps". Ver elegir_dispositivo().
DISPOSITIVO_PEDIDO = os.environ.get("VIBEVOICE_DISPOSITIVO", "auto").strip().lower()
OV_CODIGO = os.environ.get("VIBEVOICE_OV_CODIGO", "")
IR_LM = os.environ.get("VIBEVOICE_IR_LM", "")
IR_CABEZA = os.environ.get("VIBEVOICE_IR_CABEZA", "")
IR_ACUSTICO = os.environ.get("VIBEVOICE_IR_ACUSTICO", "")


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
_candado_modelo = threading.Lock()
_bearer = HTTPBearer(auto_error=False)


class GeneracionCancelada(Exception):
    """Senal interna: el cliente se fue, aborta generate()."""


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
                MODELO_DIR, HILOS, IR_LM, IR_CABEZA, IR_ACUSTICO
            )
            modelo.set_ddpm_inference_steps(PASOS)
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

    modelo.set_ddpm_inference_steps(PASOS)
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


class StreamerCancelable:
    """Envuelve AsyncAudioStreamer anadiendo cancelacion cooperativa.

    put() corre en el HILO de generate(). Si el consumidor HTTP marco
    `cancelado`, lanzar aqui desmonta la pila de generate() y libera los 6
    nucleos al instante. Es el unico punto de corte que ofrece la API: el
    generate() del modelo no mira ningun flag externo, asi que sin esto un
    cliente que se va dejaria la CPU 20 s generando audio para nadie.
    """

    def __init__(self):
        from vibevoice.modular import AsyncAudioStreamer
        self.interno = AsyncAudioStreamer(batch_size=1, stop_signal=None)
        self.cancelado = False
        # generate() ya cerro el flujo por su cuenta. Lo que llegue despues
        # sobra, pero NO es que el cliente se haya ido.
        self.terminado = False

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
        self.interno.put(trozos, indices)

    def end(self, indices=None):
        self.terminado = True
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


def _sintetizar(texto, voz, cfg_scale, streamer, semilla=None, pasos=None):
    """Cuerpo sincrono de la sintesis; corre en un hilo del executor."""
    procesador = _estado["procesador"]
    try:
        with _candado_modelo:
            _ajustar_pasos(pasos)
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
        # end() es idempotente.
        if streamer is not None:
            streamer.end()


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
        """Texto que entro pero que el modelo no llego a decir."""
        with self._cond:
            return self._ids[self.consumidos:]

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


class ColaAudioSesion:
    """El `audio_streamer` que espera generate(), volcado a una cola asincrona.

    No cierra la cola de la sesion al terminar: una sesion larga puede encadenar
    varias generate() (al llegar al tope de caché) sobre el MISMO flujo de audio.
    """

    def __init__(self, lazo, cola):
        self.lazo, self.cola = lazo, cola
        self.cerrado = False
        self.trozos = 0

    def put(self, trozos, indices):
        # Tras end() lo que llegue sobra: generate() no sale del bucle de 6
        # latentes aunque el EOS salte a mitad, y esos ultimos son silencio.
        if self.cerrado:
            return
        for i, idx in enumerate(indices):
            if int(idx) != 0:
                continue
            self.trozos += 1
            self.lazo.call_soon_threadsafe(
                self.cola.put_nowait, trozos[i].detach().float().cpu())

    def end(self, indices=None):
        self.cerrado = True


class SesionViva:
    """Una generate() viva en su hilo, con una cola de texto por delante."""

    def __init__(self, nombre, voz, cfg_scale, semilla, pasos, lazo):
        self.nombre = nombre
        self.voz = voz
        self.cfg_scale = cfg_scale
        self.semilla = semilla
        self.pasos = pasos
        self.lazo = lazo
        self.cola = asyncio.Queue()
        self.visto = time.time()
        self.cerrada = False
        self.terminada = False
        self.escuchando = False
        self.generaciones = 0
        self.error = None
        self.eos_temprano = 0     # veces que el modelo callo con texto pendiente
        self._pendiente = []
        self._alimentador = None
        self._entre_locuciones = False   # parado, pero no dentro de generate()
        self._cond = threading.Condition()
        self._hilo = threading.Thread(target=self._correr, daemon=True,
                                      name=f"sesion-{nombre}")

    # ---- API ----
    def arrancar(self):
        self._hilo.start()

    def alimentar(self, texto: str) -> int:
        """Encola texto. Va al alimentador vivo si lo hay; si no, a la reserva
        para la generate() siguiente."""
        # Exactamente como lo tokeniza el procesador para una peticion normal
        # (process_input_with_cached_prompt: text.strip() + "\n"), asi que una
        # frase por sesion produce los mismos tokens que esa frase suelta.
        ids = _estado["procesador"].tokenizer.encode(
            texto.strip() + "\n", add_special_tokens=False)
        self.visto = time.time()
        with self._cond:
            if self.cerrada:
                raise HTTPException(409, f"sesion '{self.nombre}' ya cerrada")
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
            self.cerrada = True
            al = self._alimentador
            self._cond.notify_all()
        if al is not None:
            al.sellar()

    def estado(self) -> dict:
        al = self._alimentador
        return {
            "sesion": self.nombre, "voz": self.voz,
            "viva": not self.terminada, "cerrada": self.cerrada,
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
                        al_pausar=_candado_modelo.release,
                        al_reanudar=_candado_modelo.acquire,
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

    def _generar(self, al: TextoEnCurso):
        procesador = _estado["procesador"]
        base = prefijo_voz(self.voz)
        _candado_modelo.acquire()
        try:
            _ajustar_pasos(self.pasos)
            if self.semilla is not None:
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
                    audio_streamer=ColaAudioSesion(self.lazo, self.cola),
                )
        finally:
            _candado_modelo.release()


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
    semilla: Optional[int] = Field(None, ge=0, lt=2**31,
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
    semilla: Optional[int] = Field(None, ge=0, lt=2**31)
    pasos: Optional[int] = Field(None, ge=4, le=20)
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
        "voz_defecto": VOZ_DEFECTO,
        "rtf_esperado": RTF_MEDIDO,
        "ocupado": _candado.locked() or _candado_modelo.locked(),
        "auth": "bearer" if TOKEN else "abierta",
        "sesiones": {"activas": SESIONES_ACTIVAS,
                     "abiertas": sorted(_SESIONES),
                     "espera_texto_s": ESPERA_TEXTO,
                     "tope_cache": TOPE_CACHE},
    }


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
            streamer = StreamerCancelable()
            lazo = asyncio.get_running_loop()
            # generate() es bloqueante -> hilo del executor.
            tarea = lazo.run_in_executor(
                None, _sintetizar, pet.texto, pet.voz, pet.cfg_scale, streamer,
                pet.semilla, pet.pasos,
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
                       pet.semilla, pet.pasos, asyncio.get_running_loop())
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


def main() -> None:
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
