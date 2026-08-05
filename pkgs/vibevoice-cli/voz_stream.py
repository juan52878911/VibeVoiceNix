"""Servidor de TTS con respuesta HTTP en streaming.

    POST /tts/stream  {"texto": "..."}  ->  audio/wav troceado

El cliente empieza a oír ~0,2 s después de pedirlo, mientras el resto se
genera. Verificado: los trozos emitidos son bit a bit identicos al audio
completo (mismo md5 que la generacion no-streaming).

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
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
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

    def put(self, trozos, indices):
        if self.cancelado:
            raise GeneracionCancelada()
        self.interno.put(trozos, indices)

    def end(self, indices=None):
        self.interno.end(indices)

    def flujo(self):
        return self.interno.get_stream(0)


def _sintetizar(texto, voz, cfg_scale, streamer):
    """Cuerpo sincrono de la sintesis; corre en un hilo del executor."""
    procesador = _estado["procesador"]
    base = prefijo_voz(voz)
    # deepcopy DOBLE: ni el procesador ni generate() tocan el pristino.
    entradas = procesador.process_input_with_cached_prompt(
        text=texto, cached_prompt=copy.deepcopy(base),
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    if EN_GPU:
        entradas = a_dispositivo(entradas)
    try:
        with torch.no_grad():
            _estado["modelo"].generate(
                **entradas,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=procesador.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                # El streamer ya entrega cada trozo segun sale; sin esto
                # generate() ADEMAS acumula la sintesis entera y la concatena
                # al final, para devolver algo que aqui se ignora.
                return_speech=False,
                all_prefilled_outputs=copy.deepcopy(base),
                audio_streamer=streamer,
            )
    except GeneracionCancelada:
        pass  # cliente desconectado: salida limpia, sin ruido en el log
    finally:
        # Pase lo que pase, cierra la cola: sin esto un fallo dentro de
        # generate() dejaria al consumidor esperando un trozo que no llega.
        # end() es idempotente.
        if streamer is not None:
            streamer.end()


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
        "ocupado": _candado.locked(),
        "auth": "bearer" if TOKEN else "abierta",
    }


@app.post("/tts/stream")
async def tts_stream(pet: PeticionTTS, _=Depends(autorizar)) -> StreamingResponse:
    prefijo_voz(pet.voz)  # valida ANTES de enviar cabeceras, para dar un 404 limpio

    async def generador():
        # El candado se toma DENTRO del generador: si hay otra sintesis en
        # curso, esta espera su turno sin bloquear el bucle de eventos.
        async with _candado:
            streamer = StreamerCancelable()
            lazo = asyncio.get_running_loop()
            # generate() es bloqueante -> hilo del executor.
            tarea = lazo.run_in_executor(
                None, _sintetizar, pet.texto, pet.voz, pet.cfg_scale, streamer,
            )
            try:
                yield cabecera_wav_flujo()
                async for trozo in streamer.flujo():
                    yield a_pcm16(trozo)  # ~133 ms de audio por trozo
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
            "X-Ritmo-Hz": str(RITMO),
            "X-RTF-Esperado": str(RTF_MEDIDO),
            "Content-Disposition": 'inline; filename="voz.wav"',
        },
    )


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
