"""Motor de inferencia de VibeVoice sobre OpenVINO.

Sustituye tres piezas del modelo por sus grafos compilados: el backbone TTS
(Qwen2 de 20 capas, con la KV como variables internas de OV), la cabeza de
difusion y el decodificador acustico. El resto -LM de texto, conector,
clasificador de fin de frase- sigue en PyTorch con int8 dinamico.

POR QUE GANA, medido en un i7-8700T de 6 nucleos:

    componente        PyTorch int8    OpenVINO
    tts_lm              37-41 ms       24,5 ms
    cabeza                4,2 ms        2,9 ms
    acustico          165-168 ms       65,6 ms   <- el grueso
    ------------------------------------------------
    RTF end-to-end          2,19          1,09

El decodificador acustico es el 58% del tiempo y NO estaba limitado por
memoria: leer sus pesos a los 17,2 GB/s medidos costaria 20-40 ms, y tardaba
165. Esos ~130 ms sobrantes eran despacho de Python sobre decenas de
convoluciones pequenas, y eso es justo lo que elimina un grafo compilado.

DONDE VA EL TIEMPO HOY (GET /crono en voz_stream.py, 684 fotogramas de una
tanda real, sin solapar y con 6 hilos; un fotograma son 133,3 ms de audio):

    tts_lm      42,1 ms/fotograma  (34%)   2,11 pasadas de 20,0 ms
    acustico    55,6 ms            (45%)   1 pasada
    cabeza      15,6 ms            (13%)   6 pasadas de 2,60 ms
    resto       10,2 ms             (8%)   torch, conector, EOS, Python
    -------------------------------------
    generate   123,6 ms                    RTF 0,99

Las 2,11 pasadas de backbone por fotograma no son un error de cuenta: el
bucle de Microsoft evalua el backbone DOS VECES, con el contexto real y con
la secuencia negativa del CFG. Ver NEG_CADA en voz_stream.py.

Y ESTA CARGA YA NO ESTA LIMITADA POR MEMORIA. Era el hallazgo que gobernaba
todo cuando la RAM iba en canal unico; con los dos modulos puestos, las tres
pruebas que lo comprobarian salen que no:

  - una pasada de backbone de 2 tokens cuesta 1,77x la de 1 token (si mandara
    la lectura de pesos costaria ~1,0x, que son los mismos 156 MB)
  - el decodificador con 6 latentes por llamada NO gana sobre 1 por llamada
    (mismos 344 MB leidos una vez en lugar de seis)
  - bajar el decodificador de int8 (344 MB) a int4 (212 MB) gana un 7 %, no
    el 38 % que darian los bytes

Manda el COMPUTO, en seis nucleos a 2,4 GHz con AVX2 y sin VNNI. Lo que queda
por ganar esta en hacer menos trabajo, no en mover menos bytes.

TRAMPA IMPORTANTE
OMP_PLACES=cores + OMP_PROC_BIND=close acelera PyTorch un 3% pero RALENTIZA
esto un 118% (89 ms/llamada sin anclaje, 195 con el). El modulo NixOS lo
desactiva cuando el motor es openvino; si ejecutas a mano, no lo pongas.

Orden de carga pensado para el techo de 5 GB de RAM:
  fp32 completo (pico ~4,1 GB) -> soltar el tts_lm de torch (-2,3 GB con su
  embed muerto) -> int8 en language_model -> compilar los IR (+0,3 GB).
"""
import gc
import time
import weakref

import numpy as np
import torch

CRONO = {"tts_lm": [0.0, 0], "cabeza": [0.0, 0], "acustico": [0.0, 0],
         # pasadas del backbone que el agrupado de la rama incondicional se
         # ahorro: cuentan el tiempo de acumular, no el de inferir
         "tts_lm_aplazado": [0.0, 0]}


def _referencia(obj):
    """Callable que devuelve `obj`, debil si se puede. None -> None.

    Debil porque quien lo usa (AcusticoOV) apunta a un objeto de generate() y no
    debe alargarle la vida. El repliegue fuerte es por si algun dia el objeto no
    admite weakref: mejor retener 711 KB de mas que tumbar el motor entero.
    """
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda o=obj: o


class SalidaLM:
    __slots__ = ("last_hidden_state", "past_key_values", "attentions")

    def __init__(self, h, cache):
        self.last_hidden_state = h
        self.past_key_values = cache
        self.attentions = None


class CacheOV:
    """Viaja por model_kwargs; el estado real vive en el InferRequest."""

    __slots__ = ("peticion", "longitud", "_positivo", "_pendientes", "_ultimo")

    def __init__(self, peticion, longitud):
        self.peticion = peticion
        self.longitud = longitud
        # Marca que pone voz_stream.py: True en la rama CONDICIONAL (la unica
        # que llega a recibir tokens de texto). Se usa para agrupar solo la
        # otra; ver TtsLmOV.forward.
        self._positivo = False
        self._pendientes = []       # (embeds, position_ids) sin procesar aun
        self._ultimo = None         # hidden de la ultima pasada REAL

    def get_seq_length(self, layer_idx=0):
        return self.longitud


def _tensores_cache(cache, n_capas):
    """Extrae (k,v) por capa de lo que venga en el .pt de la voz."""
    if hasattr(cache, "key_cache") and len(getattr(cache, "key_cache", [])) == n_capas:
        return [(cache.key_cache[i], cache.value_cache[i]) for i in range(n_capas)]
    if hasattr(cache, "layers"):
        return [(c.keys if hasattr(c, "keys") else c.key_cache,
                 c.values if hasattr(c, "values") else c.value_cache) for c in cache.layers]
    return [(cache[i][0], cache[i][1]) for i in range(n_capas)]  # tupla legada


class TtsLmOV(torch.nn.Module):
    """Reemplazo de Qwen2Model(20 capas): IR con estado de OpenVINO.

    AGRUPADO DE LA RAMA INCONDICIONAL (`neg_cada`)
    El bucle de Microsoft llama a este backbone DOS VECES por fotograma
    acustico: una con el contexto real (condicional) y otra con la secuencia
    negativa, que arranca de un solo <|image_pad|> y solo recibe los latentes
    ya generados -- nunca el texto. Las dos salidas son las que alimentan la
    guia sin clasificador de la difusion.

    Esa segunda pasada es la mitad del backbone y, medido en la VM, el
    backbone es el 77 % del reloj de una generacion. No es una rama barata:
    cuesta lo mismo que la buena, porque el coste de este modelo son sus
    pesos, no su secuencia.

    `neg_cada > 1` no la SALTA -- eso desincronizaria su cache --, la AGRUPA:
    los embeds de los fotogramas intermedios se acumulan y entran de golpe en
    una sola pasada de longitud N. La cache negativa queda BIT A BIT como
    estaba (la atencion es causal: procesar [x1,x2] de una vez da lo mismo que
    x1 y luego x2), y una pasada de N tokens cuesta practicamente lo mismo que
    una de 1 porque hay que leer los mismos 156 MB de pesos.

    LO QUE SI CAMBIA es que la condicion negativa que ve la difusion se queda
    hasta N-1 fotogramas vieja. Es la rama que NO mira el texto, asi que su
    hidden se mueve despacio; cuanto cuesta eso en fidelidad esta medido en
    voz_stream.py (agrupar_rama_negativa).
    """

    def __init__(self, ruta_xml, hilos, n_capas=20, neg_cada=1):
        super().__init__()
        import openvino as ov
        self._ov = ov
        core = ov.Core()
        self.comp = core.compile_model(ruta_xml, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.n_capas = n_capas
        self.neg_cada = max(1, int(neg_cada))
        self.device = torch.device("cpu")

    def _arrancar_flujo(self, cache_torch):
        import re
        pet = self.comp.create_infer_request()
        pares = _tensores_cache(cache_torch, self.n_capas)
        longitud = pares[0][0].shape[2]
        for est in pet.query_state():
            m = re.search(r"past_key_values\.(\d+)\.(key|value)", est.name)
            i, cual = int(m.group(1)), m.group(2)
            t = pares[i][0] if cual == "key" else pares[i][1]
            est.state = self._ov.Tensor(np.ascontiguousarray(t.detach().float().numpy()))
        return CacheOV(pet, longitud)

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None, **kw):
        ini = time.perf_counter()
        cache = past_key_values
        estrenando = not isinstance(cache, CacheOV)
        if estrenando:
            cache = self._arrancar_flujo(cache)
        S = inputs_embeds.shape[1]
        if position_ids is not None:
            pos = position_ids.detach().numpy().astype(np.int64)[:, -S:]
        else:
            pos = np.arange(cache.longitud, cache.longitud + S, dtype=np.int64)[None]
        emb = np.ascontiguousarray(inputs_embeds.detach().float().numpy())

        # ---- rama incondicional agrupada ----
        # Solo cuando ya hay un hidden anterior que devolver: la primera
        # pasada de un flujo no se puede aplazar, no habria que responder.
        if (self.neg_cada > 1 and not cache._positivo and not estrenando
                and cache._ultimo is not None):
            cache._pendientes.append((emb, pos))
            # La longitud avanza AUNQUE no se ejecute: es lo que mantiene
            # sincronizados los position_ids que prepare_inputs_for_generation
            # calcula fuera, y por tanto lo que hace que el agrupado sea fiel.
            cache.longitud += S
            if len(cache._pendientes) < self.neg_cada:
                CRONO["tts_lm_aplazado"][0] += time.perf_counter() - ini
                CRONO["tts_lm_aplazado"][1] += 1
                return SalidaLM(cache._ultimo, cache)
            emb = np.ascontiguousarray(
                np.concatenate([e for e, _ in cache._pendientes], axis=1))
            pos = np.ascontiguousarray(
                np.concatenate([p for _, p in cache._pendientes], axis=1))
            cache._pendientes.clear()
            S = 0                       # la longitud ya se conto al acumular

        res = cache.peticion.infer(
            {"inputs_embeds": emb, "position_ids": pos},
            share_inputs=True, share_outputs=True)
        h = torch.from_numpy(np.array(res[self.comp.output("hidden")]))
        cache.longitud += S
        cache._ultimo = h
        CRONO["tts_lm"][0] += time.perf_counter() - ini
        CRONO["tts_lm"][1] += 1
        return SalidaLM(h, cache)


class CabezaOV(torch.nn.Module):
    """Reemplazo de VibeVoiceDiffusionHead: IR estatico [2,64]/[2]/[2,896]."""

    def __init__(self, ruta_xml, hilos):
        super().__init__()
        import openvino as ov
        core = ov.Core()
        self.comp = core.compile_model(ruta_xml, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.pet = self.comp.create_infer_request()
        self.device = torch.device("cpu")

    def forward(self, noisy, timesteps, condition=None):
        ini = time.perf_counter()
        res = self.pet.infer([noisy.detach().float().numpy(),
                              timesteps.detach().float().numpy(),
                              condition.detach().float().numpy()],
                             share_inputs=True, share_outputs=True)
        salida = torch.from_numpy(np.array(res[self.comp.output(0)]))
        CRONO["cabeza"][0] += time.perf_counter() - ini
        CRONO["cabeza"][1] += 1
        return salida


class AcusticoOV:
    """Reemplazo de acoustic_tokenizer.decode: IR con estado, formas fijas.

    El estado interno de OV sustituye a VibeVoiceTokenizerStreamingCache: las
    34 colas de las convoluciones causales, que en el original viven en el
    objeto `cache` que generate() pasa en cada llamada a decode().

    EL ESTADO ES DE UNA CORRIENTE, NO DEL PROCESO  (esto era un fallo)
    Aqui solo hay UN InferRequest, y su estado es unico para todo el proceso.
    Al principio se ponia a cero una sola vez, en el constructor, y a partir de
    ahi cada sintesis heredaba las colas que dejo la ANTERIOR. Efecto medido en
    la VM (motor openvino, mismo texto, misma voz, misma semilla 11, 6 pasos):

        A1  827c4228...      <- cada una arranca donde acabo la de antes
        A2  675d3c46...
        A3  18df1982...
        A4  f981fc55...      <- y a partir de aqui converge a un punto fijo
        A5  f981fc55...
        A6  f981fc55...
        B   (otro texto)
        A7  3b33e3fd...      <- B ensucia el estado y vuelve a empezar la deriva

    Los 121600 bytes eran SIEMPRE los mismos: el ruido de la difusion si estaba
    bien sembrado -- torch.manual_seed() lo gobierna, porque sample_speech_tokens
    sigue siendo la de torch --, y lo que cambiaba era solo el contexto del que
    partia este decodificador. Convergir a un punto fijo es justo la firma de una
    memoria convolucional que se desvanece; ruido sin semilla no convergeria
    nunca.

    EL ARREGLO: EL ESTADO SIGUE AL `cache` QUE LO PIDE
    generate() crea un VibeVoiceTokenizerStreamingCache NUEVO por llamada
    (modeling_vibevoice_streaming_inference.py, linea 661) y lo pasa en cada
    decode(). Ese objeto es, por tanto, la identidad de la corriente. Aqui se
    mira: si el que llega no es el dueño del estado que hay puesto, se le hace
    una foto al estado, se le cuelga al dueño anterior, y se carga la del nuevo
    (ceros si nunca ha hablado). Asi cada generate() tiene su propio hilo de
    estado acustico, igual que en torch, sin tocar upstream.

    POR QUE ASI Y NO RESETEANDO AL EMPEZAR CADA SINTESIS
    Un reset por sintesis arregla /tts/stream, pero NO las sesiones vivas: una
    sesion suelta el candado del modelo en cada pausa y otra se cuela en mitad de
    su locucion, con su propia generate() y su propio cache. Con reset a secas la
    primera reanudaria con el estado de la segunda. Siguiendo al cache, cada una
    recupera EL SUYO, que es lo que hace que dos sesiones concurrentes con la
    misma semilla den el mismo audio -- la misma garantia que el RNG por sesion
    de SesionViva._pausar en voz_stream.py.

    POR QUE UNA FOTO Y NO UN InferRequest POR CORRIENTE
    Un InferRequest por sintesis se llevaria tambien los tensores intermedios del
    grafo, y esto corre en una VM de 5 GB. La foto son 711 KB y cuesta 1,4 ms
    medidos (foto + reposicion), contra 52 ms de UN solo decode de los ~40 que
    lleva una frase. Ademas la foto es fiel bit a bit: comprobado que reponerla y
    seguir da exactamente el mismo audio que no haber parado.

    CEBADO
    El decodificador es causal: en frio no tiene contexto por la izquierda y la
    primera muestra sale con un salto. Es el mismo problema que cebar_decoder_
    acustico() resuelve en el camino torch, y que aqui no se aplicaba porque
    cargar_modelo() no llega a esa parte cuando el motor es openvino. Medido con
    este IR: primera muestra -2,88e-05 en frio, -3,52e-07 tras cebar con un
    latente nulo. Cuesta un decode (52 ms) por sintesis, no por fotograma.
    """

    def __init__(self, ruta_xml, hilos):
        import re
        import openvino as ov
        from decoder_manual import formas_estado_lista
        self._ov = ov
        core = ov.Core()
        self.comp = core.compile_model(ruta_xml, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.pet = self.comp.create_infer_request()
        formas = formas_estado_lista()
        self._ceros = {}
        for est in self.pet.query_state():
            c, l = formas[int(re.search(r"est\.(\d+)\.", est.name).group(1))]
            self._ceros[est.name] = np.zeros((1, c, l), dtype=np.float32)
        # weakref: el dueño es un objeto de generate(), y cuando esa generate()
        # muere su estado sobra. Con una referencia normal lo mantendriamos vivo
        # -- a el y a sus 711 KB -- hasta la sintesis siguiente.
        self._duenno = None
        self.nueva_sesion()

    # ---- estado: leerlo, ponerlo, y cambiar de corriente ----
    def _foto(self):
        # copy=True de verdad: .data es una VISTA de la memoria de OV, que la
        # siguiente infer() sobrescribe.
        return {est.name: np.array(est.state.data, copy=True)
                for est in self.pet.query_state()}

    def _poner(self, estados):
        for est in self.pet.query_state():
            # .copy() para no entregarle a OV la misma memoria que guardamos:
            # si la compartiera, la foto dejaria de ser una foto.
            est.state = self._ov.Tensor(
                np.ascontiguousarray(estados[est.name], dtype=np.float32).copy())

    def nueva_sesion(self):
        """Estado a cero == cache vacia del original. Suelta al dueño actual."""
        self.pet.reset_state()
        self._poner(self._ceros)
        self._duenno = None

    def _cambiar_a(self, cache):
        """Deja puesto el estado de `cache`. Devuelve True si estrena (venia
        de ceros y por tanto toca cebar)."""
        viejo = self._duenno() if self._duenno is not None else None
        if cache is not None and viejo is cache:
            return False                      # sigue la misma generate()
        if viejo is not None:
            viejo._estado_ov = self._foto()
        # cache=None es use_cache=False del original: cada llamada, en frio.
        guardado = getattr(cache, "_estado_ov", None) if cache is not None else None
        if guardado is None:
            self.pet.reset_state()
            self._poner(self._ceros)
        else:
            self._poner(guardado)
        self._duenno = _referencia(cache)
        return guardado is None

    def decode(self, latents, cache=None, sample_indices=None, use_cache=True, debug=False):
        ini = time.perf_counter()
        lat = latents.detach().float()
        if lat.shape[1] != 64:          # [1,1,64] -> [1,64,1]
            lat = lat.permute(0, 2, 1)
        lat = np.ascontiguousarray(lat.numpy())
        if self._cambiar_a(cache):
            # cebado: un fotograma de silencio para que la primera muestra real
            # no salte desde la nada. Se tira la salida.
            self.pet.infer({"lat": np.zeros_like(lat)},
                           share_inputs=True, share_outputs=True)
        res = self.pet.infer({"lat": lat}, share_inputs=True, share_outputs=True)
        salida = torch.from_numpy(np.array(res[self.comp.output("audio")]))
        CRONO["acustico"][0] += time.perf_counter() - ini
        CRONO["acustico"][1] += 1
        return salida


def cargar(modelo_path, hilos, ir_lm, ir_cabeza, ir_acustico=None,
           hilos_acustico=None, neg_cada=1):
    """hilos_acustico separa el presupuesto del decodificador del resto.

    Solo tiene sentido cuando voz_stream.py lo solapa en otro hilo: entonces el
    decodificador y el bucle (tts_lm + cabeza) corren A LA VEZ, y darles a los
    dos `hilos` enteros es pedir el doble de nucleos de los que hay. Aqui el
    reparto SI es exacto, porque INFERENCE_NUM_THREADS es por modelo compilado
    -- en el camino torch no se puede, que ahi el pool intra-op es uno para
    todo el proceso.

    None = el comportamiento de antes: los mismos hilos para todo.
    """
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    procesador = VibeVoiceStreamingProcessor.from_pretrained(modelo_path)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        modelo_path, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="sdpa")
    modelo.eval()

    # 1) fuera el backbone torch (y su embed muerto de 0,5 GB) ANTES de nada
    modelo.model.tts_language_model = None
    if ir_acustico:
        # el decoder torch (1,4 GB fp32) y el encoder (nunca se usa) sobran
        modelo.model.acoustic_tokenizer.decoder = None
        modelo.model.acoustic_tokenizer.encoder = None
        # sin parametros, .device (transformers) revienta: ancla minima
        modelo.model.acoustic_tokenizer._ancla = torch.nn.Parameter(torch.zeros(1))
    gc.collect()

    # 2) int8 dinamico en el LM de texto (como en produccion)
    torch.ao.quantization.quantize_dynamic(
        modelo.model.language_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    gc.collect()

    # 3) enchufar OpenVINO
    # El bucle se queda con lo que no se lleve el decodificador. Minimo 1: un
    # reparto mal puesto no debe dejar el camino critico sin hilos.
    hilos_bucle = max(1, hilos - hilos_acustico) if hilos_acustico else hilos
    modelo.model.tts_language_model = TtsLmOV(ir_lm, hilos_bucle, neg_cada=neg_cada)
    if ir_cabeza:
        modelo.model.prediction_head = CabezaOV(ir_cabeza, hilos_bucle)
    else:
        torch.ao.quantization.quantize_dynamic(
            modelo.model.prediction_head, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    if ir_acustico:
        acustico = AcusticoOV(ir_acustico, hilos_acustico or hilos)
        modelo.model.acoustic_tokenizer.decode = acustico.decode
        modelo._acustico_ov = acustico
    gc.collect()
    return procesador, modelo


