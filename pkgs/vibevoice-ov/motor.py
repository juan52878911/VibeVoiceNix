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

DONDE VA EL TIEMPO HOY (GET /crono de produccion, 13-09-2026, 1008 fotogramas,
sin solapar y con 6 hilos; un fotograma son 133,3 ms de audio):

    tts_lm      43,1 ms/fotograma  (39%)   2,11 pasadas de ~20 ms
    acustico    41,3 ms            (37%)   1 pasada (antes 55,6-76,3: ver SubidaTr)
    cabeza      15,9 ms            (14%)   6 pasadas de 2,6 ms
    resto       11,1 ms            (10%)   torch, conector, EOS, Python
    -------------------------------------
    generate   111,4 ms                    RTF 0,98 de extremo a extremo

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

Orden de carga pensado para el techo de 5 GB de RAM (ver _construir_parcial):
  modelo en `meta` -> soltar LM TTS, decodificador, codificador y cabeza de
  torch SIN haberlos leido -> leer del safetensors solo lo vivo -> int8 en
  language_model -> compilar los IR. Antes se materializaba el fp32 entero y
  el pico de la carga era 4,39 GB de VmHWM (medido en la VM).
"""
import gc
import os
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
    """Reemplazo de VibeVoiceDiffusionHead: IR estatico [2,64]/[2]/[2,896].

    Se compila en la PRIMERA llamada, no al cargar. Con la difusion en un grafo
    (DifusionOV) solo se la llama si una peticion pide otros pasos, y compilarla
    siempre eran 42 MB de pesos y su repack sin usarse nunca.
    """

    def __init__(self, ruta_xml, hilos):
        super().__init__()
        self._ruta, self._hilos = ruta_xml, hilos
        self.comp = self.pet = None
        self.device = torch.device("cpu")

    def _compilar(self):
        import openvino as ov
        self.comp = ov.Core().compile_model(self._ruta, "CPU",
                                            {"INFERENCE_NUM_THREADS": self._hilos, "NUM_STREAMS": 1,
                                             "PERFORMANCE_HINT": "LATENCY"})
        self.pet = self.comp.create_infer_request()

    def forward(self, noisy, timesteps, condition=None):
        if self.pet is None:
            self._compilar()
        ini = time.perf_counter()
        res = self.pet.infer([noisy.detach().float().numpy(),
                              timesteps.detach().float().numpy(),
                              condition.detach().float().numpy()],
                             share_inputs=True, share_outputs=True)
        salida = torch.from_numpy(np.array(res[self.comp.output(0)]))
        CRONO["cabeza"][0] += time.perf_counter() - ini
        CRONO["cabeza"][1] += 1
        return salida


class DifusionOV:
    """El bucle de difusion entero de un fotograma en UNA llamada (convertir_difusion.py).

    Sustituye las `pasos` llamadas a CabezaOV y el solver en torch. Solo vale para los
    pasos con los que se convirtio, que van en el nombre del fichero (difusion_p6_int8.xml):
    con otros pasos voz_stream.py sigue por el camino de siempre. El ruido lo pone quien
    llama, con torch.randn, para que la semilla consuma lo mismo que antes.
    """

    def __init__(self, ruta_xml, hilos):
        import re
        import openvino as ov
        core = ov.Core()
        self.comp = core.compile_model(ruta_xml, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.pet = self.comp.create_infer_request()
        m = re.search(r"_p(\d+)_", ruta_xml)
        self.pasos = int(m.group(1)) if m else None

    def __call__(self, condition, speech, cfg_scale, freno):
        ini = time.perf_counter()
        res = self.pet.infer([condition.detach().float().numpy(), speech.detach().float().numpy(),
                              np.array(cfg_scale, dtype=np.float32), np.array(freno, dtype=np.float32)],
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
        modelo = core.read_model(ruta_xml)
        formas = formas_estado_lista()
        # ESTADO EXPLICITO PARA LOS IR SIN ESTADO (decoder_mm_*). Las 34 colas
        # entran y salen como tensores (lat, est.0..33 / audio, est.0..33) y
        # las lleva esta clase en una lista de arrays; los IR con estado de
        # OpenVINO (decoder_estado_*) siguen con sus variables internas.
        #
        # No es gusto: con las subidas por productos de matrices (ver SubidaTr
        # en decoder_manual.py), el estado de OpenVINO sobre un grafo leido de
        # fichero realimenta mal. Medido frente a torch con los pesos reales:
        # estado aplicado en memoria 118 dB de SNR; en cuanto el grafo pasa por
        # fichero, -2,9 dB desde el primer fotograma -- guardado con estado o
        # puesto al releer, emparejado por nombre o por posicion, en fp16 o en
        # float32 --; el mismo fichero con el estado explicito, 71 dB, lo mismo
        # que el decodificador viejo. La causa no esta identificada.
        #
        # Lo que cuesta: copiar las colas que devuelve cada llamada (711 KB).
        self._explicito = not any(op.get_type_name() == "ReadValue" for op in modelo.get_ops())
        self.comp = core.compile_model(modelo, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.pet = self.comp.create_infer_request()
        if self._explicito:
            self._nombres_est = ["est.%d.in" % i for i in range(len(formas))]
            self._salidas_est = [self.comp.output("est.%d.out" % i) for i in range(len(formas))]
            for i, (c, l) in enumerate(formas):
                for puerto in (self.comp.input(self._nombres_est[i]), self._salidas_est[i]):
                    assert list(puerto.get_shape()) == [1, c, l], (i, puerto.get_shape(), (c, l))
            self._ceros = [np.zeros((1, c, l), dtype=np.float32) for c, l in formas]
            self._estado = list(self._ceros)
        else:
            self._ceros = {}
            for est in self.pet.query_state():
                c, l = formas[int(re.search(r"est\.(\d+)\.", est.name).group(1))]
                self._ceros[est.name] = np.zeros((1, c, l), dtype=np.float32)
        # weakref: el dueño es un objeto de generate(), y cuando esa generate()
        # muere su estado sobra. Con una referencia normal lo mantendriamos vivo
        # -- a el y a sus 711 KB -- hasta la sintesis siguiente.
        self._duenno = None
        # El estado que deja el cebado. Se calcula en el PRIMER arranque y a
        # partir de ahi se repone sin llamar al modelo: el cebado es siempre el
        # mismo -- estado a cero y un latente nulo --, asi que su resultado
        # tambien. Ahorra un decode (~40 ms) al principio de cada sintesis, que
        # es justo donde se espera al primer sonido, y el audio sale identico
        # bit a bit (la foto es fiel, ver _foto).
        self._cebado = None
        self.nueva_sesion()

    # ---- estado: leerlo, ponerlo, y cambiar de corriente ----
    def _foto(self):
        if self._explicito:
            # Los arrays de la lista no se tocan nunca en su sitio: cada
            # llamada deja una lista NUEVA de copias. Basta con la lista.
            return list(self._estado)
        # copy=True de verdad: .data es una VISTA de la memoria de OV, que la
        # siguiente infer() sobrescribe.
        return {est.name: np.array(est.state.data, copy=True)
                for est in self.pet.query_state()}

    def _poner(self, estados):
        if self._explicito:
            self._estado = list(estados)
            return
        for est in self.pet.query_state():
            # .copy() para no entregarle a OV la misma memoria que guardamos:
            # si la compartiera, la foto dejaria de ser una foto.
            est.state = self._ov.Tensor(
                np.ascontiguousarray(estados[est.name], dtype=np.float32).copy())

    def _a_cero(self):
        if not self._explicito:
            self.pet.reset_state()
        self._poner(self._ceros)

    def nueva_sesion(self):
        """Estado a cero == cache vacia del original. Suelta al dueño actual."""
        self._a_cero()
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
            self._a_cero()
        else:
            self._poner(guardado)
        self._duenno = _referencia(cache)
        return guardado is None

    def _inferir(self, lat):
        """Un fotograma: devuelve el audio y deja el estado avanzado."""
        if not self._explicito:
            res = self.pet.infer({"lat": lat}, share_inputs=True, share_outputs=True)
            return np.array(res[self.comp.output("audio")])
        entradas = {"lat": lat}
        entradas.update(zip(self._nombres_est, self._estado))
        res = self.pet.infer(entradas, share_inputs=True, share_outputs=True)
        # copias: share_outputs deja vistas que la siguiente infer() pisa
        self._estado = [np.array(res[s], copy=True) for s in self._salidas_est]
        return np.array(res[self.comp.output("audio")])

    def decode(self, latents, cache=None, sample_indices=None, use_cache=True, debug=False):
        ini = time.perf_counter()
        lat = latents.detach().float()
        if lat.shape[1] != 64:          # [1,1,64] -> [1,64,1]
            lat = lat.permute(0, 2, 1)
        lat = np.ascontiguousarray(lat.numpy())
        if self._cambiar_a(cache):
            # cebado: un fotograma de silencio para que la primera muestra real
            # no salte desde la nada. Se tira la salida. Solo se calcula una vez
            # (ver self._cebado).
            if self._cebado is None:
                self._inferir(np.zeros_like(lat))
                self._cebado = self._foto()
            else:
                self._poner(self._cebado)
        salida = torch.from_numpy(self._inferir(lat))
        CRONO["acustico"][0] += time.perf_counter() - ini
        CRONO["acustico"][1] += 1
        return salida


def _memoria_mb():
    """(RSS, VmHWM) del proceso en MB. (0, 0) donde no hay /proc."""
    try:
        with open("/proc/self/status") as f:
            campos = dict(linea.split(":", 1) for linea in f if linea.startswith(("VmRSS", "VmHWM")))
        return int(campos["VmRSS"].split()[0]) // 1024, int(campos["VmHWM"].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        return 0, 0


def _marca(paso):
    rss, pico = _memoria_mb()
    print(f"[carga] {paso}: {rss} MB residentes, pico {pico} MB", flush=True)


class EmbeddingMmap(torch.nn.Module):
    """La tabla de embeddings del LM de texto, leida del safetensors por mmap.

    Son 151936 x 896 en bf16 en el fichero. from_pretrained(float32) la
    materializaba entera en fp32: 545 MB anonimos para consultar unas decenas de
    filas por ventana de texto. Aqui la tabla son paginas del fichero -- se
    comparten con la cache de disco y el kernel las reclama sin swap -- y solo se
    pasan a fp32 las filas consultadas.

    ES BIT A BIT LO MISMO: bf16 -> fp32 es exacto, y consultar filas y luego
    convertirlas da los mismos bytes que convertir la tabla y luego consultar.
    """

    def __init__(self, ruta_st, clave):
        super().__init__()
        import json
        import mmap
        import struct
        import warnings
        with open(ruta_st, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            info = json.loads(fh.read(n))[clave]
            self._mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if info["dtype"] != "BF16":
            raise ValueError(f"{clave}: se esperaba BF16 y el fichero trae {info['dtype']}")
        ini, fin = info["data_offsets"]
        with warnings.catch_warnings():
            # frombuffer avisa de que el buffer no admite escritura: es a
            # proposito, una tabla de consulta no se escribe
            warnings.simplefilter("ignore", UserWarning)
            plano = torch.frombuffer(self._mm, dtype=torch.bfloat16,
                                     count=(fin - ini) // 2, offset=8 + n + ini)
        self._tabla = plano.view(*info["shape"])
        self.num_embeddings, self.embedding_dim = info["shape"]

    def forward(self, ids):
        return torch.nn.functional.embedding(ids, self._tabla).float()


CLAVE_EMBEDDINGS = "model.language_model.embed_tokens.weight"


def _construir_parcial(modelo_path, soltar_acustico, soltar_cabeza):
    """El modelo torch con solo los pesos que se usan, sin materializar el resto.

    from_pretrained(float32) construia el checkpoint ENTERO en fp32 (~4 GB; el
    pico de 4,39 GB de VmHWM medido en la VM, que obligaba al swap) para soltar
    acto seguido el LM TTS, el decodificador, el codificador y la cabeza, que
    sustituye OpenVINO. Aqui el modelo se construye en `meta` (sin memoria), esas
    piezas se sueltan ANTES de leer nada, y del safetensors se lee solo lo que
    queda vivo, pasado a fp32 como hacia from_pretrained. Los embeddings del LM
    de texto van por mmap (EmbeddingMmap).

    Mismos tensores, mismos dtypes, mismos buffers: lo comprueba la huella de
    scripts/lab_fase0.py contra la carga de antes.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference as Clase,
    )

    config = Clase.config_class.from_pretrained(modelo_path)
    # Lo que hace from_pretrained con dtype=float32 antes de construir: el
    # config y sus subconfigs pasan a float32, y el __init__ lo lee para sus
    # .to(dtype).
    config.dtype = torch.float32
    for sub in getattr(config, "sub_configs", {}):
        if getattr(config, sub, None) is not None:
            getattr(config, sub).dtype = torch.float32
    with init_empty_weights():          # parametros en meta; los buffers, de verdad
        modelo = Clase._from_config(config, dtype=torch.float32, attn_implementation="sdpa")
    modelo.eval()

    m = modelo.model
    m.tts_language_model = None
    if soltar_acustico:
        m.acoustic_tokenizer.decoder = None
        m.acoustic_tokenizer.encoder = None
        # sin parametros, .device (transformers) revienta: ancla minima
        m.acoustic_tokenizer._ancla = torch.nn.Parameter(torch.zeros(1))
    if soltar_cabeza:
        m.prediction_head = None

    ruta = os.path.join(modelo_path, "model.safetensors")
    vivos = dict(modelo.named_parameters())
    vivos.update(modelo.named_buffers())
    sd = {}
    with safe_open(ruta, framework="pt") as f:
        claves = set(f.keys())
        for nombre in vivos:
            if nombre in claves and nombre != CLAVE_EMBEDDINGS:
                t = f.get_tensor(nombre)
                sd[nombre] = t.float() if t.is_floating_point() else t
    faltan = [n for n, t in vivos.items()
              if t.is_meta and n not in sd and n != CLAVE_EMBEDDINGS]
    if faltan:
        raise RuntimeError(f"pesos vivos que no estan en el checkpoint: {faltan[:5]}")
    modelo.load_state_dict(sd, strict=False, assign=True)
    m.language_model.embed_tokens = EmbeddingMmap(ruta, CLAVE_EMBEDDINGS)
    return modelo


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
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    procesador = VibeVoiceStreamingProcessor.from_pretrained(modelo_path)
    # 1) solo lo que no sustituye OpenVINO, y sin pasar por el fp32 entero
    modelo = _construir_parcial(modelo_path, soltar_acustico=bool(ir_acustico),
                                soltar_cabeza=bool(ir_cabeza))
    gc.collect()
    _marca("pesos torch leidos")

    # 2) int8 dinamico en el LM de texto (como en produccion)
    torch.ao.quantization.quantize_dynamic(
        modelo.model.language_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    gc.collect()
    _marca("LM de texto en int8")

    # 3) enchufar OpenVINO
    # El bucle se queda con lo que no se lleve el decodificador. Minimo 1: un
    # reparto mal puesto no debe dejar el camino critico sin hilos.
    hilos_bucle = max(1, hilos - hilos_acustico) if hilos_acustico else hilos
    modelo.model.tts_language_model = TtsLmOV(ir_lm, hilos_bucle, neg_cada=neg_cada)
    _marca("LM TTS compilado")
    if ir_cabeza:
        modelo.model.prediction_head = CabezaOV(ir_cabeza, hilos_bucle)
    else:
        torch.ao.quantization.quantize_dynamic(
            modelo.model.prediction_head, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    if ir_acustico:
        acustico = AcusticoOV(ir_acustico, hilos_acustico or hilos)
        modelo.model.acoustic_tokenizer.decode = acustico.decode
        modelo._acustico_ov = acustico
        _marca("decodificador compilado")
    gc.collect()
    return procesador, modelo


