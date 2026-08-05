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

import numpy as np
import torch

CRONO = {"tts_lm": [0.0, 0], "cabeza": [0.0, 0], "acustico": [0.0, 0]}


class SalidaLM:
    __slots__ = ("last_hidden_state", "past_key_values", "attentions")

    def __init__(self, h, cache):
        self.last_hidden_state = h
        self.past_key_values = cache
        self.attentions = None


class CacheOV:
    """Viaja por model_kwargs; el estado real vive en el InferRequest."""

    def __init__(self, peticion, longitud):
        self.peticion = peticion
        self.longitud = longitud

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
    """Reemplazo de Qwen2Model(20 capas): IR con estado de OpenVINO."""

    def __init__(self, ruta_xml, hilos, n_capas=20):
        super().__init__()
        import openvino as ov
        self._ov = ov
        core = ov.Core()
        self.comp = core.compile_model(ruta_xml, "CPU",
                                       {"INFERENCE_NUM_THREADS": hilos, "NUM_STREAMS": 1,
                                        "PERFORMANCE_HINT": "LATENCY"})
        self.n_capas = n_capas
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
        if not isinstance(cache, CacheOV):
            cache = self._arrancar_flujo(cache)
        S = inputs_embeds.shape[1]
        if position_ids is not None:
            pos = position_ids.detach().numpy().astype(np.int64)[:, -S:]
        else:
            pos = np.arange(cache.longitud, cache.longitud + S, dtype=np.int64)[None]
        res = cache.peticion.infer(
            {"inputs_embeds": np.ascontiguousarray(inputs_embeds.detach().float().numpy()),
             "position_ids": np.ascontiguousarray(pos)},
            share_inputs=True, share_outputs=True)
        h = torch.from_numpy(np.array(res[self.comp.output("hidden")]))
        cache.longitud += S
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

    El estado interno de OV sustituye a VibeVoiceTokenizerStreamingCache;
    nueva_sesion() lo pone a cero (== cache vacia) antes de cada generate.
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
        self.nueva_sesion()

    def nueva_sesion(self):
        self.pet.reset_state()
        for est in self.pet.query_state():
            est.state = self._ov.Tensor(self._ceros[est.name])

    def decode(self, latents, cache=None, sample_indices=None, use_cache=True, debug=False):
        lat = latents.detach().float()
        if lat.shape[1] != 64:          # [1,1,64] -> [1,64,1]
            lat = lat.permute(0, 2, 1)
        res = self.pet.infer({"lat": np.ascontiguousarray(lat.numpy())},
                             share_inputs=True, share_outputs=True)
        return torch.from_numpy(np.array(res[self.comp.output("audio")]))


def cargar(modelo_path, hilos, ir_lm, ir_cabeza, ir_acustico=None):
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
    modelo.model.tts_language_model = TtsLmOV(ir_lm, hilos)
    if ir_cabeza:
        modelo.model.prediction_head = CabezaOV(ir_cabeza, hilos)
    else:
        torch.ao.quantization.quantize_dynamic(
            modelo.model.prediction_head, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    if ir_acustico:
        acustico = AcusticoOV(ir_acustico, hilos)
        modelo.model.acoustic_tokenizer.decode = acustico.decode
        modelo._acustico_ov = acustico
    gc.collect()
    return procesador, modelo


