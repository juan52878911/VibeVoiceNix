#!/usr/bin/env python
"""Variante CON ESTADO del backbone TTS: KV per-capa como variables de OV.

El barrido (barrido_ov.py) enseno que los pesos int4 ya van a ~suelo de ancho
de banda pero la parte de atencion cuesta +22 ms por cada 500 posiciones de
cache: el grafo sin estado materializa la expansion GQA 2->14 y trafica la
cache entera por E/S en cada llamada.

Aqui replicamos la receta de optimum-intel:
  - KV per-capa como entradas/salidas sueltas (past_key_values.i.key/value ->
    present.i.key/value), no apiladas, para que el plugin de CPU reconozca el
    patron ReadValue -> Concat -> SDPA y lo fusione en su kernel interno.
  - apply_make_stateful_transformation convierte cada par entrada/salida en
    una variable interna: cero trafico de cache por E/S.
  - El estado se puede INYECTAR (query_state().state = tensor), que es lo que
    necesita VibeVoice para cargar el prefijo de voz precalculado, y hay un
    InferRequest por flujo (positivo / negativo del CFG).

Al final comprime int8_sym / int4_sym, compila, inyecta cache de 500 y mide.
"""
import os
import gc
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import openvino as ov

from qwen2_manual import Qwen2Manual, cargar_desde_safetensors

ST = os.environ["VIBEVOICE_MODELO"] + "/model.safetensors"
LD = os.environ["VIBEVOICE_IR"]
N, HILOS = 20, 6


class PorCapas(torch.nn.Module):
    """Mismo computo que Qwen2Manual pero con KV suelto por capa (40 E / 40 S)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, inputs_embeds, position_ids, *kvs):
        pks, pvs = kvs[0::2], kvs[1::2]
        S = inputs_embeds.shape[1]
        P = pks[0].shape[2]
        frec = position_ids[..., None].float() * self.base.inv_freq[None, None, :]
        emb = torch.cat([frec, frec], dim=-1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        q_pos = P + torch.arange(S, device=inputs_embeds.device)
        k_pos = torch.arange(P + S, device=inputs_embeds.device)
        permitido = k_pos[None, :] <= q_pos[:, None]
        mascara = torch.where(permitido, 0.0, -1e9)[None, None].to(inputs_embeds.dtype)
        h = inputs_embeds
        salidas = []
        for capa, pk, pv in zip(self.base.capas, pks, pvs):
            h, k, v = capa(h, cos, sin, mascara, pk, pv)
            salidas += [k, v]
        return (self.base.norm(h), *salidas)


def convertir():
    base = cargar_desde_safetensors(ST, "model.tts_language_model", N)
    m = PorCapas(base)
    kv = [torch.randn(1, 2, 8, 64) for _ in range(2 * N)]
    ej = (torch.randn(1, 3, 896), torch.arange(8, 11)[None], *kv)
    with torch.no_grad():
        mo = ov.convert_model(m, example_input=ej)

    nombres_e = ["inputs_embeds", "position_ids"]
    nombres_s = ["hidden"]
    for i in range(N):
        nombres_e += ["past_key_values.%d.key" % i, "past_key_values.%d.value" % i]
        nombres_s += ["present.%d.key" % i, "present.%d.value" % i]
    formas = {"inputs_embeds": ov.PartialShape([1, -1, 896]),
              "position_ids": ov.PartialShape([1, -1])}
    for i, (p, nom) in enumerate(zip(mo.inputs, nombres_e)):
        p.get_node().set_friendly_name(nom)
        p.get_tensor().set_names({nom})
        if i >= 2:
            formas[nom] = ov.PartialShape([1, 2, -1, 64])
    for p, nom in zip(mo.outputs, nombres_s):
        p.get_tensor().set_names({nom})
    mo.reshape(formas)

    # ---> con estado: cada par past/present pasa a ser variable interna
    from openvino._offline_transformations import apply_make_stateful_transformation
    pares = {"past_key_values.%d.%s" % (i, c): "present.%d.%s" % (i, c)
             for i in range(N) for c in ("key", "value")}
    apply_make_stateful_transformation(mo, pares)
    ov.save_model(mo, f"{LD}/tts_lm_estado_fp16.xml")
    print("guardado IR con estado; entradas:", [list(p.get_names()) for p in mo.inputs], flush=True)
    del m, base, mo
    gc.collect()


def comprimir():
    import nncf
    core = ov.Core()
    for modo, gs, ruta in [(nncf.CompressWeightsMode.INT8_SYM, None, f"{LD}/tts_lm_estado_int8.xml"),
                           (nncf.CompressWeightsMode.INT4_SYM, 128, f"{LD}/tts_lm_estado_int4.xml")]:
        m = core.read_model(f"{LD}/tts_lm_estado_fp16.xml")
        kw = dict(mode=modo)
        if gs:
            kw.update(group_size=gs, ratio=1.0)
        m = nncf.compress_weights(m, **kw)
        ov.save_model(m, ruta)
        print("guardado", ruta, flush=True)
        del m
        gc.collect()


def bancada():
    core = ov.Core()
    rng = np.random.default_rng(7)
    embeds = rng.standard_normal((1, 1, 896), dtype=np.float32)
    for etiqueta in ["int8", "int4"]:
        comp = core.compile_model(f"{LD}/tts_lm_estado_{etiqueta}.xml", "CPU",
                                  {"INFERENCE_NUM_THREADS": HILOS, "NUM_STREAMS": 1,
                                   "PERFORMANCE_HINT": "LATENCY"})
        pet = comp.create_infer_request()
        for pasado in [0, 500, 1000]:
            pet.reset_state()
            if pasado:
                for est in pet.query_state():
                    kv = (rng.standard_normal((1, 2, pasado, 64)) * 0.5).astype(np.float32)
                    est.state = ov.Tensor(kv)
            entradas = {"inputs_embeds": embeds,
                        "position_ids": np.array([[pasado]], dtype=np.int64)}
            for _ in range(10):
                pet.infer(entradas, share_inputs=True, share_outputs=True)
            n = 60
            ini = time.perf_counter()
            for _ in range(n):
                pet.infer(entradas, share_inputs=True, share_outputs=True)
            ms = (time.perf_counter() - ini) / n * 1000
            # la cache crece 70 posiciones durante la medida; sesgo ~ +35 pos
            print("estado-%s  cache=%4d  %8.2f ms" % (etiqueta, pasado, ms), flush=True)
        del comp, pet
        gc.collect()


if __name__ == "__main__":
    if "solo-bancada" not in sys.argv:
        convertir()
        comprimir()
    bancada()
