#!/usr/bin/env python
"""Decodificador acustico manual -> IR de OpenVINO con estado + compresion.

Formas 100% estaticas (1 frame -> 3200 muestras), 34 estados de conv que
make_stateful convierte en variables internas (reset_state() == cache vacia
del original). compress_weights solo toca MatMul: las FFN de los bloques
(~78% de los 343M de parametros); las convs quedan en fp16.
"""
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import openvino as ov

from decoder_manual import cargar, estados_cero

ST = os.environ["VIBEVOICE_MODELO"] + "/model.safetensors"
LD = os.environ["VIBEVOICE_IR"]

m = cargar(ST)
n_est = len(m.formas_estado)
print("params %.0fM, %d estados" % (sum(p.numel() for p in m.parameters()) / 1e6, n_est), flush=True)

ej = (torch.zeros(1, 64, 1), *estados_cero(m))
with torch.no_grad():
    mo = ov.convert_model(m, example_input=ej)

nombres_e = ["lat"] + ["est.%d.in" % i for i in range(n_est)]
nombres_s = ["audio"] + ["est.%d.out" % i for i in range(n_est)]
for p, nom in zip(mo.inputs, nombres_e):
    p.get_node().set_friendly_name(nom)
    p.get_tensor().set_names({nom})
for p, nom in zip(mo.outputs, nombres_s):
    p.get_tensor().set_names({nom})

# formas estaticas ANTES de make_stateful: el estado debe medir exactamente
# su contexto (si queda dinamico, el estado inicial sale {0,C,0} y el concat
# revienta; ademas las kernels estaticas compilan mejor)
formas = {"lat": ov.PartialShape([1, 64, 1])}
for i, (c, l) in enumerate(m.formas_estado):
    formas["est.%d.in" % i] = ov.PartialShape([1, c, l])
mo.reshape(formas)

from openvino._offline_transformations import apply_make_stateful_transformation
apply_make_stateful_transformation(mo, {"est.%d.in" % i: "est.%d.out" % i for i in range(n_est)})
ov.save_model(mo, LD + "/decoder_estado_fp16.xml")
print("IR con estado guardado; entradas:", [list(p.get_names()) for p in mo.inputs], flush=True)
del m, mo
gc.collect()

import nncf
core = ov.Core()
for modo, gs, ruta in [(nncf.CompressWeightsMode.INT8_ASYM, None, LD + "/decoder_estado_int8.xml"),
                       (nncf.CompressWeightsMode.INT4_ASYM, 128, LD + "/decoder_estado_int4.xml")]:
    mm = core.read_model(LD + "/decoder_estado_fp16.xml")
    kw = dict(mode=modo)
    if gs:
        kw.update(group_size=gs, ratio=1.0)
    mm = nncf.compress_weights(mm, **kw)
    ov.save_model(mm, ruta)
    print("guardado", ruta, flush=True)
    del mm
    gc.collect()
