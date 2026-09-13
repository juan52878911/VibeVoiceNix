#!/usr/bin/env python
"""Decodificador acustico manual -> IR de OpenVINO (estado explicito) + compresion.

Formas 100% estaticas (1 frame -> 3200 muestras) y 34 estados de conv como
tensores de entrada/salida explicitos, que lleva motor.AcusticoOV (ceros ==
cache vacia del original). compress_weights en int8 comprime todo
lo que lleva pesos (FFN, subidas, convoluciones); el int4 solo los MatMul.
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
# decoder_mm y no decoder_estado: las subidas ya no son convoluciones
# traspuestas (ver SubidaTr en decoder_manual.py) y el IR va SIN estado. Nombre
# nuevo para que el IR viejo siga en su sitio -- volver atras es cambiar una
# ruta -- y para que el servicio de Nix, que solo genera lo que falta, lo genere.
NOMBRE = os.environ.get("VIBEVOICE_DECODER_NOMBRE", "decoder_mm")

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

# formas estaticas: cada estado debe medir exactamente
# su contexto (si queda dinamico, el estado inicial sale {0,C,0} y el concat
# revienta; ademas las kernels estaticas compilan mejor)
formas = {"lat": ov.PartialShape([1, 64, 1])}
for i, (c, l) in enumerate(m.formas_estado):
    formas["est.%d.in" % i] = ov.PartialShape([1, c, l])
mo.reshape(formas)

# SIN make_stateful: el estado (las 34 colas) entra y sale como tensores
# explicitos y lo lleva motor.AcusticoOV en Python. Con las subidas por
# productos de matrices, un IR con estado de OpenVINO leido desde fichero
# realimenta mal -- 118 dB de SNR frente a torch con el estado aplicado en
# memoria, -2,9 dB en cuanto el grafo pasa por fichero (con estado guardado o
# puesto al releer, por nombre o por posicion, en fp16 o en float32) --,
# mientras que el mismo fichero con el estado explicito da 71 dB, lo mismo que
# el decodificador viejo. Orden: lat, est.0..33 / audio, est.0..33.
ov.save_model(mo, LD + "/" + NOMBRE + "_fp16.xml")
print("IR sin estado guardado; entradas:", len(mo.inputs), "salidas:", len(mo.outputs), flush=True)
del m, mo
gc.collect()

import nncf
core = ov.Core()
# int8 e int4. El int4 necesita group_size=32 y no 128: el decodificador tiene
# capas de 32 y 64 canales, y agruparlas de 128 en 128 aborta la conversion
# entera con nncf.errors.InvalidGroupSizeError.
#
# POR QUE EL int4 AQUI TIENE SENTIDO Y EN LA CABEZA NO
# Este grafo paga por sus bytes. Medido en la VM con el banco aislado, mismo
# decodificador y 6 hilos:
#
#   decoder fp16   687 MB   70,0 ms
#   decoder int8   344 MB   42,9 ms
#
# Doblar los pesos cuesta 1,63x el tiempo. La cabeza, en cambio, empeoro con
# int4 (2,86 ms frente a 2,60 en int8) y ademas sesga el fin de frase.
for modo, gs, ruta in [(nncf.CompressWeightsMode.INT8_ASYM, None, LD + "/" + NOMBRE + "_int8.xml"),
                       (nncf.CompressWeightsMode.INT4_SYM, 32, LD + "/" + NOMBRE + "_int4.xml")]:
    mm = core.read_model(LD + "/" + NOMBRE + "_fp16.xml")
    # Las subidas (productos de matrices desde SubidaTr) se comprimen como las
    # demas, igual que ya se comprimian cuando eran convoluciones traspuestas:
    # el int8 de nncf tambien toca las convoluciones, y el IR viejo las llevaba
    # en u8. Medido frente al fp16, con el camino del servicio: int8 viejo 16,9
    # dB, int8 nuevo con las subidas en u8 16,9 dB, y con ellas en fp16 16,9
    # dB. El error lo pone la cuantizacion de las FFN; dejar las subidas en
    # fp16 solo cuesta 40 MB de RAM.
    kw = dict(mode=modo)
    if gs:
        kw.update(group_size=gs, ratio=1.0)
    mm = nncf.compress_weights(mm, **kw)
    ov.save_model(mm, ruta)
    print("guardado", ruta, flush=True)
    del mm
    gc.collect()
