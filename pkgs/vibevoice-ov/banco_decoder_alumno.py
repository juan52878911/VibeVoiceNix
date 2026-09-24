#!/usr/bin/env python
"""¿Cuanto mas rapido seria un decodificador mas pequeno? Antes de entrenar nada: alumnos con pesos
aleatorios (el tiempo no depende de los valores) convertidos como produccion (IR sin estado, int8 de
nncf) frente al decoder_mm_int8.xml de produccion, un fotograma por llamada, en esta CPU.

  VIBEVOICE_IR=/var/lib/voz/ov python banco_decoder_alumno.py [hilos]
"""
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openvino as ov
import torch

from decoder_alumno import DecodificadorAlumno

HILOS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
core = ov.Core()


def compilar(ruta):
    from motor import config_cpu                      # la misma configuracion que produccion
    return core.compile_model(ruta, "CPU", config_cpu(HILOS))


def medir(comp, n=150):
    pet = comp.create_infer_request()
    entradas = [np.zeros(tuple(p.get_partial_shape().to_shape()), dtype=np.float32) for p in comp.inputs]
    entradas[0] = np.random.randn(*entradas[0].shape).astype(np.float32)
    for _ in range(15):
        pet.infer(entradas)
    t = []
    for _ in range(n):
        i = time.perf_counter()
        pet.infer(entradas)
        t.append((time.perf_counter() - i) * 1000)
    return statistics.median(t)


def convertir(escala, prof):
    import nncf
    m = DecodificadorAlumno(escala, prof).eval()
    ej = (torch.zeros(1, 64, 1), *[torch.zeros(1, c, k) for c, k in m.formas_estado])
    with torch.no_grad():
        mo = ov.convert_model(m, example_input=ej)
    mo.reshape({p: ov.PartialShape(list(t.shape)) for p, t in zip(mo.inputs, ej)})   # estaticas, como convertir_decoder
    d = tempfile.mkdtemp()
    ov.save_model(mo, f"{d}/a_fp16.xml")
    m8 = nncf.compress_weights(core.read_model(f"{d}/a_fp16.xml"), mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(m8, f"{d}/a_int8.xml")
    n = sum(p.numel() for p in m.parameters())
    return f"{d}/a_int8.xml", n


prod = os.environ.get("VIBEVOICE_IR", "/var/lib/voz/ov") + "/decoder_mm_int8.xml"
print(f"produccion (decoder_mm_int8): {medir(compilar(prod)):.1f} ms por fotograma, {HILOS} hilos", flush=True)
for escala, prof in ((0.5, (8, 3, 3, 3, 3, 3, 3)), (0.5, (2, 2, 2, 2, 2, 2, 2)),
                     (0.35, (4, 2, 2, 2, 2, 2, 2))):
    ruta, n = convertir(escala, prof)
    print(f"alumno escala {escala} profundidades {prof}: {n / 1e6:.0f} M parametros, "
          f"{medir(compilar(ruta)):.1f} ms por fotograma", flush=True)
