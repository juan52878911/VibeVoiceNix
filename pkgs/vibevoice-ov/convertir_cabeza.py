#!/usr/bin/env python
"""Cabeza de difusion de VibeVoice -> OpenVINO IR + compresion NNCF.

Es la pieza pequena (84 MB bf16) que se ejecuta `pasos` veces por token de
voz con formas SIEMPRE estaticas: batch 2 (CFG positivo+negativo), latente 64,
condicion 896. Ideal para validar la cadena entera (convertir -> comprimir ->
paridad -> medir) antes de gastar tiempo en el backbone de 869 MB.

Fases (todas caben de sobra en RAM porque el modulo es pequeno):
  1. Cargar SOLO model.prediction_head.* del safetensors (lectura parcial).
  2. ov.convert_model con entradas de ejemplo estaticas.
  3. nncf.compress_weights INT8_ASYM e INT4 (group_size 64: los canales de
     entrada minimos son 64, el latente; 128 no divide 64 y NNCF lo saltaria).
  4. Paridad numerica OV vs PyTorch fp32.
  5. Benchmark: torch fp32, torch int8 dinamico, OV fp32, OV int8, OV int4.
"""
import json
import time

import numpy as np
import torch

MODELO = os.environ["VIBEVOICE_MODELO"]
SALIDA = os.environ["VIBEVOICE_IR"]
HILOS = 6

torch.set_num_threads(HILOS)


def cargar_cabeza():
    """Instancia la cabeza y carga solo sus pesos (lectura parcial del safetensors)."""
    from safetensors import safe_open
    from vibevoice.modular.configuration_vibevoice import VibeVoiceDiffusionHeadConfig
    from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead

    cfg = json.load(open(f"{MODELO}/config.json"))["diffusion_head_config"]
    cfg.pop("model_type", None)
    cabeza = VibeVoiceDiffusionHead(VibeVoiceDiffusionHeadConfig(**cfg))

    pesos = {}
    with safe_open(f"{MODELO}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.prediction_head."):
                # bf16 en disco -> fp32 en memoria
                pesos[k[len("model.prediction_head."):]] = f.get_tensor(k).float()
    faltan, sobran = cabeza.load_state_dict(pesos, strict=False)
    print("pesos cargados: %d  faltan: %s  sobran: %s" % (len(pesos), faltan, sobran), flush=True)
    cabeza.eval()
    return cabeza


class CabezaEnvuelta(torch.nn.Module):
    """Adaptador con firma posicional plana para el trazado de OpenVINO."""

    def __init__(self, cabeza):
        super().__init__()
        self.cabeza = cabeza

    def forward(self, noisy, t, cond):
        # t llega como f32 desde OV; la cabeza lo usa en la embedding sinusoidal
        return self.cabeza(noisy, t, cond)


def medir(fn, n=300, calentamiento=20):
    for _ in range(calentamiento):
        fn()
    ini = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - ini) / n * 1000  # ms


def main():
    import openvino as ov
    import nncf

    cabeza = cargar_cabeza()

    # Entradas de ejemplo: batch 2 fijo (CFG duplica), latente 64, condicion 896
    ej = (torch.randn(2, 64), torch.tensor([500.0, 500.0]), torch.randn(2, 896))

    with torch.no_grad():
        ref = cabeza(*ej)

    envuelta = CabezaEnvuelta(cabeza)
    modelo_ov = ov.convert_model(envuelta, example_input=ej)
    ov.save_model(modelo_ov, f"{SALIDA}/cabeza_fp16.xml")  # fp16 por defecto
    print("convertida a IR", flush=True)

    core = ov.Core()
    conf = {"INFERENCE_NUM_THREADS": HILOS, "NUM_STREAMS": 1,
            "PERFORMANCE_HINT": "LATENCY"}

    entradas_np = [x.numpy() for x in ej]

    resultados = {}
    # --- variantes OV: fp16(=fp32 en CPU), int8, int4 ---
    for nombre, ruta in [("ov-fp16", None), ("ov-int8", "int8"), ("ov-int4", "int4")]:
        m = core.read_model(f"{SALIDA}/cabeza_fp16.xml")
        if ruta == "int8":
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM)
            ov.save_model(m, f"{SALIDA}/cabeza_int8.xml")
        elif ruta == "int4":
            # group_size 64: divide todos los canales de entrada (64, 896, 2688)
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT4_ASYM,
                                      group_size=64, ratio=1.0)
            ov.save_model(m, f"{SALIDA}/cabeza_int4.xml")
        comp = core.compile_model(m, "CPU", conf)
        pet = comp.create_infer_request()
        salida = pet.infer(entradas_np)[comp.output(0)]
        err = float(np.max(np.abs(salida - ref.numpy())))
        med = np.mean(np.abs(ref.numpy()))
        resultados[nombre] = (medir(lambda: pet.infer(entradas_np)), err, med)

    # --- variantes torch: fp32 e int8 dinamico ---
    with torch.no_grad():
        resultados["torch-fp32"] = (medir(lambda: cabeza(*ej)), 0.0, 0.0)
    torch.ao.quantization.quantize_dynamic(cabeza, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    with torch.no_grad():
        salida = cabeza(*ej)
        err8 = float(np.max(np.abs(salida.numpy() - ref.numpy())))
        resultados["torch-int8"] = (medir(lambda: cabeza(*ej)), err8, 0.0)

    print("\n|ref| medio = %.4f" % np.mean(np.abs(ref.numpy())), flush=True)
    print("%-11s %9s %10s" % ("VARIANTE", "MS/PASO", "ERR_MAX"), flush=True)
    for k in ["torch-fp32", "torch-int8", "ov-fp16", "ov-int8", "ov-int4"]:
        ms, err, _ = resultados[k]
        print("%-11s %9.3f %10.4f" % (k, ms, err), flush=True)


if __name__ == "__main__":
    import os
    os.makedirs(SALIDA, exist_ok=True)
    main()
