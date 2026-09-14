#!/usr/bin/env python
"""Fase 0.3 del plan de rendimiento: los IR de producción en la iGPU UHD 630 con el plugin GPU de OpenVINO.

Corre en el LXC de laboratorio de pve (con /dev/dri) o en la VM (solo CPU), con openvino y numpy:

    python lab_igpu.py medir --ir /root/ir/decoder_mm_int8.xml --dispositivo GPU
    python lab_igpu.py snr --ir /root/ir/decoder_mm_fp16.xml          # GPU frente a CPU, mismo IR
    python lab_igpu.py bucle --ir /root/ir/decoder_mm_int8.xml --dispositivo GPU --segundos 60

medir  ms por llamada (mediana y p90 de --n tras calentar), tiempo de compilación y primera llamada.
       Decodificador: un fotograma [1,64,1] con las 34 colas explícitas realimentadas. LM TTS: una
       pasada de 1 token con la caché llena hasta --posiciones. Difusión: el bucle de un fotograma.
snr    el decodificador encadenado --n fotogramas con los mismos latentes en CPU y en GPU: SNR del
       audio GPU contra CPU (informativo; la puerta de C1 es contra torch fp32).
bucle  llamadas seguidas durante --segundos, con una fila por segundo (epoch, llamadas, mediana ms):
       sirve de carga fija para ver cuánto frena la GPU a la CPU (umbral de 0.3).
"""
import argparse
import json
import statistics
import time

import numpy as np
import openvino as ov


def tipo(modelo):
    nombres = {n for p in modelo.inputs for n in p.get_names()}
    if "lat" in nombres:
        return "decoder"
    if "inputs_embeds" in nombres:
        return "lm"
    return "difusion"


class Banco:
    def __init__(self, ir, dispositivo, posiciones=500, hilos=None, precision=None):
        core = ov.Core()
        self.modelo = core.read_model(ir)
        self.tipo = tipo(self.modelo)
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if dispositivo == "CPU":
            cfg.update({"INFERENCE_NUM_THREADS": hilos or 6, "NUM_STREAMS": 1})
        elif precision:
            # la GPU calcula en f16 por defecto aunque el IR sea int8 o fp16
            cfg["INFERENCE_PRECISION_HINT"] = precision
        t = time.perf_counter()
        self.comp = core.compile_model(self.modelo, dispositivo, cfg)
        self.compilar_s = time.perf_counter() - t
        self.pet = self.comp.create_infer_request()
        self.rng = np.random.default_rng(0)
        self.precision = str(self.comp.get_property("INFERENCE_PRECISION_HINT")) if dispositivo == "GPU" else "f32"
        if self.tipo == "decoder":
            self.est_in = sorted((p for p in self.comp.inputs if p.get_any_name() != "lat"),
                                 key=lambda p: int(p.get_any_name().split(".")[1]))
            self.est_out = {p.get_any_name().replace(".out", ".in"): p for p in self.comp.outputs
                            if p.get_any_name() != "audio"}
            self.estado = {p.get_any_name(): np.zeros(list(p.get_shape()), np.float32) for p in self.est_in}
        elif self.tipo == "lm":
            self.pos = posiciones
            emb = np.zeros((1, 1, 896), np.float32)
            self.pet.infer({"inputs_embeds": emb, "position_ids": np.zeros((1, 1), np.int64)})
            for est in self.pet.query_state():
                b, h, _, d = est.state.shape
                est.state = ov.Tensor((self.rng.standard_normal((b, h, posiciones, d)) * 0.5).astype(np.float32))
        else:
            self.entradas = [(self.rng.standard_normal(list(p.get_shape())) * (0.1 if i < 2 else 0)).astype(np.float32)
                             for i, p in enumerate(self.comp.inputs)]
            if len(self.entradas) >= 4:
                self.entradas[2] = np.array(3.0, np.float32).reshape(self.entradas[2].shape)
                self.entradas[3] = np.array(0.75, np.float32).reshape(self.entradas[3].shape)

    def llamada(self, lat=None):
        if self.tipo == "decoder":
            if lat is None:
                lat = (self.rng.standard_normal((1, 64, 1)) * 0.5).astype(np.float32)
            entradas = {"lat": lat, **self.estado}
            res = self.pet.infer(entradas)
            self.estado = {k: np.array(res[p], copy=True) for k, p in self.est_out.items()}
            return np.array(res[self.comp.output("audio")], copy=True)
        if self.tipo == "lm":
            emb = (self.rng.standard_normal((1, 1, 896)) * 0.02).astype(np.float32)
            self.pet.infer({"inputs_embeds": emb, "position_ids": np.array([[self.pos]], np.int64)})
            return None
        self.pet.infer(self.entradas)
        return None


def cmd_medir(a):
    b = Banco(a.ir, a.dispositivo, a.posiciones, a.hilos, a.precision)
    t = time.perf_counter()
    b.llamada()
    primera = time.perf_counter() - t
    for _ in range(a.calentar):
        b.llamada()
    ts = []
    for _ in range(a.n):
        t = time.perf_counter()
        b.llamada()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(json.dumps({"ir": a.ir.split("/")[-1], "tipo": b.tipo, "dispositivo": a.dispositivo,
                      "precision": b.precision, "compilar_s": round(b.compilar_s, 2),
                      "primera_s": round(primera, 3), "ms_mediana": round(statistics.median(ts), 2),
                      "ms_p90": round(ts[int(0.9 * len(ts))], 2), "n": a.n}), flush=True)


def cmd_snr(a):
    """Con latentes aleatorios la salida puede quedar casi muda y la SNR no dice nada: se imprime el RMS
    de la referencia y se barre la escala. La puerta de C1 se mide con latentes reales contra torch fp32."""
    for escala in a.escalas:
        rng = np.random.default_rng(1)
        lats = [(rng.standard_normal((1, 64, 1)) * escala).astype(np.float32) for _ in range(a.n)]
        salidas = {}
        for d in ("CPU", "GPU"):
            b = Banco(a.ir, d, precision=a.precision)
            salidas[d] = np.concatenate([b.llamada(x).reshape(-1) for x in lats])
        ref, x = salidas["CPU"].astype(np.float64), salidas["GPU"].astype(np.float64)
        snr = 10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30))
        print(json.dumps({"ir": a.ir.split("/")[-1], "precision_gpu": a.precision or "defecto",
                          "escala": escala, "fotogramas": a.n,
                          "rms_cpu": float(np.sqrt((ref ** 2).mean())), "rms_gpu": float(np.sqrt((x ** 2).mean())),
                          "snr_gpu_vs_cpu_db": round(float(snr), 2), "dif_max": float(np.abs(ref - x).max())}),
              flush=True)


def cmd_bucle(a):
    b = Banco(a.ir, a.dispositivo, a.posiciones, a.hilos, a.precision)
    fin = time.time() + a.segundos
    while time.time() < fin:
        seg, ts = int(time.time()), []
        while int(time.time()) == seg:
            t = time.perf_counter()
            b.llamada()
            ts.append((time.perf_counter() - t) * 1000)
        print(f"{seg},{len(ts)},{statistics.median(ts):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("medir", "snr", "bucle"):
        p = sub.add_parser(nombre)
        p.add_argument("--ir", required=True)
        p.add_argument("--dispositivo", default="GPU")
        p.add_argument("--posiciones", type=int, default=500)
        p.add_argument("--hilos", type=int)
        p.add_argument("--n", type=int, default=200 if nombre == "medir" else 40)
        p.add_argument("--calentar", type=int, default=20)
        p.add_argument("--segundos", type=int, default=60)
        p.add_argument("--escalas", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 2.0, 5.0])
        p.add_argument("--precision", choices=("f16", "f32"), help="INFERENCE_PRECISION_HINT de la GPU")
    a = ap.parse_args()
    {"medir": cmd_medir, "snr": cmd_snr, "bucle": cmd_bucle}[a.cmd](a)


if __name__ == "__main__":
    main()
