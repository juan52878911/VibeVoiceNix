#!/usr/bin/env python
"""Fase 0 del plan de rendimiento: lo que falta medir, en un proceso de laboratorio.

Corre EN la VM con el python del entorno de Nix (vibevoice-env), sin tocar el servicio:

    PY=/nix/store/...-vibevoice-env/bin/python
    $PY scripts/lab_fase0.py carga --modo viejo --codigo pkgs/vibevoice-ov --huella viejo.json
    $PY scripts/lab_fase0.py carga --modo nuevo --codigo pkgs/vibevoice-ov --huella nuevo.json
    $PY scripts/lab_fase0.py comparar viejo.json nuevo.json
    $PY scripts/lab_fase0.py lm --ir /var/lib/voz/ov/tts_lm_estado_int4.xml
    $PY scripts/lab_fase0.py lm_texto --codigo pkgs/vibevoice-ov

carga     RSS y VmHWM tras cada paso de la carga, cada compile_model incluido. `viejo` repite la
          carga de produccion hasta el 14-09-2026 (from_pretrained del fp32 entero); `nuevo` llama a
          motor.cargar. Con --huella deja, por cada tensor vivo del modelo torch, forma, dtype y
          sha256 (los Linear cuantizados por int_repr, escalas y sesgo): es la puerta previa de A1+A2.
          Pica 4,4 GB en modo viejo: con voz-stream parado.
lm        La pasada del LM TTS (IR con estado) aislada, a varios hilos y con la cache llena hasta N
          posiciones; y el par condicional + negativa en paralelo, que es lo que haria B2.
lm_texto  forward del LM de texto (4 capas, torch int8) por ventana de texto.

Todo lo que imprime lleva la unidad. Las filas de tiempos son medianas de --n llamadas tras calentar.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

MODELO = os.environ.get("VIBEVOICE_MODELO", "")
IR = "/var/lib/voz/ov"


def memoria():
    with open("/proc/self/status") as f:
        c = dict(linea.split(":", 1) for linea in f if linea.startswith(("VmRSS", "VmHWM", "RssAnon", "RssFile")))
    return {k: int(v.split()[0]) // 1024 for k, v in c.items()}


def marca(paso, filas):
    m = memoria()
    filas.append({"paso": paso, **m})
    print(f"  {paso:<34} RSS {m['VmRSS']:>5} MB  anon {m['RssAnon']:>5}  fichero {m['RssFile']:>4}"
          f"  pico {m['VmHWM']:>5} MB", flush=True)


def trim():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


# ------------------------------------------------------------------ huella
def huella(modelo):
    """nombre -> [forma, dtype, sha256] de todo lo que el modelo torch usa al inferir."""
    import torch
    salida = {}

    def h(t):
        t = t.detach().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]

    for nombre, t in list(modelo.named_parameters()) + list(modelo.named_buffers()):
        if t.is_meta:
            salida[nombre] = ["META"]
            continue
        salida[nombre] = [list(t.shape), str(t.dtype), h(t)]
    for nombre, mod in modelo.named_modules():
        # Linear dinamico int8 (su hijo LinearPackedParams tambien tiene _packed_params y no weight())
        if isinstance(mod, torch.ao.nn.quantized.dynamic.Linear):
            w = mod.weight()
            escalas = w.q_per_channel_scales() if w.qscheme() in (torch.per_channel_affine,) else torch.tensor([w.q_scale()])
            salida[nombre + ".weight(int8)"] = [list(w.shape), str(w.dtype), h(w.int_repr()), h(escalas)]
            if mod.bias() is not None:
                salida[nombre + ".bias"] = [list(mod.bias().shape), str(mod.bias().dtype), h(mod.bias())]
        tabla = getattr(mod, "_tabla", None)         # EmbeddingMmap: se compara como si fuera fp32
        if tabla is not None:
            salida[nombre + ".weight"] = [list(tabla.shape), "torch.float32", h(tabla.float())]
    salida["__attn__"] = [str(getattr(modelo.config, "_attn_implementation", None)),
                          str(getattr(modelo.model.language_model.config, "_attn_implementation", None))]
    return salida


def cmd_carga(a):
    import torch
    sys.path.insert(0, a.codigo)
    import openvino as ov
    import motor
    torch.set_num_threads(a.hilos)
    ir_lm, ir_cab = f"{IR}/tts_lm_estado_int4.xml", f"{IR}/cabeza_int8.xml"
    ir_ac, ir_dif = f"{IR}/decoder_mm_int8.xml", f"{IR}/difusion_p6_int8.xml"
    filas = []
    marca("inicio (torch + openvino importados)", filas)
    if a.modo == "viejo":
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference as Clase)
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
        modelo = Clase.from_pretrained(a.modelo, torch_dtype=torch.float32, device_map="cpu",
                                       attn_implementation="sdpa")
        modelo.eval()
        marca("from_pretrained fp32 entero", filas)
        modelo.model.tts_language_model = None
        modelo.model.acoustic_tokenizer.decoder = None
        modelo.model.acoustic_tokenizer.encoder = None
        modelo.model.acoustic_tokenizer._ancla = torch.nn.Parameter(torch.zeros(1))
        gc.collect()
        marca("soltados LM TTS y decodificador", filas)
        torch.ao.quantization.quantize_dynamic(modelo.model.language_model, {torch.nn.Linear},
                                               dtype=torch.qint8, inplace=True)
        gc.collect()
        marca("LM de texto en int8", filas)
        trim()
        marca("malloc_trim", filas)
        modelo.model.tts_language_model = motor.TtsLmOV(ir_lm, a.hilos)
        marca("compile LM TTS int4", filas)
        comp = ov.Core().compile_model(ir_cab, "CPU", {"INFERENCE_NUM_THREADS": a.hilos, "NUM_STREAMS": 1,
                                                       "PERFORMANCE_HINT": "LATENCY"})
        pet = comp.create_infer_request()
        marca("compile cabeza int8 (sin uso)", filas)
        modelo.model.prediction_head = None
        acustico = motor.AcusticoOV(ir_ac, a.hilos)
        modelo.model.acoustic_tokenizer.decode = acustico.decode
        marca("compile decodificador int8", filas)
    else:
        _, modelo = motor.cargar(a.modelo, a.hilos, ir_lm, ir_cab, ir_ac)
        marca("motor.cargar", filas)
    difusion = motor.DifusionOV(ir_dif, a.hilos)
    marca("compile difusion p6 int8", filas)
    trim()
    marca("malloc_trim final", filas)
    if a.huella:
        # la huella no mira los modulos que sustituye OpenVINO
        modelo.model.tts_language_model = None
        modelo.model.prediction_head = None
        hu = huella(modelo)
        json.dump({"memoria": filas, "huella": hu}, open(a.huella, "w"), indent=1)
        print(f"huella de {len(hu)} tensores en {a.huella}")
    del difusion


def cmd_comparar(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    hx, hy = x["huella"], y["huella"]
    solo_x = sorted(set(hx) - set(hy))
    solo_y = sorted(set(hy) - set(hx))
    distintos = sorted(k for k in set(hx) & set(hy) if hx[k] != hy[k])
    print(f"{len(hx)} / {len(hy)} tensores; solo en {a.a}: {len(solo_x)}; solo en {a.b}: {len(solo_y)}; "
          f"distintos: {len(distintos)}")
    for k in solo_x[:20]:
        print("  solo A ", k, hx[k])
    for k in solo_y[:20]:
        print("  solo B ", k, hy[k])
    for k in distintos[:20]:
        print("  DISTINTO", k, hx[k], hy[k])
    igual = not (solo_x or solo_y or distintos)
    print("HUELLA IDENTICA" if igual else "HUELLA DISTINTA")
    for fa, fb in [(x["memoria"][-1], y["memoria"][-1])]:
        print(f"pico A {max(f['VmHWM'] for f in x['memoria'])} MB  B {max(f['VmHWM'] for f in y['memoria'])} MB; "
              f"RSS final A {fa['VmRSS']} MB  B {fb['VmRSS']} MB")
    sys.exit(0 if igual else 1)


# ------------------------------------------------------------------ LM TTS aislado
def _llenar(pet, posiciones, rng):
    """Deja la cache KV de un InferRequest con `posiciones` entradas aleatorias."""
    import openvino as ov
    emb = (rng.standard_normal((1, 1, 896)) * 0.02).astype(np.float32)
    pet.infer({"inputs_embeds": emb, "position_ids": np.zeros((1, 1), np.int64)})
    for est in pet.query_state():
        b, h, _, d = est.state.shape
        est.state = ov.Tensor((rng.standard_normal((b, h, posiciones, d)) * 0.5).astype(np.float32))


def _medir(fn, n, calentar=5):
    for _ in range(calentar):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def cmd_lm(a):
    import openvino as ov
    core = ov.Core()
    rng = np.random.default_rng(0)
    emb = (rng.standard_normal((1, 1, 896)) * 0.02).astype(np.float32)
    resultados = []
    for ronda in range(a.rondas):
        for P in a.posiciones:
            fila = {"ronda": ronda, "posiciones": P}
            for h in a.hilos:
                comp = core.compile_model(a.ir, "CPU", {"INFERENCE_NUM_THREADS": h, "NUM_STREAMS": 1,
                                                        "PERFORMANCE_HINT": "LATENCY"})
                pet = comp.create_infer_request()
                _llenar(pet, P, rng)
                pos = np.array([[P]], np.int64)
                fila[f"ms_{h}h"] = _medir(lambda: pet.infer({"inputs_embeds": emb, "position_ids": pos}), a.n)
                del pet, comp
            # B2: condicional y negativa a la vez. (1) un modelo con 2 streams y los 6 hilos repartidos;
            # (2) dos modelos compilados de 3 hilos cada uno.
            comp2 = core.compile_model(a.ir, "CPU", {"INFERENCE_NUM_THREADS": 6, "NUM_STREAMS": 2})
            fila["streams_reales"] = int(comp2.get_property("NUM_STREAMS"))
            p1, p2 = comp2.create_infer_request(), comp2.create_infer_request()
            _llenar(p1, P, rng)
            _llenar(p2, P, rng)
            pos = np.array([[P]], np.int64)
            entrada = {"inputs_embeds": emb, "position_ids": pos}

            def par(x=p1, y=p2):
                x.start_async(entrada)
                y.start_async(entrada)
                x.wait()
                y.wait()
            fila["ms_par_2streams"] = _medir(par, a.n)
            del p1, p2, comp2
            ca = core.compile_model(a.ir, "CPU", {"INFERENCE_NUM_THREADS": 3, "NUM_STREAMS": 1,
                                                  "PERFORMANCE_HINT": "LATENCY"})
            cb = core.compile_model(a.ir, "CPU", {"INFERENCE_NUM_THREADS": 3, "NUM_STREAMS": 1,
                                                  "PERFORMANCE_HINT": "LATENCY"})
            q1, q2 = ca.create_infer_request(), cb.create_infer_request()
            _llenar(q1, P, rng)
            _llenar(q2, P, rng)
            fila["ms_par_2modelos"] = _medir(lambda: par(q1, q2), a.n)
            del q1, q2, ca, cb
            if "ms_6h" in fila:
                fila["ratio_3h_6h"] = fila.get("ms_3h", float("nan")) / fila["ms_6h"]
                fila["ratio_par_2streams"] = fila["ms_par_2streams"] / (2 * fila["ms_6h"])
                fila["ratio_par_2modelos"] = fila["ms_par_2modelos"] / (2 * fila["ms_6h"])
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}),
                  flush=True)
            resultados.append(fila)
    if a.salida:
        json.dump(resultados, open(a.salida, "w"), indent=1)


def cmd_resumen_lm(a):
    """Aplica la regla de docs/plan-rendimiento.md (0.2 repetida), fijada antes de medir:
    mediana por configuracion y longitud sobre todas las rondas; cocientes con esas medianas; el mejor
    de los dos montajes del par; la medida no vale si el IQR de t6 pasa del 25 % de su mediana."""
    filas = json.load(open(a.lm))
    veredictos, valida = [], True
    for P in sorted({f["posiciones"] for f in filas}):
        grupo = [f for f in filas if f["posiciones"] == P]
        med = {k: statistics.median(f[k] for f in grupo)
               for k in ("ms_6h", "ms_3h", "ms_2h", "ms_par_2streams", "ms_par_2modelos")}
        t6 = sorted(f["ms_6h"] for f in grupo)
        q1, q3 = np.percentile(t6, 25), np.percentile(t6, 75)
        iqr_rel = (q3 - q1) / med["ms_6h"]
        ok_iqr = iqr_rel <= 0.25
        valida &= ok_iqr
        r3 = med["ms_3h"] / med["ms_6h"]
        par = min(med["ms_par_2streams"], med["ms_par_2modelos"]) / (2 * med["ms_6h"])
        cumple = r3 <= 1.6 and par <= 0.80
        veredictos.append(cumple)
        print(f"{P:>5} posiciones, {len(grupo)} rondas: medianas "
              + " ".join(f"{k[3:]}={v:.2f}" for k, v in med.items())
              + f" | t3/t6={r3:.3f} par/(2 t6)={par:.3f} -> {'cumple' if cumple else 'NO cumple'}"
              + f" | IQR t6 {100 * iqr_rel:.0f} % {'ok' if ok_iqr else 'DEMASIADO RUIDO'}")
    if not valida:
        print("MEDIDA NO VALIDA (IQR de t6 > 25 %): B2 sin decidir, repetir con el host en reposo")
    else:
        print("B2 SE ABRE" if all(veredictos) else "B2 SE CIERRA")


# ------------------------------------------------------------------ LM de texto
def cmd_lm_texto(a):
    import torch
    from transformers import DynamicCache
    sys.path.insert(0, a.codigo)
    import motor
    torch.set_num_threads(a.hilos)
    modelo = motor._construir_parcial(a.modelo, soltar_acustico=True, soltar_cabeza=True)
    lm = modelo.model.language_model
    torch.ao.quantization.quantize_dynamic(lm, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    for C in a.contextos:
        with torch.no_grad():
            cache = DynamicCache()
            lm(input_ids=torch.randint(0, 150000, (1, C)), past_key_values=cache, use_cache=True)
            base = cache.get_seq_length()

            def ventana():
                # la misma posicion cada vez: se recorta la cache a su largo base
                cache.crop(base)
                lm(input_ids=torch.randint(0, 150000, (1, a.ventana)), past_key_values=cache, use_cache=True)
            ms = _medir(ventana, a.n)
        print(json.dumps({"contexto": C, "ventana": a.ventana, "ms_por_ventana": round(ms, 2)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("carga")
    c.add_argument("--modo", choices=("viejo", "nuevo"), required=True)
    c.add_argument("--codigo", required=True)
    c.add_argument("--modelo", default=MODELO)
    c.add_argument("--hilos", type=int, default=6)
    c.add_argument("--huella")
    k = sub.add_parser("comparar")
    k.add_argument("a")
    k.add_argument("b")
    m = sub.add_parser("lm")
    m.add_argument("--ir", default=f"{IR}/tts_lm_estado_int4.xml")
    m.add_argument("--hilos", type=lambda s: [int(x) for x in s.split(",")], default=[6, 3, 2])
    m.add_argument("--posiciones", type=lambda s: [int(x) for x in s.split(",")], default=[500, 1500])
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--rondas", type=int, default=2)
    m.add_argument("--salida")
    r = sub.add_parser("resumen_lm")
    r.add_argument("lm")
    t = sub.add_parser("lm_texto")
    t.add_argument("--codigo", required=True)
    t.add_argument("--modelo", default=MODELO)
    t.add_argument("--hilos", type=int, default=6)
    t.add_argument("--ventana", type=int, default=5)
    t.add_argument("--contextos", type=lambda s: [int(x) for x in s.split(",")], default=[50, 200, 500])
    t.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    {"carga": cmd_carga, "comparar": cmd_comparar, "lm": cmd_lm, "resumen_lm": cmd_resumen_lm,
     "lm_texto": cmd_lm_texto}[a.cmd](a)


if __name__ == "__main__":
    main()
