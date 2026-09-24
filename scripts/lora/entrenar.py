#!/usr/bin/env python3
"""Entrena un LoRA de VibeVoice-Realtime-0.5B con la pasada forzada validada (forzado.py, puerta 0).

  python3 entrenar.py --datos datos/ --salida corrida1/ --pasos 3000 [--rango 16] [--lr 1e-4]

Los datos son los que deja datos.py: fragmentos .pt con ejemplos {idioma, hablante, ref_lat, ref_txt,
lat, txt}: la referencia es OTRO enunciado del mismo hablante (el prefijo de voz, como al clonar) y el
objetivo es lo que el modelo tiene que decir. Un 3 % se aparta para validar (por hablante).

Guarda el LoRA cada --cada pasos y el mejor por perdida de validacion; la perdida no decide nada por si
sola: la puerta la ponen evaluar.py (generando, no forzando) y el banco.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402
import lora as LR  # noqa: E402


def cargar_modelo(modelo_dir, dispositivo):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    proc = VibeVoiceStreamingProcessor.from_pretrained(modelo_dir)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        modelo_dir, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa")
    modelo.model.tts_language_model.embed_tokens = modelo.model.language_model.embed_tokens
    return proc, modelo.to(dispositivo)


def cargar_datos(carpeta):
    ejemplos = []
    for f in sorted(Path(carpeta).glob("*.pt")):
        ejemplos += torch.load(f, map_location="cpu")
    return ejemplos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--acumular", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--alfa", type=int, default=32)
    ap.add_argument("--ramas", default="language_model,tts_language_model")
    ap.add_argument("--fin", action="store_true", help="entrenar tambien el clasificador de fin (entero)")
    ap.add_argument("--cabeza", action="store_true", help="entrenar tambien la cabeza de difusion (entera)")
    ap.add_argument("--cada", type=int, default=250)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 (T4); por defecto fp32")
    a = ap.parse_args()
    random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    proc, modelo = cargar_modelo(a.modelo, d)
    tok = proc.tokenizer
    LR.poner(modelo, a.rango, a.alfa, 0.05, tuple(a.ramas.split(",")))
    extra = (("tts_eos_classifier",) if a.fin else ()) + (("model.prediction_head",) if a.cabeza else ())
    params = LR.congelar_salvo_lora(modelo, extra)
    n_ent = sum(p.numel() for p in params)
    print(f"[lora] entrenables {n_ent / 1e6:.2f} M de {sum(p.numel() for p in modelo.parameters()) / 1e6:.0f} M "
          f"(rango {a.rango}, ramas {a.ramas}{', fin' if a.fin else ''}{', cabeza' if a.cabeza else ''})", flush=True)
    ejemplos = cargar_datos(a.datos)
    hablantes = sorted({(e["idioma"], e["hablante"]) for e in ejemplos})
    random.Random(1).shuffle(hablantes)
    val_h = set(hablantes[: min(len(hablantes) - 1, max(1, len(hablantes) // 33))])
    val = [e for e in ejemplos if (e["idioma"], e["hablante"]) in val_h][:200]
    ent = [e for e in ejemplos if (e["idioma"], e["hablante"]) not in val_h]
    por_idioma = {}
    for e in ent:
        por_idioma.setdefault(e["idioma"], []).append(e)
    print(f"[lora] {len(ent)} ejemplos de entrenamiento, {len(val)} de validacion; "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(por_idioma.items())), flush=True)
    (sal / "config.json").write_text(json.dumps({**vars(a), "entrenables_M": round(n_ent / 1e6, 2),
                                                 "val_hablantes": sorted(map(list, val_h))}, indent=1))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda p: min(1.0, (p + 1) / 100) * max(0.1, 1 - p / a.pasos))
    escalador = torch.cuda.amp.GradScaler(enabled=a.fp16)
    programa = FZ.Programa(dispositivo=d)
    idiomas = sorted(por_idioma)

    def validar():
        modelo.eval()
        gen = torch.Generator(device=d).manual_seed(123)
        tot, n = {"dif": 0.0, "neg": 0.0, "fin": 0.0}, 0
        with torch.no_grad():
            estado_rng = torch.get_rng_state(), torch.cuda.get_rng_state() if d == "cuda" else None
            torch.manual_seed(123)
            for e in val:
                _, info = FZ.pasada(modelo, tok, e, programa)
                if info:
                    for k in tot:
                        tot[k] += info[k]
                    n += 1
            torch.set_rng_state(estado_rng[0])
            if estado_rng[1] is not None:
                torch.cuda.set_rng_state(estado_rng[1])
        modelo.train()
        return {k: round(v / max(1, n), 5) for k, v in tot.items()}

    registro = open(sal / "registro.jsonl", "a")
    base = validar()
    print(f"[lora] validacion antes de entrenar: {base}", flush=True)
    registro.write(json.dumps({"paso": 0, "val": base}) + "\n")
    mejor = base["dif"]
    modelo.train()
    t0, acum, media = time.time(), 0, {"dif": 0.0, "fin": 0.0, "neg": 0.0, "n": 0}
    for paso in range(1, a.pasos + 1):
        for _ in range(a.acumular):
            e = random.choice(por_idioma[random.choice(idiomas)])     # idiomas equilibrados
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
                perdida, info = FZ.pasada(modelo, tok, e, programa)
            if perdida is None:
                continue
            escalador.scale(perdida / a.acumular).backward()
            for k in ("dif", "fin", "neg"):
                media[k] += info[k]
            media["n"] += 1
        escalador.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        escalador.step(opt)
        escalador.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if paso % 25 == 0:
            n = max(1, media["n"])
            print(f"[lora] paso {paso}/{a.pasos} · dif {media['dif'] / n:.4f} · fin {media['fin'] / n:.4f} · "
                  f"neg {media['neg'] / n:.4f} · lr {sched.get_last_lr()[0]:.2e} · {(time.time() - t0) / paso:.2f} s/paso",
                  flush=True)
            media = {"dif": 0.0, "fin": 0.0, "neg": 0.0, "n": 0}
        if paso % a.cada == 0 or paso == a.pasos:
            v = validar()
            print(f"[lora] validacion paso {paso}: {v} (antes {base})", flush=True)
            registro.write(json.dumps({"paso": paso, "val": v}) + "\n")
            registro.flush()
            torch.save(LR.estado(modelo, extra), sal / f"lora_{paso}.pt")
            if v["dif"] < mejor:
                mejor = v["dif"]
                torch.save(LR.estado(modelo, extra), sal / "lora_mejor.pt")
    print(f"[lora] fin: {a.pasos} pasos en {(time.time() - t0) / 60:.1f} min; mejor dif de validacion {mejor:.5f} "
          f"(antes {base['dif']:.5f})", flush=True)


if __name__ == "__main__":
    main()
