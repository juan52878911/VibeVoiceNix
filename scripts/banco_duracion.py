#!/usr/bin/env python
"""¿Cuanto audio de referencia hace falta para clonar bien una voz?

    python scripts/banco_duracion.py

LA PREGUNTA
`scripts/clonar_voz.py` admite varias muestras de la misma voz. Pero nadie ha
medido si mas audio MEJORA el clon, ni donde deja de compensar. Las voces
oficiales llevan 22-33 s; los clones hechos hasta ahora, 10-15 s. Sin la curva,
pedirle a alguien "grabame 30 segundos mas" es fe, no ingenieria.

COMO SE MIDE SIN PEDIR MAS GRABACIONES
Con una voz OFICIAL como oraculo, el lazo se cierra solo:

  1. con la voz oficial se generan K clips de referencia desde textos CONOCIDOS
     -> pares (audio, transcripcion exacta) sin transcribir nada
  2. se fabrican prefijos con 1, 2, 3 ... K clips
  3. cada prefijo dice las frases de prueba, y tambien las dice el oraculo
  4. todo se compara contra la huella de la referencia completa

El oraculo diciendo esas mismas frases da el TECHO: es la misma voz de verdad,
asi que ningun clon deberia pasar de ahi. La curva se lee contra ese techo.

Se corren dos voces, una femenina y otra masculina, porque el sesgo de subir el
tono en voces graves ya aparecio antes y no se puede dar por hecho que la curva
sea la misma en las dos.

COSTE
(K referencias + K niveles x textos x semillas + techo) x 2 voces. Con lo de por
defecto son ~70 generaciones, unos 12 minutos en un M4 mas la carga del modelo.
"""
import argparse
import copy
import csv
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banco_clonado import Huella, escribir_wav, wer  # noqa: E402
from clonar_voz import igualar_volumen  # noqa: E402
from prosodia import descripcion  # noqa: E402

RITMO = 24000
MUESTRAS_LATENTE = 3200

# Cuatro textos de referencia, ~15 s hablados cada uno. Distinto contenido a
# proposito: si todos fuesen parecidos, anadir clips no anadiria informacion.
REFERENCIAS = [
    "El backup de anoche termino sin errores y los tres servicios responden con normalidad. "
    "La temperatura del disco sigue estable en cuarenta y dos grados. No hay ninguna alerta "
    "pendiente desde el martes por la tarde.",
    "Me acuerdo perfectamente de aquella tarde de verano en el pueblo, cuando bajabamos al rio "
    "con las bicicletas viejas y volviamos de noche muertos de hambre. Que tiempos aquellos, "
    "de verdad que si.",
    "El problema no es el precio, es que nadie te explica lo que estas comprando. Te ensenan "
    "una tabla con veinte filas, te sonrien, y cuando preguntas por la letra pequena te dicen "
    "que eso ya lo veremos mas adelante.",
    "Primero se calienta el aceite a fuego medio, se anade la cebolla bien picada y se deja "
    "unos ocho minutos hasta que quede transparente. Luego el ajo, que se quema en nada, y "
    "por ultimo el tomate con una pizca de azucar.",
]

PRUEBAS = {
    "neutro": "Manana por la manana hay que revisar el certificado del tunel, que caduca el viernes.",
    # sin signos de apertura: MEDIDO que "¿" y "¡" disparan el WER (ver banco_clonado.py)
    "expresivo": "En serio? No me lo puedo creer! Eso si que no me lo esperaba para nada.",
}


def cargar_modelo(modelo_dir, cache, disp, pasos, con_encoder=True):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    from clonar_voz import encoder_comunitario
    proc = VibeVoiceStreamingProcessor.from_pretrained(modelo_dir)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        modelo_dir, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    modelo.model.tts_language_model.embed_tokens = modelo.model.language_model.embed_tokens
    if con_encoder:
        modelo.model.acoustic_tokenizer.encoder.load_state_dict(
            {k: v.to(torch.float32) for k, v in encoder_comunitario(Path(cache)).items()},
            strict=True)
    modelo.set_ddpm_inference_steps(pasos)
    return proc, modelo.to(disp)


def hablar(proc, modelo, prefijo, texto, cfg, semilla, disp):
    torch.manual_seed(semilla)
    t = texto if texto.endswith("\n") else texto + "\n"
    ent = proc.process_input_with_cached_prompt(
        text=t, cached_prompt=copy.deepcopy(prefijo), padding=True,
        return_tensors="pt", return_attention_mask=True)
    ent = {k: (v.to(disp) if torch.is_tensor(v) else v) for k, v in ent.items()}
    with torch.no_grad():
        s = modelo.generate(**ent, max_new_tokens=None, cfg_scale=cfg,
                            tokenizer=proc.tokenizer, generation_config={"do_sample": False},
                            verbose=False, return_speech=True,
                            all_prefilled_outputs=copy.deepcopy(prefijo))
    return s.speech_outputs[0].detach().float().cpu().numpy().reshape(-1)


def construir_prefijo(proc, modelo, clips, textos, disp):
    """La receta de clonar_voz.py: N latentes + M texto, sin marcadores."""
    m, tok, tipo = modelo.model, proc.tokenizer, torch.float32
    texto = "".join(t if t.endswith("\n") else t + "\n" for t in textos)
    ids = torch.tensor([tok.encode(texto, add_special_tokens=False)], device=disp)
    with torch.no_grad():
        trozos = []
        for c in clips:
            n_c = math.ceil(len(c) / MUESTRAS_LATENTE)
            e = m.acoustic_tokenizer.encode(torch.from_numpy(c)[None, None].to(disp, tipo))
            z, _ = e.sample(dist_type=m.acoustic_tokenizer.std_dist_type)
            r = (z + m.speech_bias_factor) * m.speech_scaling_factor
            trozos.append(m.acoustic_connector(r[:, :n_c].to(tipo)))
        conectado = torch.cat(trozos, 1)
        n = conectado.shape[1]
        lm = modelo.forward_lm(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=True, return_dict=True)
        M = lm.last_hidden_state.shape[1]
        embeds = torch.cat(
            [conectado, torch.zeros(1, M, conectado.shape[-1], device=disp, dtype=tipo)], 1)
        tts = modelo.forward_tts_lm(
            attention_mask=torch.ones(1, n + M, dtype=torch.long, device=disp),
            inputs_embeds=embeds, lm_last_hidden_state=lm.last_hidden_state,
            tts_text_masks=torch.zeros(1, n + M, dtype=torch.long, device=disp),
            use_cache=True, return_dict=True)
        neg = torch.tensor([[tok.convert_tokens_to_ids("<|image_pad|>")]], device=disp)
        neg_lm = modelo.forward_lm(input_ids=neg, attention_mask=torch.ones_like(neg),
                                   use_cache=True, return_dict=True)
        neg_tts = modelo.forward_tts_lm(
            input_ids=neg, attention_mask=torch.ones_like(neg),
            lm_last_hidden_state=neg_lm.last_hidden_state,
            tts_text_masks=torch.ones_like(neg), use_cache=True, return_dict=True)
    # NO se pasa por a_cpu(): eso es para GUARDAR el .pt. Aqui el prefijo se
    # usa acto seguido para generar, y una cache en CPU con el modelo en MPS
    # revienta con "Passed CPU tensor to MPS op".
    return ({"lm": lm, "tts_lm": tts, "neg_lm": neg_lm, "neg_tts_lm": neg_tts}, n, M)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voces", nargs="+", default=["sp-Spk4_woman", "sp-Spk3_man"],
                    help="voces oficiales que hacen de oraculo")
    ap.add_argument("--semillas", type=int, nargs="+", default=[11, 42, 101])
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=10)
    ap.add_argument("--salida", default="banco_duracion")
    ap.add_argument("--csv", default="banco_duracion.csv")
    ap.add_argument("--dir-voces", default=os.environ.get(
        "VIBEVOICE_VOCES", str(Path.home() / ".cache/vibevoice-nix/voces")))
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    args = ap.parse_args()

    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
    huella = Huella()
    print(f"cargando el modelo en {disp}...", flush=True)
    proc, modelo = cargar_modelo(args.modelo, args.cache, disp, args.pasos)

    filas = []
    for voz in args.voces:
        print(f"\n{'='*70}\n{voz}\n{'='*70}", flush=True)
        oficial = torch.load(Path(args.dir_voces) / f"{voz}.pt",
                             weights_only=False, map_location=disp)

        # 1. las K referencias, con la voz oficial y textos conocidos
        clips, t0 = [], time.perf_counter()
        for i, t in enumerate(REFERENCIAS, 1):
            c = hablar(proc, modelo, oficial, t, args.cfg, 7, disp)
            escribir_wav(salida / f"{voz}-ref{i}.wav", c)
            clips.append(c)
            print(f"  referencia {i}: {len(c)/RITMO:.1f} s", flush=True)
        print(f"  ({time.perf_counter()-t0:.0f} s)")

        # La huella de la voz "de verdad": todas las referencias juntas.
        h_ref = huella(np.concatenate(clips))

        # 2. el TECHO: el propio oraculo diciendo las frases de prueba
        for clave, texto in PRUEBAS.items():
            for s in args.semillas:
                x = hablar(proc, modelo, oficial, texto, args.cfg, s, disp)
                filas.append(dict(voz=voz, nivel="oraculo", clips=0, segundos=0.0,
                                  posiciones=0, texto=clave, semilla=s,
                                  ecapa=float(huella(x) @ h_ref),
                                  wer=float("nan"), **descripcion(x)))
        ec = [f["ecapa"] for f in filas if f["voz"] == voz and f["nivel"] == "oraculo"]
        print(f"  TECHO (el oraculo contra si mismo): {np.mean(ec):.4f}")

        # 3. la curva: prefijos con 1, 2, ... K clips
        for k in range(1, len(clips) + 1):
            usados = igualar_volumen(clips[:k]) if k > 1 else clips[:1]
            pref, n, M = construir_prefijo(proc, modelo, usados, REFERENCIAS[:k], disp)
            segs = sum(len(c) for c in clips[:k]) / RITMO
            for clave, texto in PRUEBAS.items():
                for s in args.semillas:
                    x = hablar(proc, modelo, pref, texto, args.cfg, s, disp)
                    escribir_wav(salida / f"{voz}-k{k}-{clave}-s{s}.wav", x)
                    filas.append(dict(voz=voz, nivel=f"k{k}", clips=k, segundos=segs,
                                      posiciones=n + M, texto=clave, semilla=s,
                                      ecapa=float(huella(x) @ h_ref),
                                      wer=wer(texto, ""), **descripcion(x)))
            v = [f["ecapa"] for f in filas if f["voz"] == voz and f["nivel"] == f"k{k}"]
            print(f"  k={k}  {segs:5.1f} s  {n+M:4d} posiciones  "
                  f"ECAPA {np.mean(v):.4f} +-{np.std(v):.4f}", flush=True)

    # ------------------------------------------------------------- resumen --
    print(f"\n{'='*70}\nCURVA\n{'='*70}")
    print(f"{'voz':16} {'nivel':>7} {'seg':>6} {'pos':>6} {'ECAPA':>8} {'+-':>7} {'recorr':>7}")
    for voz in args.voces:
        for nivel in ["oraculo"] + [f"k{k}" for k in range(1, len(REFERENCIAS) + 1)]:
            v = [f for f in filas if f["voz"] == voz and f["nivel"] == nivel]
            if not v:
                continue
            e = np.array([f["ecapa"] for f in v])
            print(f"{voz:16} {nivel:>7} {v[0]['segundos']:6.1f} {v[0]['posiciones']:6d} "
                  f"{e.mean():8.4f} {e.std():7.4f} "
                  f"{np.mean([f['recorrido'] for f in v]):7.1f}")
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader(); w.writerows(filas)
        print(f"{len(filas)} filas en {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
