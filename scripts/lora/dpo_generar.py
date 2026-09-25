#!/usr/bin/env python3
"""DPO, paso 1 (plan de preferencias, 24-09): N semillas por (voz, texto) con el modelo como en produccion
(CFG 3, freno 0,75, 6 pasos). Se guardan los LATENTES que genero la cabeza (lo que DPO necesita: no se
re-codifica el audio), el wav y los metadatos.

  python3 dpo_generar.py --salida d0/ --hablantes en:3,es:3,fr:1,de:1,it:1,pt:1 --por-hablante 5 --preparar
  python3 dpo_generar.py --salida d0/ --semillas 11,22,33,44 --trozo 0/2     # y 1/2 en paralelo
  (puerta)  ... --saltar 4 --parte puerta [--lora d2/b2000/lora_mejor.pt --alfa 32 --cuantizar]

Voces: lectores de LibriTTS-R (en) y CML-TTS (resto), las fuentes de datos.py. Por lector, una referencia
(el prefijo de voz, como al clonar) y 3 clips reales suyos para la identidad (ECAPA en juez_lote.py).
--saltar N se salta los N primeros lectores validos de cada idioma: la puerta usa hablantes que el
entrenamiento no ve. Textos por lector: un tercio lectura (frases reales de OTROS lectores del mismo
idioma), un tercio dificil y un tercio de tramos de dobla que fallaron el QC (dpo_textos.json; solo hay
en ingles: en el resto, dificil). --parte separa los textos de entrenamiento y los de la puerta por hash.
El texto pasa por normalizar_para_motor, como en voz-stream.

Deja: hablantes.json, hablantes/<id>/{ref.wav, ref_lat.pt}, identidades/<id>/real*.wav, wav/<clave>.wav,
lat/<clave>.pt ([T,64] fp16) y lote.<trozo>.json (entrada de juez_lote.py). Reanudable.
"""
import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402
import lora as LR  # noqa: E402

# por idioma: (duracion maxima, similitud minima de CML-TTS, splits); el portugues va aparte (ver datos.py)
FUENTE = {"pt": (21.0, 0.9, ("train", "dev", "test"))}
VOCALES = re.compile(r"[aeiouyáéíóúàèìòùâêîôûäëïöüãõœ]+", re.I)
MUESTRAS = 3200                     # audio por latente (24000 / 7,5)


def parte_de(texto):
    """'puerta' para ~1 de cada 5 textos, fijo por el propio texto; el resto, 'entreno'."""
    return "puerta" if int(hashlib.md5(texto.encode()).hexdigest(), 16) % 5 == 0 else "entreno"


def silabas(texto):
    return max(1, len(VOCALES.findall(texto)))


def reunir_lectores(idioma, n, saltar, max_textos=400):
    """Los lectores saltar..saltar+n del flujo (en el orden en que completan 4 clips) y frases de lectura de
    otros lectores. Devuelve ([(hablante, [(audio24k, texto)])], [texto])."""
    from datos import MIN_S, MAX_S, a24, flujo
    max_s, lev, splits = FUENTE.get(idioma, (MAX_S, 0.95, ("train",)))
    por_h, completos, otros = defaultdict(list), [], []
    for leidas, (decodificar, texto, h, dur) in enumerate(flujo(idioma, lev, splits)):
        if leidas > 60000:
            break
        if not texto or len(texto) < 10:
            continue
        if len(completos) >= n + saltar:
            if len(otros) >= max_textos:
                break
            if h not in {c for c, _ in completos} and 40 <= len(texto) <= 220:
                otros.append(texto.strip())
            continue
        if h in {c for c, _ in completos}:
            continue
        if dur is not None and not (MIN_S <= dur <= max_s):
            continue
        if h not in por_h and len(por_h) >= 8 * (n + saltar):
            if 40 <= len(texto) <= 220:
                otros.append(texto.strip())
            continue
        try:
            audio = decodificar()
        except Exception:
            continue
        if not (MIN_S <= len(audio["array"]) / audio["sampling_rate"] <= max_s):
            continue
        por_h[h].append((a24(audio).astype(np.float16), texto.strip()))
        if len(por_h[h]) >= 4:
            completos.append((h, por_h.pop(h)))
    for h, clips in por_h.items():                 # lectores a medias: sus frases valen como lectura
        otros += [t for _, t in clips if 40 <= len(t) <= 220]
    return completos[saltar:saltar + n], otros


def preparar(a, sal):
    textos = json.loads((AQUI / "dpo_textos.json").read_text())
    rng = random.Random(0)
    hablantes = {}
    (sal / "hablantes").mkdir(parents=True, exist_ok=True)
    for pieza in a.hablantes.split(","):
        idioma, n = pieza.split(":")
        lectores, lectura = reunir_lectores(idioma, int(n), a.saltar)
        lectura = sorted({t for t in lectura if parte_de(t) == a.parte})
        rng.shuffle(lectura)
        dificil = [t for t in textos["dificil"].get(idioma, []) if parte_de(t) == a.parte]
        if a.parte == "puerta":
            dificil += textos.get("dificil_puerta", {}).get(idioma, [])
        qc = [t for t in textos.get("qc", {}).get(idioma, []) if parte_de(t) == a.parte]
        rng.shuffle(dificil)
        rng.shuffle(qc)
        print(f"[gen] {idioma}: {len(lectores)} lectores; textos {a.parte}: lectura {len(lectura)}, "
              f"dificil {len(dificil)}, qc {len(qc)}", flush=True)
        k = a.por_hablante
        n_qc = round(k / 3) if qc else 0
        n_dif = round(k / 3) + (round(k / 3) if not qc else 0)
        for i, (h, clips) in enumerate(lectores):
            ident = f"{idioma}-{h}"
            dest = sal / "hablantes" / ident
            ids = sal / "identidades" / ident
            dest.mkdir(parents=True, exist_ok=True)
            ids.mkdir(parents=True, exist_ok=True)
            clips = sorted(clips, key=lambda c: -len(c[0]))
            sf.write(str(dest / "ref.wav"), clips[0][0].astype(np.float32), 24000)
            reales = []
            for j, (x, t) in enumerate(clips[1:4]):
                sf.write(str(ids / f"real{j}.wav"), x.astype(np.float32), 24000)
                reales.append({"audio": str(ids / f"real{j}.wav"), "texto": t, "dur": len(x) / 24000})
            elegidos = ([("qc", qc[(i * n_qc + j) % len(qc)]) for j in range(n_qc)] if qc else []) + \
                       [("dificil", dificil[(i * n_dif + j) % len(dificil)]) for j in range(n_dif)] + \
                       [("lectura", lectura.pop()) for _ in range(k - n_qc - n_dif)]
            hablantes[ident] = {"idioma": idioma, "hablante": h, "ref": str(dest / "ref.wav"),
                                "ref_txt": clips[0][1] + "\n", "reales": reales,
                                "textos": [{"clave": f"t{j}", "tipo": tp, "texto": t} for j, (tp, t) in enumerate(elegidos)]}
    (sal / "hablantes.json").write_text(json.dumps(hablantes, ensure_ascii=False, indent=1))
    print(f"[gen] {len(hablantes)} lectores, {sum(len(v['textos']) for v in hablantes.values())} textos", flush=True)


def capturar(modelo):
    """Envuelve sample_speech_tokens (ya con el freno): apunta cada latente que sale de la cabeza."""
    lista = []
    orig = modelo.sample_speech_tokens

    def sample_speech_tokens(condition, neg_condition, cfg_scale=3.0):
        out = orig(condition, neg_condition, cfg_scale=cfg_scale)
        lista.append(out[0].detach().float().cpu())
        return out
    modelo.sample_speech_tokens = sample_speech_tokens
    return lista


def generar(modelo, proc, pref, texto, semilla, lista, tope, cfg=3.0):
    """evaluar.generar con un tope de fotogramas (un desbocado no se come la tanda) y los latentes.
    Devuelve (audio, latentes [T,64], se_paro_por_tope)."""
    lista.clear()
    entradas = proc.process_input_with_cached_prompt(
        text=texto.strip() + "\n", cached_prompt=copy.deepcopy(pref),
        padding=True, return_tensors="pt", return_attention_mask=True)
    d = next(modelo.parameters()).device
    torch.manual_seed(semilla)
    with torch.no_grad():
        out = modelo.generate(**{k: (v.to(d) if hasattr(v, "to") else v) for k, v in entradas.items()},
                              max_new_tokens=None, cfg_scale=cfg, tokenizer=proc.tokenizer,
                              generation_config={"do_sample": False}, verbose=False,
                              all_prefilled_outputs=copy.deepcopy(pref), show_progress_bar=False,
                              stop_check_fn=lambda: len(lista) >= tope)
    x = out.speech_outputs[0]
    if x is None:
        return None, None, False
    x = x.float().cpu().numpy().reshape(-1)
    # tras el fin, generate() aun muestrea el resto de la ventana de 6 pero ya no lo decodifica: el audio
    # tiene un trozo de 3200 muestras por latente valido
    T = round(len(x) / MUESTRAS)
    assert T <= len(lista) and abs(len(x) - T * MUESTRAS) < MUESTRAS // 4, (len(x), len(lista))
    return x, torch.stack(lista[:T]), len(lista) >= tope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--preparar", action="store_true", help="elegir lectores y textos (una vez, antes de generar)")
    ap.add_argument("--hablantes", default="en:3,es:3,fr:1,de:1,it:1,pt:1")
    ap.add_argument("--por-hablante", type=int, default=5)
    ap.add_argument("--saltar", type=int, default=0)
    ap.add_argument("--parte", default="entreno", choices=("entreno", "puerta"))
    ap.add_argument("--semillas", default="11,22,33,44")
    ap.add_argument("--trozo", default="0/1", help="i/n: este proceso hace los trabajos i, i+n, ...")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--alfa", type=int, default=32, help="fuerza al fundir (alfa/rango)")
    ap.add_argument("--ramas", default="language_model,tts_language_model")
    ap.add_argument("--cuantizar", action="store_true")
    ap.add_argument("--freno", type=float, default=0.75)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    sal = Path(a.salida).resolve()                 # rutas absolutas en hablantes.json y lote
    sal.mkdir(parents=True, exist_ok=True)
    if a.preparar:
        preparar(a, sal)
        return
    from normalizar_texto import normalizar_para_motor
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    from evaluar import cuantizar_como_produccion, instalar_freno
    from validar_forzado import prefijo
    d = "cuda" if torch.cuda.is_available() else "cpu"
    proc, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    if a.lora:
        LR.poner(modelo, a.rango, a.alfa, 0.0, tuple(a.ramas.split(",")))
        n, _ = LR.cargar(modelo, a.lora)
        LR.fundir(modelo)
        print(f"[gen] LoRA {a.lora}: {n} tensores fundidos con alfa {a.alfa} (rango {a.rango})", flush=True)
    if a.cuantizar:
        cuantizar_como_produccion(modelo)
    instalar_freno(modelo, a.freno)
    lista = capturar(modelo)
    modelo.eval()
    modelo.set_ddpm_inference_steps(num_steps=a.pasos)
    tok = proc.tokenizer
    for sub in ("wav", "lat"):
        (sal / sub).mkdir(exist_ok=True)
    hablantes = json.loads((sal / "hablantes.json").read_text())
    semillas = [int(s) for s in a.semillas.split(",")]
    i_trozo, n_trozos = (int(v) for v in a.trozo.split("/"))
    trabajos = [(ident, t) for ident, h in hablantes.items() for t in h["textos"]]
    trabajos = trabajos[i_trozo::n_trozos]
    lote, por_ident = [], defaultdict(list)
    for ident, t in trabajos:
        por_ident[ident].append(t)
    for ident, ts in por_ident.items():
        h = hablantes[ident]
        ruta_ref = sal / "hablantes" / ident / "ref_lat.pt"
        torch.manual_seed(11)                          # como evaluar.py: el mismo prefijo en base y LoRA
        lat_ref = FZ.latentes(modelo, sf.read(h["ref"], dtype="float32")[0], d, muestrear=True)
        if not ruta_ref.exists():
            ruta_ref.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lat_ref.cpu(), ruta_ref)
        pref = prefijo(modelo, tok, lat_ref, h["ref_txt"], d)
        modo = h["idioma"] if h["idioma"] in ("es", "en") else "no"
        for t in ts:
            texto = normalizar_para_motor(t["texto"], modo)
            tope = max(120, int(3 * silabas(texto) / 4.0 * 7.5))       # 3 veces lo esperado a 4 silabas/s
            for s in semillas:
                clave = f"{ident}__{t['clave']}__s{s}"
                w, fl = sal / "wav" / f"{clave}.wav", sal / "lat" / f"{clave}.pt"
                meta = {"clave": clave, "audio": str(w), "texto": texto, "idioma": h["idioma"], "identidad": ident,
                        "grupo": f"{ident}__{t['clave']}", "semilla": s, "tipo": t["tipo"]}
                if not (w.exists() and fl.exists()):
                    x, lat, tope_ = generar(modelo, proc, pref, texto, s, lista, tope, a.cfg)
                    if x is None:
                        print(f"[gen] {clave}: sin audio", flush=True)
                        continue
                    sf.write(str(w), x, 24000, subtype="PCM_16")
                    torch.save({"lat": lat.half(), "tope": tope_}, fl)
                lote.append(meta)
        print(f"[gen] {ident}: {len(ts)} textos x {len(semillas)} semillas", flush=True)
        (sal / f"lote.{i_trozo}.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    (sal / f"lote.{i_trozo}.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    print(f"[gen] {len(lote)} clips en {sal}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)      # datasets revienta al cerrar el interprete (ver datos.py)
