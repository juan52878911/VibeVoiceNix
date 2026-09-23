#!/usr/bin/env python3
"""Evalua un LoRA GENERANDO (no forzando): la unica puerta que vale, porque el exposure bias solo se
ve al generar. Mismo prefijo, mismas frases y mismas semillas para la base y para cada LoRA.

  python3 evaluar.py --datos datos/ --voces voces.json --salida eval/base
  python3 evaluar.py --datos datos/ --voces voces.json --salida eval/c1 --lora corrida1/lora_mejor.pt

Genera:
  - por idioma, los hablantes apartados por datos.py: clon desde su clip de referencia, diciendo las
    frases de las que existe su audio real (identidad contra ese audio);
  - las voces con consentimiento de voces.json ({nombre: {"refs": [{audio, transcripcion}], "reales":
    [wav...], "frases": {"es": [...], "en": [...]}}}).
Deja <salida>/wav/*.wav, <salida>/lote.json para juez_lote.py y <salida>/identidades/<id>/*.wav.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402
import lora as LR  # noqa: E402
from validar_forzado import prefijo  # noqa: E402


def generar(modelo, proc, pref, texto, semilla, cfg=3.0):
    entradas = proc.process_input_with_cached_prompt(
        text=texto.strip() + "\n", cached_prompt=copy.deepcopy(pref),
        padding=True, return_tensors="pt", return_attention_mask=True)
    d = next(modelo.parameters()).device
    torch.manual_seed(semilla)
    with torch.no_grad():
        out = modelo.generate(**{k: (v.to(d) if hasattr(v, "to") else v) for k, v in entradas.items()},
                              max_new_tokens=None, cfg_scale=cfg, tokenizer=proc.tokenizer,
                              generation_config={"do_sample": False}, verbose=False,
                              all_prefilled_outputs=copy.deepcopy(pref), show_progress_bar=False)
    x = out.speech_outputs[0]
    return None if x is None else x.float().cpu().numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--voces", default=None)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--alfa", type=int, default=32)
    ap.add_argument("--ramas", default="language_model,tts_language_model")
    ap.add_argument("--semillas", default="11,101")
    ap.add_argument("--pasos", type=int, default=6, help="pasos de difusion (produccion: 6)")
    ap.add_argument("--idiomas", default=None)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    d = "cuda" if torch.cuda.is_available() else "cpu"
    proc, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    if a.lora:
        LR.poner(modelo, a.rango, a.alfa, 0.0, tuple(a.ramas.split(",")))
        n, _ = LR.cargar(modelo, a.lora)
        LR.fundir(modelo)
        print(f"[eval] LoRA {a.lora}: {n} tensores fundidos", flush=True)
    modelo.eval()
    modelo.set_ddpm_inference_steps(num_steps=a.pasos)
    tok = proc.tokenizer
    sal = Path(a.salida)
    (sal / "wav").mkdir(parents=True, exist_ok=True)
    semillas = [int(s) for s in a.semillas.split(",")]
    lote = []
    trabajos = []           # (identidad, idioma, refs [(wav, txt)], [(clave, texto)], reales [wav])
    ev = json.loads((Path(a.datos) / "evaluacion.json").read_text())
    for idioma, hs in ev.items():
        if a.idiomas and idioma not in a.idiomas.split(","):
            continue
        for h in hs:
            trabajos.append((f"{idioma}-{h['hablante']}", idioma, [(h["ref"], h["ref_txt"])],
                             [(f"f{k}", f["texto"], idioma) for k, f in enumerate(h["frases"])],
                             [f["audio"] for f in h["frases"]]))
    if a.voces:
        for nombre, v in json.loads(Path(a.voces).read_text()).items():
            frases = [(f"{lg}{k}", t, lg) for lg, ts in v["frases"].items() for k, t in enumerate(ts)]
            trabajos.append((nombre, "mix", [(r["audio"], r["transcripcion"]) for r in v["refs"]], frases, v["reales"]))
    for ident, idioma, refs, frases, reales in trabajos:
        dest = sal / "identidades" / ident
        dest.mkdir(parents=True, exist_ok=True)
        for k, w in enumerate(reales):
            shutil.copy(w, dest / f"real{k}.wav")
        torch.manual_seed(11)
        lat_ref = torch.cat([FZ.latentes(modelo, sf.read(w, dtype="float32")[0], d, muestrear=True) for w, _ in refs])
        ref_txt = "".join(t if t.endswith("\n") else t + "\n" for _, t in refs)
        pref = prefijo(modelo, tok, lat_ref, ref_txt, d)
        for clave, texto, lg in frases:
            for s in semillas:
                nombre = f"{ident}__{clave}__s{s}"
                w = sal / "wav" / f"{nombre}.wav"
                if not w.exists():
                    x = generar(modelo, proc, pref, texto, s)
                    if x is None:
                        continue
                    sf.write(str(w), x, 24000, subtype="PCM_16")
                lote.append({"clave": nombre, "audio": str(w), "texto": texto, "idioma": lg, "identidad": ident})
        print(f"[eval] {ident}: {len(frases)} frases x {len(semillas)} semillas", flush=True)
    (sal / "lote.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    print(f"[eval] {len(lote)} clips en {sal}", flush=True)


if __name__ == "__main__":
    main()
