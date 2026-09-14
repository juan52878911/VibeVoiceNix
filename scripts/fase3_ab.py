#!/usr/bin/env python
"""Fase 3b en la VM: el clon con y sin su adaptador diciendo lo mismo, y lo que se puede juzgar aquí.

    python scripts/fase3_ab.py --dataset /var/lib/taller/dataset-voces --voces /var/lib/taller/fase1/voces \
        --fase2 /var/lib/taller/fase2b --adaptador /var/lib/taller/fase3/adaptador.pt --salida /var/lib/taller/fase3/ab

Diseño fijado antes de medir (docs/plan-personalidad-voz.md): motor torch a los dos lados, cfg 3,0, 6 pasos,
semillas 101 y 7; por persona evaluable (al menos 5 clips apartados en la fase 2b), los textos de sus clips
apartados más las 6 frases nuevas de fase2_texto_nuevo.py. Misma semilla y mismo texto: lo único que cambia
es el adaptador delante de la cabeza de difusión, aplicado a las dos ramas de la guía (la condición y la
negativa), como lo haría una cabeza afinada.

Deja <salida>/base y <salida>/adaptador con un WAV por clip, clips.csv y frases.json (formato de
banco_ab.py), y <salida>/juez_vm.json con lo que se mide aquí: P(real) del juez de la fase 2b y los
descriptores de perfil_vocal.py de cada clip, más el perfil real de cada persona (media y desviación sobre
todos sus clips reales). WER, UTMOS e identidad ECAPA se miden en el Mac con fase3_puerta3b.py.
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fase2_estilo as F2  # noqa: E402
import perfil_vocal as PV  # noqa: E402
from fase2_texto_nuevo import FRASES_NUEVAS  # noqa: E402

SEMILLAS = (101, 7)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--voces", required=True)
    ap.add_argument("--fase2", required=True)
    ap.add_argument("--adaptador", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=os.environ.get("VIBEVOICE_MODELO"))
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--hilos", type=int, default=6)
    ap.add_argument("--personas", nargs="*", default=[], help="limitar a estas personas (por defecto, todas las evaluables)")
    a = ap.parse_args()
    import soundfile as sf
    import torch
    import torch.nn as nn
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    torch.set_num_threads(a.hilos)
    sal = Path(a.salida)

    # ------------------------------------------------------------ que se dice
    inf2 = json.load(open(Path(a.fase2) / "informe_fase2.json"))
    d2 = np.load(Path(a.fase2) / "caracteristicas.npz", allow_pickle=True)
    _, va = F2.particion(d2["C"], d2["P"], d2["R"])
    apartados = sorted({(inf2["personas"][int(d2["P"][i])], str(d2["C"][i])) for i in np.where(va)[0]})
    filas = list(csv.DictReader(open(Path(a.dataset) / "manifiesto.csv", encoding="utf-8")))
    texto_de = {(f["hablante"], Path(f["fichero"]).stem): f["texto"].strip() for f in filas}
    cuenta = {}
    for p, _ in apartados:
        cuenta[p] = cuenta.get(p, 0) + 1
    evaluables = sorted(p for p, n in cuenta.items() if n >= 5 and (not a.personas or p in a.personas))
    frases = {}
    for p in evaluables:
        for _, c in [x for x in apartados if x[0] == p]:
            frases[f"{p}__{c}"] = texto_de[(p, c)]
        for i, t in enumerate(FRASES_NUEVAS):
            frases[f"{p}__nuevo-{i}"] = t
    print(f"evaluables {evaluables} · {len(frases)} textos x {len(SEMILLAS)} semillas x 2 variantes", flush=True)

    # ------------------------------------------------------------ modelo y adaptador
    proc = VibeVoiceStreamingProcessor.from_pretrained(a.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    modelo.model.tts_language_model.embed_tokens = modelo.model.language_model.embed_tokens
    modelo.set_ddpm_inference_steps(a.pasos)
    cabeza = modelo.model.prediction_head
    estado_ad = torch.load(a.adaptador)

    class Adaptador(nn.Module):  # misma forma que en fase3_adaptador.py
        def __init__(self, prefijo):
            super().__init__()
            for nombre in ("beta", "gamma", "U", "V"):
                setattr(self, nombre, nn.Parameter(estado_ad[f"{prefijo}.{nombre}"], requires_grad=False))

        def forward(self, c):
            rms = c.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            h = c / rms
            return c + rms * (self.beta + self.gamma * h + (h @ self.V) @ self.U.T)

    class CabezaAdaptada(nn.Module):
        def __init__(self):
            super().__init__()
            self.cabeza, self.ad = cabeza, None

        @property
        def device(self):
            return next(self.cabeza.parameters()).device

        def forward(self, noisy_images, timesteps, condition):
            if self.ad is not None:
                condition = self.ad(condition)
            return self.cabeza(noisy_images, timesteps, condition=condition)

    envoltura = CabezaAdaptada()
    modelo.model.prediction_head = envoltura

    # ------------------------------------------------------------ generar
    prefijos, t0, n = {}, time.time(), 0
    for variante in ("base", "adaptador"):
        (sal / variante).mkdir(parents=True, exist_ok=True)
        filas_csv = []
        for clave, texto in frases.items():
            persona = clave.split("__")[0]
            if persona not in prefijos:
                prefijos[persona] = torch.load(Path(a.voces) / f"{persona}.pt", weights_only=False, map_location="cpu")
            envoltura.ad = Adaptador(persona.replace("-", "_")) if variante == "adaptador" else None
            for semilla in SEMILLAS:
                fichero = f"{clave}__s{semilla}.wav"
                ruta = sal / variante / fichero
                rtf = ""
                if not ruta.exists():
                    torch.manual_seed(semilla)
                    entradas = proc.process_input_with_cached_prompt(
                        text=texto + "\n", cached_prompt=copy.deepcopy(prefijos[persona]),
                        padding=True, return_tensors="pt", return_attention_mask=True)
                    ini = time.perf_counter()
                    with torch.no_grad():
                        salida = modelo.generate(**entradas, max_new_tokens=None, cfg_scale=a.cfg, tokenizer=proc.tokenizer,
                                                 generation_config={"do_sample": False}, verbose=False, return_speech=True,
                                                 all_prefilled_outputs=copy.deepcopy(prefijos[persona]))
                    x = salida.speech_outputs[0].detach().float().cpu().numpy().reshape(-1)
                    sf.write(str(ruta), x, F2.SR, subtype="PCM_16")
                    rtf = f"{(time.perf_counter() - ini) / (len(x) / F2.SR):.3f}"
                    n += 1
                    if n % 10 == 0:
                        print(f"  {n} clips · {(time.time() - t0) / 60:.1f} min · ultimo {variante} {fichero} RTF {rtf}", flush=True)
                filas_csv.append({"fichero": fichero, "frase": clave, "voz": persona, "semilla": semilla, "rtf": rtf})
        with open(sal / variante / "clips.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_csv[0]))
            w.writeheader()
            w.writerows(filas_csv)
        json.dump(frases, open(sal / variante / "frases.json", "w"), ensure_ascii=False, indent=1)
    del modelo

    # ------------------------------------------------------------ juez y descriptores
    juez = F2.construir_modelo(len(inf2["personas"]), len(F2.DESCRIPTORES))
    juez.load_state_dict(torch.load(Path(a.fase2) / "estilo.pt"))
    juez.eval()
    nm = inf2["normalizacion"]

    def p_real(x):
        frs = [f for f in F2.fragmentos(x) if np.sqrt(np.mean(f ** 2)) >= 1e-3]
        if not frs:
            return None
        mel = (np.stack([F2.mel(f) for f in frs]) - nm["mel_media"]) / nm["mel_desv"]
        with torch.no_grad():
            _, _, _, real = juez(torch.from_numpy(mel.astype(np.float32)))
        return float(torch.sigmoid(real).mean())

    def descriptores(x, sr, texto):
        pf = PV.perfil(x, sr, texto)
        return {k: (float(pf[k]) if pf[k] == pf[k] else None) for k in F2.DESCRIPTORES}

    perfil_real = {}
    for p in evaluables:
        vals = []
        for f in filas:
            if f["hablante"] == p:
                x, sr = sf.read(str(Path(a.dataset) / f["fichero"]), dtype="float32")
                vals.append(descriptores(x, sr, f["texto"]))
        perfil_real[p] = {k: {"media": float(np.nanmean([v[k] if v[k] is not None else np.nan for v in vals])),
                              "desv": float(np.nanstd([v[k] if v[k] is not None else np.nan for v in vals]))}
                          for k in F2.DESCRIPTORES}
    clips = {}
    for variante in ("base", "adaptador"):
        for fila in csv.DictReader(open(sal / variante / "clips.csv", encoding="utf-8")):
            x, sr = sf.read(str(sal / variante / fila["fichero"]), dtype="float32")
            clips.setdefault(fila["fichero"], {"voz": fila["voz"], "frase": fila["frase"]})[variante] = {
                "p_real": p_real(x), "descriptores": descriptores(x, sr, frases[fila["frase"]])}
    json.dump({"perfil_real": perfil_real, "clips": clips, "evaluables": evaluables, "argumentos": vars(a)},
              open(sal / "juez_vm.json", "w"), ensure_ascii=False, indent=1)
    for p in evaluables:
        b = [c["base"]["p_real"] for c in clips.values() if c["voz"] == p and c["base"]["p_real"] is not None]
        d = [c["adaptador"]["p_real"] for c in clips.values() if c["voz"] == p and c["adaptador"]["p_real"] is not None]
        print(f"   {p:18s} P(real) base {np.mean(b):.3f} · adaptador {np.mean(d):.3f} (n={len(b)})", flush=True)
    print(f"listo en {(time.time() - t0) / 60:.1f} min · {sal / 'juez_vm.json'}")


if __name__ == "__main__":
    main()
