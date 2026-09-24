#!/usr/bin/env python3
"""F4 del plan de mejora: ACENTO a voluntad por conversion de voz (kNN-VC, MIT).

La idea: el acento lo pone la FUENTE (una voz que ya habla como queremos) y el timbre lo pone la
conversion hacia la persona. Asi salen los cuatro cuadrantes sin datos apareados:

  fuente nativa inglesa (en-Carter / en-Emma) en ingles  -> la persona hablando ingles NATIVO
  fuente espanola (sp-Spk3 / sp-Spk0) en ingles           -> la persona, ingles con ACENTO ESPANOL
  fuente inglesa en espanol                               -> la persona, espanol con ACENTO INGLES
  (el clon directo de la persona sigue siendo el control)

kNN-VC cambia cada fotograma de WavLM de la fuente por la media de sus k vecinos en el audio REAL
de la persona (el "banco de timbre"), y un HiFi-GAN los devuelve a onda. No entrena nada: solo
necesita minutos de audio real de la persona, que no se usan para evaluar.

MEDIDO (23-09, Juan, banco de 1,6 min en espanol): el acento baja hacia el nativo (PER 0,282 -> 0,203)
pero la identidad cae al 71 % del clon directo: en el banco no hay fonemas ingleses con su timbre y
los vecinos se los presta la fuente. --extra suma el clon de la persona hablando ingles.

  python3 f4_conversion.py --fuentes wav/ --objetivos objetivos/ --salida convertidos/ \
      --plan plan.json            # [{"fuente": "en-Carter_man__en_0__s11.wav", "objetivo": "juan"}, ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuentes", required=True)
    ap.add_argument("--objetivos", required=True, help="<identidad>.wav: el banco de timbre de cada persona")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--minutos", type=float, default=None, help="recortar el banco de timbre (curva de tamano)")
    ap.add_argument("--extra", default=None,
                    help="carpeta con <identidad>__*.wav que se SUMAN al banco: el clon de la persona hablando "
                         "la lengua destino, para que haya vecinos de esos fonemas con su timbre")
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True,
                         device=a.dispositivo)
    plan = json.loads(Path(a.plan).read_text())
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    bancos, tiempos = {}, []
    for p in plan:
        nombre = f"{p['objetivo']}__desde__{Path(p['fuente']).stem}.wav"
        if (sal / nombre).exists():
            continue
        if p["objetivo"] not in bancos:
            # En trozos de 30 s: WavLM sobre 11 minutos de golpe no cabe en los 15 GB de una T4
            x, hz = torchaudio.load(str(Path(a.objetivos) / f"{p['objetivo']}.wav"))
            if a.minutos:                                  # banco recortado a N minutos
                x = x[:, :int(a.minutos * 60 * hz)]
            trozos = []
            for k, ini in enumerate(range(0, x.shape[1], 30 * hz)):
                t = sal / f"_banco_{p['objetivo']}_{k:03d}.wav"
                torchaudio.save(str(t), x[:, ini:ini + 30 * hz], hz)
                trozos.append(str(t))
            if a.extra:
                trozos += sorted(str(w) for w in Path(a.extra).glob(f"{p['objetivo']}__*.wav")
                                 if sf.info(str(w)).duration > 0.5)       # algun clip del banco sale vacio
            # vad_trigger_level=0: el recorte de silencios de kNN-VC (7 por defecto) deja a cero algunos
            # clips cortos del clon (1,7 s) y revienta; el audio real ya viene sin silencios largos
            bancos[p["objetivo"]] = knn.get_matching_set(trozos, vad_trigger_level=0 if a.extra else 7)
            for t in trozos:
                if Path(t).name.startswith("_banco_"):
                    Path(t).unlink()
        t0 = time.time()
        consulta = knn.get_features(str(Path(a.fuentes) / p["fuente"]))
        y = knn.match(consulta, bancos[p["objetivo"]], topk=a.topk)
        sf.write(str(sal / nombre), y.cpu().numpy(), 16000, subtype="PCM_16")
        dur = y.shape[-1] / 16000
        tiempos.append((time.time() - t0, dur))
        print(f"[f4] {nombre}  {dur:.1f} s en {tiempos[-1][0]:.2f} s", flush=True)
    if tiempos:
        calc, audio = sum(t for t, _ in tiempos), sum(d for _, d in tiempos)
        print(f"[f4] RTF de la conversion en {a.dispositivo}: {calc / audio:.3f} ({audio:.0f} s de audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
