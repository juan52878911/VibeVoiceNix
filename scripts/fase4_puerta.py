#!/usr/bin/env python
"""Fase 4, puerta: pausas y ritmo contra la base, con cobertura, WER, UTMOS e identidad.

    pkgs/vibevoice/.venv/bin/python scripts/fase4_puerta.py --dir dir/fase4 --reales dir/reales
    pkgs/vibevoice/.venv/bin/python scripts/fase4_puerta.py --dir dir/fase4b --reales dir/reales --primario silabas

--dir es la salida de fase4_pausas.py (una carpeta por variante y medidas_vm.json); --reales tiene
<persona>/*.wav con su audio real, solo para la huella ECAPA (se borra al acabar: datos biométricos).
Guarda lo medido por clip en <dir>/puerta4_cache.json, así que se puede relanzar a mitad.

PUERTA (fijada antes de medir, docs/plan-personalidad-voz.md, fases 4 y 4b), cada variante contra base:
  objetivo (Carlos), clips apartados:
      --primario pausas (fase 4):  |pausas/min − real| baja con IC 95 % superior < 0 y |sílabas/s − real|
                                   no sube de media
      --primario silabas (fase 4b): |sílabas/s − real| baja con IC 95 % superior < 0 (las pausas se informan)
  control (Liliana), clips apartados: |pausas/min − real| no sube más de 2 de media; |sílabas/s − real|
      no más de 0,3
  todas las personas, todos los clips: WER medio sin subir más de 0,5 puntos; NINGÚN clip con WER > 25 %
      si en la base tenía <= 10 %; ninguno con las palabras transcritas fuera de 0,85-1,15 veces las del
      texto si en la base estaba dentro; UTMOS medio >= −0,05; ECAPA contra su audio real >= −0,01
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fidelidad as FI  # noqa: E402
import naturalidad as NA  # noqa: E402


def ic95(d, n=2000):
    d = np.asarray([v for v in d if v is not None and v == v], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    medias = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def leer(ruta):
    import soundfile as sf
    x, sr = sf.read(str(ruta), dtype="float32")
    return (x.mean(1) if x.ndim > 1 else x), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reales", required=True)
    ap.add_argument("--objetivo", default="carlos-segura")
    ap.add_argument("--control", nargs="*", default=["liliana-morales"])
    ap.add_argument("--primario", choices=("pausas", "silabas"), default="pausas")
    ap.add_argument("--whisper", default="large-v3")
    a = ap.parse_args()
    import librosa
    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(os.cpu_count() or 8)
    d = Path(a.dir)
    vm = json.load(open(d / "medidas_vm.json", encoding="utf-8"))
    variantes = vm.get("variantes", ["base", "trozos", "saltos", "trozos_r", "saltos_r"])
    frases = json.load(open(d / "base" / "frases.json", encoding="utf-8"))
    cache_ruta = d / "puerta4_cache.json"
    cache = json.load(open(cache_ruta)) if cache_ruta.exists() else {}

    whisper = WhisperModel(a.whisper, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 8)
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
                                           run_opts={"device": "cpu"})
    utmos = NA.cargar_juez()

    def huella(x, sr):
        x16 = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x16).float()[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)

    real = {}
    for p in vm["personas"]:
        hs = [huella(*leer(r)) for r in sorted((Path(a.reales) / p).glob("*.wav"))]
        m = np.mean(hs, 0)
        real[p] = m / np.linalg.norm(m)
        print(f"huella real de {p}: {len(hs)} clips", flush=True)

    for i, (fichero, fila) in enumerate(sorted(vm["clips"].items()), 1):
        texto = frases[fila["frase"]]
        for var in variantes:
            clave = f"{var}/{fichero}"
            if clave in cache:
                continue
            ruta = d / var / fichero
            x, sr = leer(ruta)
            segs, _ = whisper.transcribe(str(ruta), language="es", beam_size=5, vad_filter=False,
                                         condition_on_previous_text=False, temperature=0.0)
            oido = " ".join(s.text.strip() for s in segs).strip()
            n_txt = len(FI.comparable(texto).split())
            cache[clave] = {"oido": oido, "wer": 100 * FI.wer(texto, oido),
                            "cobertura": len(FI.comparable(oido).split()) / max(n_txt, 1),
                            "utmos": NA.utmos_de(utmos, x, sr), "ecapa": float(np.dot(huella(x, sr), real[fila["voz"]]))}
        if i % 10 == 0:
            json.dump(cache, open(cache_ruta, "w"), ensure_ascii=False)
            print(f"  {i}/{len(vm['clips'])} clips", flush=True)
    json.dump(cache, open(cache_ruta, "w"), ensure_ascii=False)

    def m(var, fichero, k):
        return cache[f"{var}/{fichero}"][k]

    informe = {}
    for var in variantes[1:]:
        r = {"personas": {}}
        for p in vm["personas"]:
            clips = {f: fila for f, fila in vm["clips"].items() if fila["voz"] == p}
            ap_ = {f: fila for f, fila in clips.items() if fila["tipo"] == "apartado"}
            rv = vm["reales_val"][p]

            def dist(fila, k, v):
                a_, b_ = fila["variantes"][v][k], rv[fila["clip_real"]][k]
                return None if a_ is None or b_ is None else abs(a_ - b_)

            def dif(fila, k):
                dv, db = dist(fila, k, var), dist(fila, k, "base")
                return None if dv is None or db is None else dv - db

            dp = [dif(fl, "pausas_min") for fl in ap_.values()]
            ds = [dif(fl, "silabas_s") for fl in ap_.values()]
            dm = [dif(fl, "mediana_pausa") for fl in ap_.values()]
            wer = [m(var, f, "wer") - m("base", f, "wer") for f in clips]
            catastrofes = [f for f in clips if m(var, f, "wer") > 25 and m("base", f, "wer") <= 10]
            cobertura = [f for f in clips if not 0.85 <= m(var, f, "cobertura") <= 1.15
                         and 0.85 <= m("base", f, "cobertura") <= 1.15]
            ut = [m(var, f, "utmos") - m("base", f, "utmos") for f in clips]
            ec = [m(var, f, "ecapa") - m("base", f, "ecapa") for f in clips]
            fila = {"dist_pausas": ic95(dp), "dist_silabas": ic95(ds), "dist_mediana_pausa": ic95(dm),
                    "wer": ic95(wer), "utmos": ic95(ut), "ecapa": ic95(ec), "catastrofes": catastrofes,
                    "fuera_de_cobertura": cobertura, "n_apartados": len(ap_), "n_clips": len(clips),
                    "media": {k: float(np.mean([fl["variantes"][var][k] for fl in ap_.values()]))
                              for k in ("pausas_min", "silabas_s")},
                    "media_base": {k: float(np.mean([fl["variantes"]["base"][k] for fl in ap_.values()]))
                                   for k in ("pausas_min", "silabas_s")},
                    "media_real": {k: float(np.mean([rv[fl["clip_real"]][k] for fl in ap_.values()]))
                                   for k in ("pausas_min", "silabas_s")}}
            crit = {"wer": fila["wer"][0] <= 0.5, "sin_catastrofes": not catastrofes, "cobertura": not cobertura,
                    "utmos": fila["utmos"][0] >= -0.05, "ecapa": fila["ecapa"][0] >= -0.01}
            if p == a.objetivo:
                if a.primario == "pausas":
                    crit["pausas"] = fila["dist_pausas"][2] < 0
                    crit["silabas"] = fila["dist_silabas"][0] <= 0
                else:
                    crit["silabas"] = fila["dist_silabas"][2] < 0
            elif p in a.control:
                crit["pausas"] = fila["dist_pausas"][0] <= 2.0
                crit["silabas"] = fila["dist_silabas"][0] <= 0.3
            fila["criterios"] = crit
            fila["pasa"] = all(crit.values())
            r["personas"][p] = fila
        r["pasa"] = all(f["pasa"] for f in r["personas"].values())
        informe[var] = r
    que_pasan = [v for v in informe if informe[v]["pasa"]]
    clave_orden = "dist_pausas" if a.primario == "pausas" else "dist_silabas"
    mejor = min(que_pasan, key=lambda v: informe[v]["personas"][a.objetivo][clave_orden][0]) if que_pasan else None
    json.dump({"variantes": informe, "pasan": que_pasan, "elegida": mejor, "primario": a.primario},
              open(d / "puerta4.json", "w"), ensure_ascii=False, indent=1)

    print(f"\n== puerta (primario {a.primario}; variante − base; apartados para pausas y sílabas, todos los clips para el resto)")
    for var, r in informe.items():
        print(f"\n  {var}: {'PASA' if r['pasa'] else 'no pasa'}")
        for p, f in r["personas"].items():
            print(f"    {p} (apartados {f['n_apartados']}, clips {f['n_clips']}) · pausas/min {f['media_base']['pausas_min']:.1f} -> "
                  f"{f['media']['pausas_min']:.1f} [real {f['media_real']['pausas_min']:.1f}] · silabas/s {f['media_base']['silabas_s']:.2f} -> "
                  f"{f['media']['silabas_s']:.2f} [real {f['media_real']['silabas_s']:.2f}]")
            for k in ("dist_pausas", "dist_silabas", "dist_mediana_pausa", "wer", "utmos", "ecapa"):
                print(f"      {k:18s} {f[k][0]:+.3f} [{f[k][1]:+.3f}, {f[k][2]:+.3f}]")
            print(f"      catastrofes {len(f['catastrofes'])} · fuera de cobertura {len(f['fuera_de_cobertura'])} · "
                  + " ".join(f"{k}={'ok' if v else 'NO'}" for k, v in f["criterios"].items()))
    print(f"\n  pasan {que_pasan} · elegida {mejor}")


if __name__ == "__main__":
    main()
