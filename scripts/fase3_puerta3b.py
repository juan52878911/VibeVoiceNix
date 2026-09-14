#!/usr/bin/env python
"""Fase 3b, puerta: con y sin adaptador, emparejados por texto y semilla (lo que no se mide en la VM).

    pkgs/vibevoice/.venv/bin/python scripts/fase3_puerta3b.py --ab dir/ab --reales dir/reales

--ab es la carpeta de fase3_ab.py (base/, adaptador/, juez_vm.json); --reales tiene <persona>/*.wav con su
audio real, solo para la huella ECAPA de la persona (se borra al acabar: son datos biométricos).

Por clip y variante: WER con faster-whisper large-v3 contra el texto pedido, UTMOS22, coseno ECAPA contra
la huella media de su audio real; de juez_vm.json, P(real) del juez de la fase 2b y la distancia de los 8
descriptores al perfil real (media de |valor − media real| / desviación real).

PUERTA (fijada antes de medir, docs/plan-personalidad-voz.md), por persona evaluable, diferencia emparejada
adaptador − base con IC 95 % por bootstrap: P(real) IC inferior > 0; distancia IC superior < 0; ECAPA media
≥ −0,005; UTMOS media ≥ −0,02; WER media ≤ +0,5 puntos. Pasa si se cumple todo en todas las evaluables.
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
    d = np.asarray([v for v in d if v == v], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    medias = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ab", required=True)
    ap.add_argument("--reales", required=True)
    ap.add_argument("--whisper", default="large-v3")
    a = ap.parse_args()
    import librosa
    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(os.cpu_count() or 8)
    ab = Path(a.ab)
    vm = json.load(open(ab / "juez_vm.json", encoding="utf-8"))
    frases = json.load(open(ab / "base" / "frases.json", encoding="utf-8"))
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

    def transcribir(ruta):
        segs, _ = whisper.transcribe(str(ruta), language="es", beam_size=5, vad_filter=False,
                                     condition_on_previous_text=False, temperature=0.0)
        return " ".join(s.text.strip() for s in segs).strip()

    real = {}
    for p in vm["evaluables"]:
        hs = [huella(*NA_leer(r)) for r in sorted((Path(a.reales) / p).glob("*.wav"))]
        m = np.mean(hs, 0)
        real[p] = m / np.linalg.norm(m)
        print(f"huella real de {p}: {len(hs)} clips", flush=True)

    medidas = {}
    for fichero, meta in sorted(vm["clips"].items()):
        p = meta["voz"]
        for variante in ("base", "adaptador"):
            ruta = ab / variante / fichero
            x, sr = NA_leer(ruta)
            desc = meta[variante]["descriptores"]
            perfil = vm["perfil_real"][p]
            dist = [abs(desc[k] - perfil[k]["media"]) / perfil[k]["desv"]
                    for k in desc if desc[k] is not None and perfil[k]["desv"] > 0]
            medidas.setdefault(fichero, {"voz": p})[variante] = {
                "wer": 100 * FI.wer(frases[meta["frase"]], transcribir(ruta)),
                "utmos": NA.utmos_de(utmos, x, sr),
                "ecapa": float(np.dot(huella(x, sr), real[p])),
                "p_real": meta[variante]["p_real"],
                "distancia": float(np.mean(dist)) if dist else float("nan"),
            }
        print(f"  {fichero}: " + " · ".join(f"{k} {medidas[fichero]['base'][k]:.3f}->{medidas[fichero]['adaptador'][k]:.3f}"
                                             for k in ("p_real", "distancia", "ecapa", "utmos", "wer")
                                             if medidas[fichero]["base"][k] is not None), flush=True)

    informe, pasa = {}, True
    for p in vm["evaluables"]:
        filas = [m for m in medidas.values() if m["voz"] == p]
        dif = {k: [f["adaptador"][k] - f["base"][k] for f in filas
                   if f["adaptador"][k] is not None and f["base"][k] is not None]
               for k in ("p_real", "distancia", "ecapa", "utmos", "wer")}
        r = {k: dict(zip(("media", "ic_inf", "ic_sup"), ic95(v)), n=len(v)) for k, v in dif.items()}
        r["base"] = {k: float(np.nanmean([f["base"][k] for f in filas if f["base"][k] is not None])) for k in dif}
        criterios = {"p_real": r["p_real"]["ic_inf"] > 0, "distancia": r["distancia"]["ic_sup"] < 0,
                     "ecapa": r["ecapa"]["media"] >= -0.005, "utmos": r["utmos"]["media"] >= -0.02,
                     "wer": r["wer"]["media"] <= 0.5}
        r["criterios"] = criterios
        r["pasa"] = all(criterios.values())
        pasa = pasa and r["pasa"]
        informe[p] = r
    json.dump({"personas": informe, "pasa": pasa, "clips": medidas}, open(ab / "puerta3b.json", "w"),
              ensure_ascii=False, indent=1)

    print("\n== puerta 3b (adaptador − base, media [IC 95 %])")
    for p, r in informe.items():
        print(f"   {p} (n={r['p_real']['n']}):")
        for k in ("p_real", "distancia", "ecapa", "utmos", "wer"):
            print(f"     {k:10s} base {r['base'][k]:.3f} · dif {r[k]['media']:+.4f} [{r[k]['ic_inf']:+.4f}, {r[k]['ic_sup']:+.4f}]"
                  f" · {'ok' if r['criterios'][k] else 'NO'}")
    print(f"   {'PUERTA 3b PASA' if pasa else 'PUERTA 3b NO PASA'} · {ab / 'puerta3b.json'}")


def NA_leer(ruta):
    import soundfile as sf
    x, sr = sf.read(str(ruta), dtype="float32")
    return (x.mean(1) if x.ndim > 1 else x), sr


if __name__ == "__main__":
    main()
