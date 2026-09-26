#!/usr/bin/env python3
"""I2 · La costura del mando en tiempo real: que pasa en la ventana de +-2 s alrededor del fotograma donde entra.

    python3 scripts/red/costura.py <carpeta de dirigir.py --desde N> --desde 40 [--corpus a.json,b.json]

Por cada clip con lambda > 0 y su pareja lambda = 0 (misma voz, frase y semilla; identicos hasta el fotograma N):
  wer_ventana   whisper large-v3 con marcas por palabra sobre el clip entero; las palabras de la referencia se
                alinean con lo oido (Levenshtein) y cuentan los errores de las palabras cuyo instante cae en la ventana
  utmos_ventana UTMOS22 strong sobre la ventana recortada
Salida: tabla pareada (IC 95 %) por lambda y <carpeta>/costura.json. La ventana se mide en los DOS clips, asi que un
error que ya estaba en la base no se cuenta como costura.
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
from puerta_dirigir import IDIOMA, partes  # noqa: E402

HOP, HZ = 3200, 24000


def palabras(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9ñ']+", t)


def alinear(ref, hip):
    """Por cada palabra de la referencia: indice de la palabra oida emparejada (o la vecina, si se borro) y si hubo error."""
    n, m = len(ref), len(hip)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hip[j - 1]))
    i, j = n, m
    sal = [None] * n
    ins = defaultdict(int)          # inserciones atribuidas a la palabra de referencia siguiente
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hip[j - 1] and D[i, j] == D[i - 1, j - 1]:
            sal[i - 1] = (j - 1, False)                                   # acierto: primero
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            sal[i - 1] = (min(j, m - 1) if m else None, True)          # borrado: instante de la vecina
            i -= 1
        elif j > 0 and D[i, j] == D[i, j - 1] + 1:
            ins[min(i, n - 1)] += 1
            j -= 1
        else:
            sal[i - 1] = (j - 1, True)                                    # sustitucion
            i, j = i - 1, j - 1
    return sal, ins


def ic(v, n=4000):
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(n))
    return sum(v) / len(v), b[int(0.025 * n)], b[min(n - 1, int(0.975 * n))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--desde", type=int, required=True)
    ap.add_argument("--margen", type=float, default=2.0)
    ap.add_argument("--corpus", default=f"{AQUI.parent / 'corpus_mejora.json'},{AQUI / 'corpus_duraciones.json'}")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    import torch
    from faster_whisper import WhisperModel
    corpus = {}
    for c in a.corpus.split(","):
        corpus.update(json.loads(Path(c).read_text()))
    carpeta = Path(a.carpeta)
    t0 = a.desde * HOP / HZ
    v0, v1 = t0 - a.margen, t0 + a.margen
    whisper = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=a.hilos)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    med = {}
    for w in sorted(carpeta.glob("*.wav")):
        p = partes(w.stem)
        if not p or p["lam"] is None:
            continue
        x, hz = sf.read(str(w), dtype="float32")
        texto = corpus[p["grupo"]][p["k"]]
        segs, _ = whisper.transcribe(w.as_posix(), language=IDIOMA.get(p["grupo"], "es"), beam_size=5, word_timestamps=True,
                                     condition_on_previous_text=False, temperature=0.0)
        oidas = [(pal, (wd.start + wd.end) / 2) for s in segs for wd in s.words for pal in palabras(wd.word)]
        ref = palabras(texto)
        sal, ins = alinear(ref, [o[0] for o in oidas])
        err = tot = 0
        for i, (jh, e) in enumerate(sal):
            t = oidas[jh][1] if jh is not None and oidas else -1
            if v0 <= t <= v1:
                tot += 1
                err += int(e) + ins.get(i, 0)
        a0, a1 = max(0, int(v0 * hz)), min(len(x), int(v1 * hz))
        u = None
        if a1 - a0 > hz // 2:
            import librosa
            seg16 = librosa.resample(x[a0:a1], orig_sr=hz, target_sr=16000)
            with torch.inference_mode():
                u = round(float(utmos(torch.from_numpy(seg16)[None], 16000).item()), 3)
        med[w.stem] = dict(p, palabras_ventana=tot, errores_ventana=err, wer_ventana=(err / tot if tot else None),
                           utmos_ventana=u, dur=round(len(x) / hz, 2))
        print(w.stem, med[w.stem]["wer_ventana"], u, flush=True)
    (carpeta / "costura.json").write_text(json.dumps(med, indent=1))
    por = defaultdict(dict)
    for n, m in med.items():
        por[(m["voz"], m["grupo"], m["k"], m["s"])][m["lam"]] = m
    lams = sorted({m["lam"] for m in med.values()} - {0.0})
    print(f"\nventana {v0:.2f}-{v1:.2f} s alrededor del fotograma {a.desde}; pareado contra lambda 0")
    resumen = {}
    for lam in lams:
        dw, du, rotos = [], [], 0
        for v in por.values():
            b, x = v.get(0.0), v.get(lam)
            if not b or not x:
                continue
            if b["wer_ventana"] is not None and x["wer_ventana"] is not None:
                dw.append(100 * (x["wer_ventana"] - b["wer_ventana"]))
                rotos += int(b["wer_ventana"] <= 0.10 and x["wer_ventana"] > 0.25)
            if b["utmos_ventana"] is not None and x["utmos_ventana"] is not None:
                du.append(x["utmos_ventana"] - b["utmos_ventana"])
        if not dw:
            continue
        mw, lw, hw = ic(dw)
        mu, lu, hu = ic(du) if du else (float("nan"),) * 3
        resumen[str(lam)] = dict(n=len(dw), wer_pts=[round(mw, 2), round(lw, 2), round(hw, 2)], utmos=[round(mu, 3), round(lu, 3), round(hu, 3)],
                                 rotos=rotos)
        print(f"lambda {lam:g}: n={len(dw)} WER ventana {mw:+.2f} pts [{lw:+.2f},{hw:+.2f}]; UTMOS ventana {mu:+.3f} [{lu:+.3f},{hu:+.3f}]; "
              f"rotos {rotos}")
    (carpeta / "costura_resumen.json").write_text(json.dumps(dict(desde=a.desde, ventana_s=[v0, v1], por_lambda=resumen), indent=1))


if __name__ == "__main__":
    main()
