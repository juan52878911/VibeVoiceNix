#!/usr/bin/env python3
"""§5.5 · La sorpresa del propio modelo frente al WER de whisper, por clip.

    python3 scripts/red/sorpresa_wer.py --modelo <ruta> --inst red/inst --juez red/inst/juez.json --salida red/sorpresa.json

Para cada .npz de instrumentar.py (cond y lat por fotograma) calcula sorpresa.por_fotograma() -- la perdida v de la
cabeza de difusion sobre el latente que ella misma eligio, con ruido de semilla fija -- y la resume por clip (media,
p90, maximo y la racha maxima de fotogramas por encima del p90 global). Lo cruza con el WER (wer_norm si esta) de
juez_lote.py sobre los mismos clips: correlacion de Spearman con IC 95 % por remuestreo, global y por voz, y el
area bajo la curva de la sorpresa media para separar clips con WER > 0,10 del resto.

La sorpresa solo necesita la cabeza (84 MB) y la condicion guardada: no vuelve a generar. En CPU, 8 t por fotograma.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import modelo as MO  # noqa: E402
import sorpresa as SO  # noqa: E402


def rangos(v):
    v = np.asarray(v, float)
    orden = v.argsort(kind="mergesort")
    r = np.empty(len(v))
    r[orden] = np.arange(len(v))
    for x in np.unique(v):                       # empates: rango medio
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rangos(a), rangos(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_spearman(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    r = random.Random(0)
    vals = []
    for _ in range(n):
        idx = [r.randrange(len(a)) for _ in range(len(a))]
        s = spearman(a[idx], b[idx])
        if np.isfinite(s):
            vals.append(s)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--inst", required=True)
    ap.add_argument("--juez", required=True, help="medidas.json de juez_lote.py sobre los mismos clips")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--n-t", type=int, default=8)
    a = ap.parse_args()
    juez = json.loads(Path(a.juez).read_text())
    sal = Path(a.salida)
    cache = json.loads(sal.read_text())["clips"] if sal.exists() else {}
    m = None
    for f in sorted(Path(a.inst).glob("*.npz")):
        if f.stem in cache:
            continue
        if m is None:
            m, _ = MO.cargar(a.modelo)
        d = np.load(f, allow_pickle=True)
        reg = [dict(cond=torch.from_numpy(c).float(), lat=torch.from_numpy(z).float()) for c, z in zip(d["cond"], d["lat"])]
        s = SO.por_fotograma(m, reg, n_t=a.n_t, semilla=0).numpy()
        cache[f.stem] = dict(voz=str(d["voz"]), fotogramas=len(s), serie=[round(float(x), 4) for x in s])
        print(f"{f.stem}: {len(s)} fotogramas, sorpresa media {s.mean():.3f}", flush=True)
    todas = np.concatenate([np.array(c["serie"]) for c in cache.values()])
    p90 = float(np.percentile(todas, 90))
    filas = []
    for nombre, c in cache.items():
        s = np.array(c["serie"])
        racha = mejor = 0
        for x in s:
            racha = racha + 1 if x > p90 else 0
            mejor = max(mejor, racha)
        j = juez.get(nombre, {})
        w = j.get("wer_norm", j.get("wer"))
        c.update(media=round(float(s.mean()), 4), p90=round(float(np.percentile(s, 90)), 4), maximo=round(float(s.max()), 4),
                 racha_p90=mejor, wer=w, utmos=j.get("utmos"))
        if w is not None:
            filas.append((nombre, c["voz"], c["media"], c["p90"], c["racha_p90"], w, j.get("utmos")))
    res = dict(p90_global=round(p90, 4), n=len(filas), por_resumen={}, por_voz={})
    print(f"\n{len(filas)} clips con WER; p90 global de la sorpresa {p90:.3f}")
    for i, nombre_r in ((2, "media"), (3, "p90"), (4, "racha_p90")):
        x, y = [f[i] for f in filas], [f[5] for f in filas]
        rho = spearman(x, y)
        lo, hi = ic_spearman(x, y)
        malos = [f[i] for f in filas if f[5] > 0.10]
        buenos = [f[i] for f in filas if f[5] <= 0.10]
        res["por_resumen"][nombre_r] = dict(spearman=round(rho, 3), ic=[round(lo, 3), round(hi, 3)],
                                            auc_wer_mayor_010=round(auc(malos, buenos), 3), n_malos=len(malos))
        print(f"sorpresa {nombre_r:9s} vs WER: Spearman {rho:+.3f} [{lo:+.3f},{hi:+.3f}]; AUC (WER > 0,10: {len(malos)} clips) "
              f"{auc(malos, buenos):.3f}")
    uts = [(f[2], f[6]) for f in filas if f[6] is not None]
    if uts:
        rho = spearman([u[0] for u in uts], [u[1] for u in uts])
        lo, hi = ic_spearman([u[0] for u in uts], [u[1] for u in uts])
        res["media_vs_utmos"] = dict(spearman=round(rho, 3), ic=[round(lo, 3), round(hi, 3)])
        print(f"sorpresa media vs UTMOS: Spearman {rho:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for voz in sorted({f[1] for f in filas}):
        fs = [f for f in filas if f[1] == voz]
        rho = spearman([f[2] for f in fs], [f[5] for f in fs])
        res["por_voz"][voz] = dict(n=len(fs), spearman_media_wer=round(rho, 3), sorpresa_media=round(float(np.mean([f[2] for f in fs])), 4),
                                   wer_medio=round(float(np.mean([f[5] for f in fs])), 4))
        print(f"  {voz:14s} n={len(fs):3d} sorpresa media {res['por_voz'][voz]['sorpresa_media']:.3f} WER medio "
              f"{res['por_voz'][voz]['wer_medio']:.3f} Spearman {rho:+.3f}")
    peores = sorted(filas, key=lambda f: -f[5])[:8]
    res["peores_wer"] = [dict(clip=f[0], wer=f[5], sorpresa_media=f[2], racha=f[4]) for f in peores]
    sal.write_text(json.dumps(dict(resumen=res, clips=cache), indent=1))


if __name__ == "__main__":
    main()
