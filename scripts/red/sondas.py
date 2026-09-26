#!/usr/bin/env python3
"""I1 · Sondas lineales por capa: donde es legible cada atributo prosodico, y la direccion de cada uno.

    python3 scripts/red/sondas.py --inst red/inst --salida red/sondas

Etiquetas por fotograma (133 ms), calculadas sobre el .wav de instrumentar.py, sin datos externos:
  f0_st        tono en semitonos respecto a la mediana del clip (pyin; NaN sin voz)
  energia_db   energia del fotograma respecto a la media del clip
  sonoro       fraccion de ventanas con voz
  pausa        el fotograma esta callado (energia < -35 dB del pico)
  hasta_pausa  fotogramas hasta la siguiente pausa (tope 12)
  hasta_fin    fotogramas hasta el final
  pregunta     el texto acaba en '?'
  ventana      indice de la ventana de texto leida (control: se lee trivialmente)

CONTROL DE POSICION (medido con pesos aleatorios): la posicion del fotograma se lee casi perfecta en cualquier capa
(hasta_fin y ventana dan R^2 > 0,98 aunque el audio sea ruido), asi que cualquier etiqueta que derive con el tiempo
dentro del clip daria un R^2 falso. Por defecto a cada etiqueta se le resta, dentro del clip, lo que explica un
polinomio cubico de la posicion mas la fase dentro de la ventana de 6 fotogramas (el bucle lee 5 fichas y genera
6 latentes: hay un ritmo de periodo 6 en todo), y se informa el R^2 sobre ESE residuo (--con-posicion lo desactiva).
La fila "posicion" es una sonda que solo ve esas variables de posicion: lo que un sitio no supere sobre esa fila no
es prosodia, es reloj. hasta_fin y ventana quedan como control.

OJO (medido en seco, 26-09): la red codifica la posicion EXACTA (RoPE), asi que la correccion no basta para etiquetas
que sean funcion de la posicion; en el ensayo con pesos aleatorios y clips todos de 24 fotogramas, hasta_fin siguio
en 0,98. Con pesos reales hay que usar clips de duraciones distintas y fiarse solo de las etiquetas que no dependen
del reloj (f0_st, energia_db, pausa, pregunta), mirando siempre la fila "posicion".

Por sitio (condicion final, negativa, residual de cada capa) y etiqueta: ridge con la media por clip restada
(intra-clip, para que la direccion no lleve la voz), validacion dejando un CLIP fuera, R^2 (o exactitud para
las binarias). Guarda R^2 por capa y la direccion unitaria de cada (sitio, etiqueta) en <salida>/direcciones.npz,
lista para dirigir.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HOP = 3200
HZ = 24000


def etiquetas(onda, T, texto):
    import librosa
    x = onda[: T * HOP]
    if len(x) < T * HOP:
        x = np.pad(x, (0, T * HOP - len(x)))
    f0, sonoro, _ = librosa.pyin(x, fmin=60, fmax=450, sr=HZ, frame_length=1024, hop_length=320)
    n = len(f0)
    por = max(1, n // T)
    f0f, sonf, ene = [], [], []
    for j in range(T):
        seg_f0 = f0[j * por:(j + 1) * por]
        seg_v = sonoro[j * por:(j + 1) * por]
        f0f.append(np.nanmedian(seg_f0) if np.isfinite(seg_f0).any() else np.nan)
        sonf.append(float(np.mean(seg_v)) if len(seg_v) else 0.0)
        s = x[j * HOP:(j + 1) * HOP]
        ene.append(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9))
    f0f, ene = np.array(f0f), np.array(ene)
    st = 12 * np.log2(f0f / np.nanmedian(f0f)) if np.isfinite(f0f).any() else np.full(T, np.nan)
    pausa = (ene < ene.max() - 35).astype(float)
    hasta_pausa = np.full(T, 12.0)
    prox = None
    for j in range(T - 1, -1, -1):
        if pausa[j]:
            prox = j
        elif prox is not None:
            hasta_pausa[j] = min(12, prox - j)
    return dict(f0_st=st, energia_db=ene - ene.mean(), sonoro=np.array(sonf), pausa=pausa, hasta_pausa=hasta_pausa,
                hasta_fin=np.minimum(12, np.arange(T)[::-1]).astype(float), pregunta=np.full(T, float(texto.strip().endswith("?"))))


def posicion(T):
    """Variables de reloj de un clip: cubica en la posicion relativa y fase (0-5) dentro de la ventana de voz."""
    j = np.arange(T)
    rel = j / max(1, T - 1)
    fase = np.eye(6)[j % 6]
    return np.concatenate([np.stack([np.ones(T), rel, rel ** 2, rel ** 3], 1), fase], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--alfa", type=float, default=10.0)
    ap.add_argument("--con-posicion", action="store_true", help="no quitar la tendencia con la posicion")
    a = ap.parse_args()
    import soundfile as sf
    from sklearn.linear_model import Ridge, RidgeClassifier
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    clips = []
    for f in sorted(Path(a.inst).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        onda, _ = sf.read(str(f.with_suffix(".wav")), dtype="float32")
        T = d["cond"].shape[0]
        et = etiquetas(onda, T, str(d["texto"]))
        et["ventana"] = d["ventana"].astype(float)
        sitios = {"condicion": d["cond"], "negativa": d["neg"]}
        if d["res"].size:
            for c in range(d["res"].shape[1]):
                sitios[f"capa{c:02d}"] = d["res"][:, c].astype(np.float32)
        clips.append(dict(nombre=f.stem, sitios=sitios, et=et))
    print(f"{len(clips)} clips, {sum(c['et']['pausa'].shape[0] for c in clips)} fotogramas")
    nombres_sitios = ["posicion"] + list(clips[0]["sitios"])
    for c in clips:
        c["sitios"]["posicion"] = None
    etiq = list(clips[0]["et"])
    binarias = {"pausa", "pregunta"}
    res, direcciones = {}, {}
    for sitio in nombres_sitios:
        res[sitio] = {}
        for e in etiq:
            X, y, g = [], [], []
            for i, c in enumerate(clips):
                xs, ys = c["sitios"][sitio], c["et"][e]
                ok = np.isfinite(ys)
                if ok.sum() < 4:
                    continue
                P = posicion(len(ys))[ok]
                xs = P if xs is None else xs[ok] - xs[ok].mean(0)   # intra-clip: fuera la voz y el texto medio
                yy = ys[ok] if e in binarias else ys[ok] - ys[ok].mean()
                if e not in binarias and not a.con_posicion:
                    yy = yy - P @ np.linalg.lstsq(P, yy, rcond=None)[0]
                X.append(xs); y.append(yy); g.append(np.full(ok.sum(), i))
            if not X:
                continue
            X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
            if e in binarias and len(np.unique(y)) < 2:
                continue
            puntos = []
            for k in np.unique(g):
                tr, te = g != k, g == k
                if e in binarias:
                    if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                        continue
                    clf = RidgeClassifier(alpha=a.alfa).fit(X[tr], y[tr])
                    puntos.append((clf.predict(X[te]) == y[te]).mean())
                else:
                    rg = Ridge(alpha=a.alfa).fit(X[tr], y[tr])
                    p = rg.predict(X[te])
                    ss = ((y[te] - y[te].mean()) ** 2).sum()
                    puntos.append(1 - ((y[te] - p) ** 2).sum() / ss if ss > 0 else 0.0)
            if sitio != "posicion":
                modelo = (RidgeClassifier if e in binarias else Ridge)(alpha=a.alfa).fit(X, y)
                w = np.ravel(modelo.coef_)
                direcciones[f"{sitio}/{e}"] = w / (np.linalg.norm(w) + 1e-9)
            res[sitio][e] = round(float(np.mean(puntos)), 3) if puntos else None
    json.dump(res, open(sal / "r2_por_sitio.json", "w"), indent=1)
    np.savez(sal / "direcciones.npz", **direcciones)
    print("R2 (exactitud en pausa/pregunta), deja-un-clip-fuera:")
    print("sitio      " + " ".join(f"{e[:10]:>10}" for e in etiq))
    for s in nombres_sitios:
        print(f"{s:10} " + " ".join(f"{res[s].get(e, float('nan')) if res[s].get(e) is not None else float('nan'):10.3f}" for e in etiq))


if __name__ == "__main__":
    main()
