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
(intra-clip, para que la direccion no lleve la voz), R^2 por clip apartado promediado (o exactitud para las binarias).
Guarda R^2 por capa y la direccion unitaria de cada (sitio, etiqueta) en <salida>/direcciones.npz, lista para
dirigir.py.

PLIEGUES (--pliegues, 27-09): 'texto' aparta TODOS los clips de una frase (todas las voces y semillas) y es el que
decide la puerta; 'voz' aparta una voz entera; 'clip' aparta un clip (lo del ensayo en seco). Con 'clip' la misma
frase dicha con otra semilla o por otra voz queda en el entrenamiento y su contorno es casi el mismo: la sonda puede
aprenderse la frase en vez de leer la prosodia. El ajuste es ridge en forma cerrada con las matrices de Gram
acumuladas por pliegue (igual que sklearn Ridge con intercepto; 160 clips x 22 sitios en minutos, no en horas).
ALFA RELATIVA (27-09): la varianza por dimension de la condicion es ~250 veces la de los residuales, asi que un alfa
fijo regulariza mucho un sitio y nada el otro (la condicion daba R^2 negativo por sobreajuste con alfa 10). Aqui
alfa = k * traza(G)/D con k elegido DENTRO de cada pliegue externo por 5 trozos internos; la direccion final usa
el k mas elegido en los pliegues por texto.
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
    ap.add_argument("--ks", default="0.01,0.1,1,10,100", help="alfa = k * traza(G)/D; k elegido por validacion interna")
    ap.add_argument("--con-posicion", action="store_true", help="no quitar la tendencia con la posicion")
    ap.add_argument("--pliegues", default="texto,voz", help="texto (decide la puerta), voz, clip (lento)")
    a = ap.parse_args()
    a.ks = [float(x) for x in a.ks.split(",")]
    import soundfile as sf
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
        clips.append(dict(nombre=f.stem, sitios=sitios, et=et, texto=str(d["texto"]), voz=str(d["voz"])))
    print(f"{len(clips)} clips, {sum(c['et']['pausa'].shape[0] for c in clips)} fotogramas")
    nombres_sitios = ["posicion"] + list(clips[0]["sitios"])
    for c in clips:
        c["sitios"]["posicion"] = None
    etiq = list(clips[0]["et"])
    binarias = {"pausa", "pregunta"}
    modos = a.pliegues.split(",")
    grupo_de = dict(clip=lambda c: c["nombre"], texto=lambda c: c["texto"], voz=lambda c: c["voz"])
    res = {m: {} for m in modos}
    direcciones, alfas = {}, {}
    for sitio in nombres_sitios:
        for m in modos:
            res[m][sitio] = {}
        for e in etiq:
            # por clip: X intra-clip, y (sin tendencia de posicion) o y en {-1, 1} para las binarias
            datos = []
            for c in clips:
                xs, ys = c["sitios"][sitio], c["et"][e]
                ok = np.isfinite(ys)
                if ok.sum() < 4:
                    continue
                P = posicion(len(ys))[ok]
                X = P if xs is None else (xs[ok] - xs[ok].mean(0)).astype(np.float64)
                if e in binarias:
                    yy = np.where(ys[ok] > 0.5, 1.0, -1.0)
                else:
                    yy = ys[ok] - ys[ok].mean()
                    if not a.con_posicion:
                        yy = yy - P @ np.linalg.lstsq(P, yy, rcond=None)[0]
                datos.append(dict(c=c, X=X, y=yy, G=X.T @ X, b=X.T @ yy, sx=X.sum(0), sy=yy.sum(), n=len(yy)))
            if not datos:
                continue
            if e in binarias and len(np.unique(np.concatenate([d["y"] for d in datos]))) < 2:
                continue
            D = datos[0]["X"].shape[1]
            CLAVES = ("G", "b", "sx", "sy", "n")

            def suma(ds):
                return {k: sum(d[k] for d in ds) for k in CLAVES}

            def ajustar(G, b, sx, sy, n, k):
                mx, my = sx / n, sy / n
                Gc = G - n * np.outer(mx, mx)
                bc = b - n * mx * my
                alfa = k * np.trace(Gc) / D                   # alfa relativa a la escala del sitio
                w = np.linalg.solve(Gc + alfa * np.eye(D), bc)
                return w, mx, my

            def puntuar(ds, w, mx, my):
                out = []
                for d in ds:
                    p_ = (d["X"] - mx) @ w + my
                    if e in binarias:
                        out.append(float((np.sign(p_) == d["y"]).mean()))
                    else:
                        ss = ((d["y"] - d["y"].mean()) ** 2).sum()
                        out.append(1 - ((d["y"] - p_) ** 2).sum() / ss if ss > 0 else 0.0)
                return out

            def elegir_k(grupos):
                """k por validacion INTERNA: 5 trozos de los grupos de entrenamiento (nunca ve el grupo apartado)."""
                claves = sorted(grupos)
                if len(claves) < 2 or len(a.ks) == 1:
                    return a.ks[0]
                trozos = [claves[i::min(5, len(claves))] for i in range(min(5, len(claves)))]
                tot_i = suma([d for g in claves for d in grupos[g]])
                mejor, mejor_p = a.ks[0], -np.inf
                for k in a.ks:
                    pts = []
                    for tr in trozos:
                        ds = [d for g in tr for d in grupos[g]]
                        resto = {c: tot_i[c] - v for c, v in suma(ds).items()}
                        if resto["n"] < 20:
                            continue
                        pts += puntuar(ds, *ajustar(**resto, k=k))
                    if pts and np.mean(pts) > mejor_p:
                        mejor, mejor_p = k, float(np.mean(pts))
                return mejor
            tot = suma(datos)
            ks_elegidas = []
            for m in modos:
                por_grupo = {}
                for d in datos:
                    por_grupo.setdefault(grupo_de[m](d["c"]), []).append(d)
                puntos = []
                for g, ds in por_grupo.items():
                    resto_g = {h: v for h, v in por_grupo.items() if h != g}
                    resto = {c: tot[c] - v for c, v in suma(ds).items()}
                    if resto["n"] < 20:
                        continue
                    k = elegir_k(resto_g)
                    if m == modos[0]:
                        ks_elegidas.append(k)
                    puntos += puntuar(ds, *ajustar(**resto, k=k))
                res[m][sitio][e] = round(float(np.mean(puntos)), 3) if puntos else None
            if sitio != "posicion":
                k = max(set(ks_elegidas), key=ks_elegidas.count) if ks_elegidas else a.ks[0]
                w, _, _ = ajustar(**tot, k=k)
                direcciones[f"{sitio}/{e}"] = (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)
                alfas[f"{sitio}/{e}"] = k
        print(f"  {sitio} listo", flush=True)
    json.dump(dict(pliegues=modos, n_clips=len(clips), ks=a.ks, k_por_direccion=alfas, r2=res), open(sal / "r2_por_sitio.json", "w"), indent=1)
    np.savez(sal / "direcciones.npz", **direcciones)
    for m in modos:
        print(f"\nR2 (exactitud en pausa/pregunta), pliegues por {m}:")
        print("sitio      " + " ".join(f"{e[:10]:>10}" for e in etiq))
        for s_ in nombres_sitios:
            fila = res[m][s_]
            print(f"{s_:10} " + " ".join(f"{fila[e] if fila.get(e) is not None else float('nan'):10.3f}" for e in etiq))

if __name__ == "__main__":
    main()
