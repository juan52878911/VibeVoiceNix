#!/usr/bin/env python
"""Fase 4 del plan de personalidad: pausas y ritmo por voz, sin tocar el modelo.

    python scripts/fase4_pausas.py --dataset /var/lib/taller/dataset-voces --fase2 /var/lib/taller/fase2b \
        --voces /var/lib/taller/fase1/voces --salida /var/lib/taller/fase4

LO QUE SE SABE (docs/plan-personalidad-voz.md, fase 4):
  - El clon de Carlos hace 10 pausas por minuto y Carlos 21, y va más lento (4,6 frente a 5,2 sílabas/s).
  - Carlos pausa, de media, una vez por signo de puntuación (7,2 palabras por pausa y 7,2 por signo),
    con pausas de mediana 0,52 s. Liliana, en unos dos de cada tres signos y más cortas (0,35 s).
  - El modelo pausa en comas y puntos, pero se salta muchas, y nunca pausa dentro de una frase
    (voz_stream.py, EL RESPIRO). Solo "\n" fuerza su parada de fin de locución.

LAS VARIANTES, mismo texto y misma semilla por voz-stream (el motor de producción):
  base      el texto tal cual
  trozos    partido en los signos donde toca pausar, una petición por trozo, unidos con una pausa
  saltos    una sola petición con "\n" en esos signos
  *_r       lo mismo con la velocidad de la persona (WSOLA de estirar.py)
En todas menos la base, cada pausa (racha callada >= 150 ms, la definición de perfil_vocal.py) pasa a
durar lo que sale de la distribución de pausas reales de la persona, sorteada con semilla por clip.

DÓNDE PAUSAR: en un signo, si desde la última pausa van al menos K palabras. K se calibra con los clips de
ENTRENAMIENTO para igualar sus palabras por pausa reales. Los trozos de menos de 4 palabras se unen al
siguiente: un trozo muy corto es donde el modelo se inventa cosas. La velocidad se calibra con 12 textos
de entrenamiento (factor 0,85-1,20). Los clips apartados de la fase 2b no calibran nada.

Deja <salida>/<variante>/ con los WAV, clips.csv y frases.json (el texto SIN los "\n": es lo que se pidió
decir) y <salida>/medidas_vm.json con pausas y velocidad de cada clip y de los clips reales apartados.
WER, cobertura, UTMOS e identidad se miden en el Mac (fase4_puerta.py). Cada paso se salta lo que ya existe.
"""
import argparse
import csv
import json
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent / "pkgs" / "vibevoice-cli"))  # estirar.py (en la VM va junto a los scripts)
import fase2_estilo as F2  # noqa: E402
from fase2_texto_nuevo import FRASES_NUEVAS, pedir_wav  # noqa: E402

SR = 24000
HOP = 240
MIN_RACHA = 15          # 150 ms, como perfil_vocal.py
MIN_TROZO = 4
SEMILLAS = (101, 7)
SIGNOS = ",.;:?!"
VOCALES = re.compile(r"[aeiouáéíóúü]+", re.I)
CORTE = re.compile(r"(?<=[,.;:?!])\s+")
VARIANTES = ("base", "trozos", "saltos", "trozos_r", "saltos_r")


# ------------------------------------------------------------------ medida
def actividad(x):
    import librosa
    rms = librosa.feature.rms(y=x, frame_length=2 * HOP, hop_length=HOP)[0]
    db = 20 * np.log10(rms / (rms.max() + 1e-9) + 1e-9)
    return db > -35


def rachas(x):
    """Rachas calladas INTERIORES de >= 150 ms, en muestras [ini, fin)."""
    act = actividad(x)
    if not act.any():
        return []
    i0, i1 = int(np.argmax(act)), len(act) - int(np.argmax(act[::-1]))
    out, ini = [], None
    for i in range(i0, i1):
        if not act[i]:
            ini = i if ini is None else ini
        else:
            if ini is not None and i - ini >= MIN_RACHA:
                out.append((ini * HOP, min(i * HOP, len(x))))
            ini = None
    return out


def palabras(t):
    return len(t.split())


def medir(x, texto):
    dur = len(x) / SR
    r = rachas(x)
    return {"dur_s": dur, "pausas": len(r), "pausas_min": len(r) / (dur / 60),
            "silabas_s": len(VOCALES.findall(texto)) / dur,
            "mediana_pausa": float(np.median([(b - a) / SR for a, b in r])) if r else None}


# ------------------------------------------------------------------ texto
def trozos(texto, k):
    piezas = [p for p in CORTE.split(texto.strip()) if p]
    out, cur = [], []
    for p in piezas:
        cur.append(p)
        if palabras(" ".join(cur)) >= k and p[-1] in SIGNOS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    unidos = []
    for t in out:
        if unidos and palabras(unidos[-1]) < MIN_TROZO:
            unidos[-1] = f"{unidos[-1]} {t}"
        else:
            unidos.append(t)
    if len(unidos) > 1 and palabras(unidos[-1]) < MIN_TROZO:
        ultimo = unidos.pop()
        unidos[-1] = f"{unidos[-1]} {ultimo}"
    return unidos


def calibrar_k(textos, ppp_real):
    mejor = None
    for k in range(1, 25):
        cortes = sum(len(trozos(t, k)) - 1 for t in textos)
        ppp = sum(palabras(t) for t in textos) / max(cortes, 1)
        if mejor is None or abs(ppp - ppp_real) < abs(mejor[1] - ppp_real):
            mejor = (k, ppp)
    return mejor


# ------------------------------------------------------------------ audio
def rellenar(s, m):
    """m muestras de silencio hechas con la propia racha: los bordes intactos y el centro en espejo."""
    borde = min(len(s) // 2, int(0.05 * SR))
    if m <= 2 * borde:
        return np.concatenate([s[:m // 2], s[len(s) - (m - m // 2):]])
    centro = s[borde:len(s) - borde]
    if len(centro) < 32:
        centro = np.zeros(32, np.float32)
    falta, piezas, n, adelante = m - 2 * borde, [], 0, True
    while n < falta:
        piezas.append(centro if adelante else centro[::-1])
        n += len(centro)
        adelante = not adelante
    return np.concatenate([s[:borde], np.concatenate(piezas)[:falta], s[len(s) - borde:]])


def forma_pausas(x, dist, semilla):
    rng = np.random.default_rng(semilla)
    partes, previo = [], 0
    for a, b in rachas(x):
        partes += [x[previo:a], rellenar(x[a:b], int(rng.choice(dist) * SR))]
        previo = b
    partes.append(x[previo:])
    return np.concatenate(partes).astype(np.float32)


def recortar_bordes(y):
    act = actividad(y)
    if not act.any():
        return y
    margen = int(0.04 * SR)
    i0 = max(0, int(np.argmax(act)) * HOP - margen)
    i1 = min(len(y), (len(act) - int(np.argmax(act[::-1]))) * HOP + margen)
    return y[i0:i1]


def unir(ondas, semilla):
    """Trozos sin sus bordes callados, separados por un hueco de suelo de sala que forma_pausas ajusta luego."""
    rng = np.random.default_rng(semilla)
    out = []
    for i, y in enumerate(ondas):
        if i:
            out.append((rng.standard_normal(int(0.3 * SR)) * 7e-4).astype(np.float32))
        out.append(recortar_bordes(y))
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fase2", required=True)
    ap.add_argument("--voces", required=True, help="los .pt de clonar_voz.py de la fase 1")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--personas", nargs="+", default=["carlos-segura", "liliana-morales"])
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--token-env", default="/var/lib/voz/token.env")
    ap.add_argument("--voces-servicio", default="/run/voz-stream/voces")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--n-calibracion", type=int, default=12)
    a = ap.parse_args()
    import soundfile as sf
    from estirar import estirar
    sal, ds = Path(a.salida), Path(a.dataset)

    def leer(ruta):
        x, _ = sf.read(str(ruta), dtype="float32")
        return x

    def escribir(ruta, x):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ruta), np.clip(x, -1, 1), SR, subtype="PCM_16")

    filas = list(csv.DictReader(open(ds / "manifiesto.csv", encoding="utf-8")))
    inf2 = json.load(open(Path(a.fase2) / "informe_fase2.json"))
    d2 = np.load(Path(a.fase2) / "caracteristicas.npz", allow_pickle=True)
    _, va = F2.particion(d2["C"], d2["P"], d2["R"])
    apartados = {(inf2["personas"][int(d2["P"][i])], str(d2["C"][i])) for i in np.where(va)[0]}

    # ------------------------------------------------ calibracion con los clips de ENTRENAMIENTO
    calib, reales_val, trabajo = {}, {}, []
    for p in a.personas:
        ent = [f for f in filas if f["hablante"] == p and (p, Path(f["fichero"]).stem) not in apartados]
        n_pal = n_pau = 0
        dur = voc = 0.0
        dist = []
        for f in ent:
            x = leer(ds / f["fichero"])
            r = rachas(x)
            n_pal += palabras(f["texto"])
            n_pau += len(r)
            dist += [(b - c) / SR for c, b in r]
            dur += len(x) / SR
            voc += len(VOCALES.findall(f["texto"]))
        ppp_real = n_pal / max(n_pau, 1)
        k, ppp_regla = calibrar_k([f["texto"] for f in ent], ppp_real)
        largos = [f for f in ent if palabras(f["texto"]) >= 8]
        rng = np.random.default_rng(zlib.crc32(p.encode()))
        cal = [largos[i] for i in rng.choice(len(largos), min(a.n_calibracion, len(largos)), replace=False)]
        calib[p] = {"k": k, "palabras_por_pausa_real": ppp_real, "palabras_por_pausa_regla": ppp_regla,
                    "pausas_min_real": n_pau / (dur / 60), "silabas_s_real": voc / dur,
                    "pausa_real_p25_p50_p75": [float(np.percentile(dist, q)) for q in (25, 50, 75)],
                    "dist": [float(v) for v in np.clip(dist, 0.15, 1.2)],
                    "calibracion": [Path(f["fichero"]).stem for f in cal], "clips_entreno": len(ent)}
        print(f"{p}: K={k} (palabras por pausa real {ppp_real:.1f}, regla {ppp_regla:.1f}) · "
              f"{calib[p]['pausas_min_real']:.1f} pausas/min · {calib[p]['silabas_s_real']:.2f} silabas/s · "
              f"{len(ent)} clips de entreno", flush=True)
        reales_val[p] = {}
        for f in filas:
            clip = Path(f["fichero"]).stem
            if f["hablante"] == p and (p, clip) in apartados:
                reales_val[p][clip] = medir(leer(ds / f["fichero"]), f["texto"])
                trabajo.append((p, clip, f["texto"].strip(), "apartado", SEMILLAS))
        for i, t in enumerate(FRASES_NUEVAS):
            trabajo.append((p, f"nuevo-{i}", t, "nuevo", SEMILLAS))
        for f in cal:
            trabajo.append((p, f"cal-{Path(f['fichero']).stem}", f["texto"].strip(), "calibracion", (101,)))

    # ------------------------------------------------ generar por voz-stream
    token = next(linea.split("=", 1)[1].strip() for linea in open(a.token_env) if linea.startswith("VOZ_TOKEN="))
    copiadas, t0, n = [], time.time(), 0

    def pedir(ruta, texto, persona, semilla):
        nonlocal n
        if ruta.exists():
            return
        y = pedir_wav(a.url, token, texto, f"taller-{persona}", a.cfg, semilla, a.pasos)
        escribir(ruta, y)
        n += 1
        if n % 20 == 0:
            print(f"  {n} peticiones · {(time.time() - t0) / 60:.1f} min", flush=True)

    try:
        for p in a.personas:
            destino = Path(a.voces_servicio) / f"taller-{p}.pt"
            shutil.copyfile(Path(a.voces) / f"{p}.pt", destino)
            shutil.chown(destino, "voz-stream", "voz-stream")
            copiadas.append(destino)
        print(f"== generando {len(trabajo)} textos", flush=True)
        for p, clave, texto, tipo, semillas in trabajo:
            partes = trozos(texto, calib[p]["k"])
            for s in semillas:
                nombre = f"{p}__{clave}__s{s}"
                if tipo != "calibracion":
                    pedir(sal / "_crudo" / "base" / f"{nombre}.wav", texto, p, s)
                for i, t in enumerate(partes):
                    pedir(sal / "_crudo" / "trozos" / f"{nombre}__t{i}.wav", t, p, s)
                pedir(sal / "_crudo" / "saltos" / f"{nombre}.wav", "\n".join(partes), p, s)
    finally:
        for c in copiadas:
            c.unlink(missing_ok=True)
    print(f"== generado en {(time.time() - t0) / 60:.1f} min ({n} peticiones nuevas)", flush=True)

    # ------------------------------------------------ variantes
    texto_de = {(p, clave): texto for p, clave, texto, _, _ in trabajo}

    def onda_cruda(p, clave, s, variante):
        nombre = f"{p}__{clave}__s{s}"
        if variante == "base":
            return leer(sal / "_crudo" / "base" / f"{nombre}.wav")
        if variante == "saltos":
            return leer(sal / "_crudo" / "saltos" / f"{nombre}.wav")
        partes = trozos(texto_de[(p, clave)], calib[p]["k"])
        return unir([leer(sal / "_crudo" / "trozos" / f"{nombre}__t{i}.wav") for i in range(len(partes))],
                    zlib.crc32(nombre.encode()))

    velocidad = {}
    for p in a.personas:
        velocidad[p] = {}
        for var in ("trozos", "saltos"):
            voc_r = dur_r = voc_v = dur_v = 0.0
            for p2, clave, texto, tipo, _ in trabajo:
                if p2 != p or tipo != "calibracion":
                    continue
                nombre = f"{p}__{clave}__s101"
                y = forma_pausas(onda_cruda(p, clave, 101, var), calib[p]["dist"], zlib.crc32(nombre.encode()))
                real = leer(ds / p / f"{clave[4:]}.wav")
                voc = len(VOCALES.findall(texto))
                voc_r, dur_r = voc_r + voc, dur_r + len(real) / SR
                voc_v, dur_v = voc_v + voc, dur_v + len(y) / SR
            velocidad[p][var] = float(np.clip((voc_r / dur_r) / (voc_v / dur_v), 0.85, 1.20))
        print(f"{p}: velocidad calibrada {velocidad[p]}", flush=True)

    medidas, frases = {}, {}
    for p, clave, texto, tipo, semillas in trabajo:
        if tipo == "calibracion":
            continue
        frases[f"{p}__{clave}"] = texto
        for s in semillas:
            nombre = f"{p}__{clave}__s{s}"
            sem = zlib.crc32(nombre.encode())
            fila = {"voz": p, "frase": f"{p}__{clave}", "tipo": tipo, "clip_real": clave if tipo == "apartado" else None,
                    "variantes": {}}
            for var in VARIANTES:
                ruta = sal / var / f"{nombre}.wav"
                if not ruta.exists():
                    base_var = var.removesuffix("_r")
                    y = onda_cruda(p, clave, s, base_var)
                    if var.endswith("_r"):
                        y = estirar(y, velocidad[p][base_var])
                    if var != "base":
                        y = forma_pausas(y, calib[p]["dist"], sem)
                    escribir(ruta, y)
                fila["variantes"][var] = medir(leer(ruta), texto)
            medidas[f"{nombre}.wav"] = fila
    for var in VARIANTES:
        with open(sal / var / "clips.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["fichero", "frase", "voz", "semilla"])
            w.writeheader()
            for fichero, fila in medidas.items():
                w.writerow({"fichero": fichero, "frase": fila["frase"], "voz": fila["voz"],
                            "semilla": fichero.rsplit("__s", 1)[1].removesuffix(".wav")})
        json.dump(frases, open(sal / var / "frases.json", "w"), ensure_ascii=False, indent=1)
    for c in calib.values():
        c.pop("dist")
    json.dump({"calibracion": calib, "velocidad": velocidad, "reales_val": reales_val, "clips": medidas,
               "personas": a.personas, "argumentos": vars(a)},
              open(sal / "medidas_vm.json", "w"), ensure_ascii=False, indent=1)

    print("\n== pausas/min y silabas/s de media (clips apartados; real entre corchetes)")
    for p in a.personas:
        ap_clips = [f for f in medidas.values() if f["voz"] == p and f["tipo"] == "apartado"]
        rp = np.mean([reales_val[p][f["clip_real"]]["pausas_min"] for f in ap_clips])
        rs = np.mean([reales_val[p][f["clip_real"]]["silabas_s"] for f in ap_clips])
        print(f"   {p} [real {rp:.1f} pausas/min · {rs:.2f} silabas/s]")
        for var in VARIANTES:
            print(f"     {var:9s} {np.mean([f['variantes'][var]['pausas_min'] for f in ap_clips]):5.1f} pausas/min · "
                  f"{np.mean([f['variantes'][var]['silabas_s'] for f in ap_clips]):.2f} silabas/s")
    print(f"listo en {(time.time() - t0) / 60:.1f} min · {sal / 'medidas_vm.json'}")


if __name__ == "__main__":
    main()
