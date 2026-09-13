#!/usr/bin/env python
"""Banco A/B exhaustivo de dos variantes del sintetizador: mismo texto, misma voz, misma semilla.

    pkgs/vibevoice/.venv/bin/python scripts/banco_ab.py --base dir/viejo --contra dir/nuevo dir/control_int4

Cada carpeta es la salida de un generador de corpus (un WAV por voz x frase x semilla,
con clips.csv y frases.json). Se empareja clip a clip con --base y se mide lo que un oido
podria notar, separado por familias, y siempre con la variante base al lado:

  FORMA DE ONDA  duracion exacta, SNR contra la base
  TIMBRE         distorsion mel-cepstral (MCD, dB) y distancia log-espectral (LSD, dB)
                 fotograma a fotograma; identidad ECAPA contra la huella media de la voz
                 (dejando fuera el propio clip) y coseno directo contra el clip base
  TONO           F0 por fotograma con las octavas corregidas (prosodia.py): acuerdo de
                 sonoridad, desviacion en cents donde las dos suenan, y los descriptores
                 de prosodia.descripcion (tono medio, recorrido, desviacion, movimiento)
  RITMO          comienzo y final del habla, pausas internas (numero y duracion),
                 velocidad (palabras por segundo) y desfase de cada palabra (marcas de whisper)
  INTELIGIBLE    WER con faster-whisper large-v3 contra el texto pedido (fidelidad.wer,
                 con su normalizacion) y si la transcripcion sale identica
  NATURALIDAD    UTMOS22 (naturalidad.py)

POR QUE FOTOGRAMA A FOTOGRAMA Y NO LA DISTANCIA DE MELODIA DE prosodia.comparar
Esa distancia NO esta validada (ver su docstring). Aqui no hace falta: con la misma semilla
y el mismo texto, los clips de las dos variantes duran lo mismo y cada fotograma de uno
corresponde al mismo instante del otro, asi que la diferencia se mide directamente.

LAS DIFERENCIAS SE DAN CON INTERVALO. Media de la diferencia emparejada y su IC 95 % por
bootstrap (2000 remuestreos), por voz y global. Un IC que contiene el 0 no es un cambio.

EL CONTROL. Pasa una variante con un cambio de timbre CONOCIDO (el decodificador en int4)
como --contra. Si el banco no la separa de la base, el banco no sirve para concluir nada.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fidelidad as FI  # noqa: E402
import naturalidad as NA  # noqa: E402
import prosodia as PR  # noqa: E402

CLONES = {"andres", "isis", "juan", "santiago"}
HOP = 240                     # 10 ms a 24 kHz
SILENCIO_DB = -35.0           # por debajo del pico del clip
PAUSA_MIN = 15                # fotogramas: 150 ms


# ------------------------------------------------------------------ utilidades
def ic95(dif, n=2000, semilla=0):
    d = np.asarray([v for v in dif if v is not None and not math.isnan(v)], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    medias = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def energia_db(x):
    n = len(x) // HOP
    marcos = x[: n * HOP].reshape(n, HOP)
    rms = np.sqrt((marcos ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)


def ritmo(x):
    e = energia_db(x)
    activo = e > SILENCIO_DB
    if not activo.any():
        return {"inicio_s": float("nan"), "fin_s": float("nan"), "pausas": 0, "pausa_s": 0.0}, e
    i0, i1 = int(np.argmax(activo)), len(activo) - int(np.argmax(activo[::-1]))
    pausas, largo, total = 0, 0, 0
    for a in activo[i0:i1]:
        if not a:
            largo += 1
        else:
            if largo >= PAUSA_MIN:
                pausas += 1
                total += largo
            largo = 0
    return {"inicio_s": i0 * HOP / PR.RITMO, "fin_s": i1 * HOP / PR.RITMO,
            "pausas": pausas, "pausa_s": total * HOP / PR.RITMO}, e


def alinear(xb, xc):
    """Desfase (muestras) que mejor casa las envolventes, dentro de +-0,5 s.

    Hace falta cuando la variante cambia la AMPLITUD: el recorte del silencio de entrada y
    la cola final se deciden sobre el audio, asi que un clip puede empezar un fotograma
    antes o despues (se vio con el decodificador en int4: 16 de 238 clips con otro largo y
    el resto desplazados). Sin alinear, lo de fotograma a fotograma compara instantes
    distintos. Devuelve los dos trozos solapados y el desfase."""
    eb, ec = energia_db(xb), energia_db(xc)
    eb, ec = np.maximum(eb, -60.0) + 60.0, np.maximum(ec, -60.0) + 60.0
    maximo = int(0.5 * PR.RITMO / HOP)
    mejor, desfase = -np.inf, 0
    for d in range(-maximo, maximo + 1):
        a = eb[max(0, d):len(eb) + min(0, d)]
        b = ec[max(0, -d):len(ec) + min(0, -d)]
        n = min(len(a), len(b))
        if n < 20:
            continue
        c = float(np.dot(a[:n] - a[:n].mean(), b[:n] - b[:n].mean()) /
                  (np.linalg.norm(a[:n] - a[:n].mean()) * np.linalg.norm(b[:n] - b[:n].mean()) + 1e-9))
        if c > mejor:
            mejor, desfase = c, d
    s = desfase * HOP
    xb2 = xb[max(0, s):]
    xc2 = xc[max(0, -s):]
    n = min(len(xb2), len(xc2))
    return xb2[:n], xc2[:n], s


def timbre(xb, xc, sr):
    import librosa
    n = min(len(xb), len(xc))
    xb, xc = xb[:n], xc[:n]
    kw = dict(sr=sr, n_fft=1024, hop_length=HOP, n_mels=80)
    mb = librosa.feature.mfcc(y=xb, n_mfcc=25, **kw)[1:]
    mc = librosa.feature.mfcc(y=xc, n_mfcc=25, **kw)[1:]
    sb = np.abs(librosa.stft(xb, n_fft=1024, hop_length=HOP)) + 1e-8
    sc = np.abs(librosa.stft(xc, n_fft=1024, hop_length=HOP)) + 1e-8
    eb = energia_db(np.pad(xb, (0, HOP)))
    m = min(mb.shape[1], sb.shape[1], len(eb))
    activo = eb[:m] > SILENCIO_DB
    if activo.sum() < 5:
        activo[:] = True
    # librosa da los cepstros de un mel en dB; la formula de MCD los espera en logaritmo
    # natural. Sin pasarlos, sale inflada 10/ln10 (x4,34): 5 dB para dos clips a 39 dB de SNR.
    d = ((mb[:, :m] - mc[:, :m]) * (np.log(10) / 10))[:, activo]
    mcd = float(np.mean((10 / np.log(10)) * np.sqrt(2 * (d ** 2).sum(0))))
    # suelo de 80 dB bajo el maximo del clip: sin el, los bins casi vacios (silencio, agudos
    # sin energia) dominan el promedio con diferencias de logaritmos de numeros diminutos
    suelo = max(sb.max(), sc.max()) * 1e-4
    lb = 20 * np.log10(np.maximum(sb[:, :m][:, activo], suelo))
    lc = 20 * np.log10(np.maximum(sc[:, :m][:, activo], suelo))
    lsd_f = np.sqrt(((lb - lc) ** 2).mean(0))
    ruido = xc - xb
    snr = 10 * np.log10((xb ** 2).sum() / max((ruido ** 2).sum(), 1e-20))
    return {"snr_db": float(snr), "mcd_db": mcd, "lsd_db": float(lsd_f.mean())}


def tono(xb, xc):
    fb = PR.corregir_octavas(PR.contorno(xb))
    fc = PR.corregir_octavas(PR.contorno(xc))
    n = min(len(fb), len(fc))
    fb, fc = fb[:n], fc[:n]
    vb, vc = ~np.isnan(fb), ~np.isnan(fc)
    ambos = vb & vc
    cents = 1200 * np.log2(fc[ambos] / fb[ambos]) if ambos.any() else np.array([])
    db, dc = PR.descripcion(xb), PR.descripcion(xc)
    return {
        "sonoridad_acuerdo": float((vb == vc).mean()) if n else float("nan"),
        "cents_mediana": float(np.median(np.abs(cents))) if len(cents) else float("nan"),
        "cents_p95": float(np.percentile(np.abs(cents), 95)) if len(cents) else float("nan"),
        "tono_st": 12 * math.log2(dc["hz"] / db["hz"]) if db["hz"] and dc["hz"] else float("nan"),
        "d_recorrido_st": dc["recorrido"] - db["recorrido"],
        "d_desviacion_st": dc["desviacion"] - db["desviacion"],
        "d_movimiento": dc["movimiento"] - db["movimiento"],
        "hz_base": db["hz"], "recorrido_base": db["recorrido"],
    }


# ------------------------------------------------------------------ jueces pesados
class Jueces:
    def __init__(self, whisper):
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier
        torch.set_num_threads(os.cpu_count() or 8)
        t = time.time()
        self.whisper = WhisperModel(whisper, device="cpu", compute_type="int8",
                                    cpu_threads=os.cpu_count() or 8)
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
            run_opts={"device": "cpu"})
        self.utmos = NA.cargar_juez()
        self.torch = torch
        print(f"jueces cargados en {time.time() - t:.0f} s (whisper {whisper})", flush=True)

    def transcribir(self, ruta):
        segs, _ = self.whisper.transcribe(str(ruta), language="es", beam_size=5,
                                          word_timestamps=True, vad_filter=False,
                                          condition_on_previous_text=False, temperature=0.0)
        texto, palabras = [], []
        for s in segs:
            texto.append(s.text)
            for w in (s.words or []):
                palabras.append((w.start, w.end, w.word))
        return " ".join(t.strip() for t in texto).strip(), palabras

    def huella(self, x, sr):
        import librosa
        x16 = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        with self.torch.no_grad():
            e = self.ecapa.encode_batch(self.torch.from_numpy(x16).float()[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------ banco
def leer_corpus(carpeta):
    filas = list(csv.DictReader(open(Path(carpeta) / "clips.csv", encoding="utf-8")))
    frases = json.load(open(Path(carpeta) / "frases.json", encoding="utf-8"))
    return {f["fichero"]: f for f in filas}, frases


def analizar_variante(nombre, carpeta, jueces, cache):
    """Lo que es de UN clip (whisper, huella, UTMOS, ritmo), con cache en disco."""
    ruta_cache = Path(cache) / f"{nombre}.json"
    datos = json.load(open(ruta_cache)) if ruta_cache.exists() else {}
    clips, frases = leer_corpus(carpeta)
    t0, nuevos = time.time(), 0
    for i, (fichero, meta) in enumerate(sorted(clips.items())):
        if fichero in datos:
            continue
        x, sr = NA.leer_wav(Path(carpeta) / fichero)
        texto, palabras = jueces.transcribir(Path(carpeta) / fichero)
        r, _ = ritmo(x)
        span = (palabras[-1][1] - palabras[0][0]) if len(palabras) > 1 else float("nan")
        datos[fichero] = {
            "texto": texto, "palabras": palabras,
            "wer": FI.wer(frases[meta["frase"]], texto),
            "huella": jueces.huella(x, sr).tolist(),
            "utmos": NA.utmos_de(jueces.utmos, x, sr),
            "duracion_s": len(x) / sr,
            "palabras_s": (len(palabras) / span) if span and span > 0 else float("nan"),
            "rtf": float(meta["rtf"]) if meta.get("rtf") else float("nan"),
            **r,
        }
        nuevos += 1
        if nuevos % 20 == 0:
            json.dump(datos, open(ruta_cache, "w"))
            print(f"  [{nombre}] {i + 1}/{len(clips)} · {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(datos, open(ruta_cache, "w"))
    return datos, clips, frases


def comparar(base_nombre, base_dir, base, contra_nombre, contra_dir, contra, clips):
    """Lo que es de una PAREJA de clips, fotograma a fotograma."""
    filas = []
    # huella media de cada voz con las dos variantes, para la identidad dejando fuera el clip
    por_voz = {}
    for f, m in clips.items():
        if f in contra:
            por_voz.setdefault(m["voz"], []).append((f, np.array(base[f]["huella"]), np.array(contra[f]["huella"])))
    for f, m in sorted(clips.items()):
        if f not in contra:
            continue
        xb, sr = NA.leer_wav(Path(base_dir) / f)
        xc, _ = NA.leer_wav(Path(contra_dir) / f)
        largo_b, largo_c = len(xb), len(xc)
        xb, xc, desfase = alinear(xb, xc)
        otros = [h for g, hb, hc in por_voz[m["voz"]] if g != f for h in (hb, hc)]
        centro = np.mean(otros, 0)
        b, c = base[f], contra[f]
        pal_b = [w for _, _, w in b["palabras"]]
        pal_c = [w for _, _, w in c["palabras"]]
        mismas = FI.comparable(" ".join(pal_b)) == FI.comparable(" ".join(pal_c))
        desfase = (float(np.median([abs(pc[0] - pb[0]) for pb, pc in zip(b["palabras"], c["palabras"])])) * 1000
                   if mismas and b["palabras"] else float("nan"))
        fila = {
            "fichero": f, "voz": m["voz"], "clon": m["voz"] in CLONES, "frase": m["frase"],
            "semilla": m["semilla"],
            "dur_base_s": b["duracion_s"], "dur_contra_s": c["duracion_s"],
            "mismo_largo": largo_b == largo_c, "desfase_ms": desfase / sr * 1000,
            **timbre(xb, xc, sr),
            **tono(xb, xc),
            "cos_directo": cos(np.array(b["huella"]), np.array(c["huella"])),
            "id_base": cos(np.array(b["huella"]), centro),
            "id_contra": cos(np.array(c["huella"]), centro),
            "wer_base": b["wer"], "wer_contra": c["wer"],
            "texto_base": b["texto"], "texto_contra": c["texto"],
            "transcripcion_identica": FI.comparable(b["texto"]) == FI.comparable(c["texto"]),
            "utmos_base": b["utmos"], "utmos_contra": c["utmos"],
            "d_inicio_ms": (c["inicio_s"] - b["inicio_s"]) * 1000,
            "d_fin_ms": (c["fin_s"] - b["fin_s"]) * 1000,
            "d_pausas": c["pausas"] - b["pausas"], "d_pausa_ms": (c["pausa_s"] - b["pausa_s"]) * 1000,
            "d_palabras_s": c["palabras_s"] - b["palabras_s"],
            "desfase_palabra_ms": desfase,
            "rtf_base": b["rtf"], "rtf_contra": c["rtf"],
        }
        filas.append(fila)
    return filas


def _med(xs):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(v) if v else float("nan")


def _pct(xs, q):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(v, q)) if v else float("nan")


def informe(base_nombre, contra_nombre, filas):
    L = []
    def ic(clave_c, clave_b=None, escala=1.0):
        d = [(f[clave_c] - (f[clave_b] if clave_b else 0)) * escala for f in filas]
        m, lo, hi = ic95(d)
        return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    n = len(filas)
    L.append(f"### {contra_nombre} frente a {base_nombre} ({n} parejas)\n")
    L.append("| familia | medida | valor |")
    L.append("|---|---|---|")
    L.append(f"| forma de onda | mismo largo | {sum(f['mismo_largo'] for f in filas)}/{n} · desplazados (alineados antes de medir) {sum(abs(f['desfase_ms']) > 0 for f in filas)} |")
    L.append(f"| forma de onda | SNR contra la base, mediana (p5) | {_med([f['snr_db'] for f in filas]):.1f} dB ({_pct([f['snr_db'] for f in filas], 5):.1f}) |")
    L.append(f"| timbre | MCD mediana (p95) | {_med([f['mcd_db'] for f in filas]):.2f} dB ({_pct([f['mcd_db'] for f in filas], 95):.2f}) |")
    L.append(f"| timbre | LSD mediana (p95) | {_med([f['lsd_db'] for f in filas]):.2f} dB ({_pct([f['lsd_db'] for f in filas], 95):.2f}) |")
    L.append(f"| timbre | coseno ECAPA directo, mediana (min) | {_med([f['cos_directo'] for f in filas]):.4f} ({min(f['cos_directo'] for f in filas):.4f}) |")
    L.append(f"| timbre | identidad (contra huella de la voz), diferencia media [IC 95 %] | {ic('id_contra', 'id_base')} |")
    L.append(f"| tono | acuerdo de sonoridad, mediana (min) | {100 * _med([f['sonoridad_acuerdo'] for f in filas]):.1f} % ({100 * min(f['sonoridad_acuerdo'] for f in filas):.1f} %) |")
    L.append(f"| tono | desvio de F0 donde suenan las dos, mediana de medianas (p95 de p95) | {_med([f['cents_mediana'] for f in filas]):.1f} cents ({_pct([f['cents_p95'] for f in filas], 95):.1f}) |")
    L.append(f"| tono | tono medio, diferencia [IC] | {ic('tono_st')} st |")
    L.append(f"| tono | recorrido, diferencia [IC] | {ic('d_recorrido_st')} st |")
    L.append(f"| tono | movimiento, diferencia [IC] | {ic('d_movimiento')} st/s |")
    L.append(f"| ritmo | comienzo del habla, diferencia [IC] | {ic('d_inicio_ms')} ms |")
    L.append(f"| ritmo | final del habla, diferencia [IC] | {ic('d_fin_ms')} ms |")
    L.append(f"| ritmo | pausas internas, diferencia [IC] | {ic('d_pausas')} · duracion {ic('d_pausa_ms')} ms |")
    L.append(f"| ritmo | velocidad, diferencia [IC] | {ic('d_palabras_s')} palabras/s |")
    L.append(f"| ritmo | desfase de cada palabra (misma transcripcion), mediana | {_med([f['desfase_palabra_ms'] for f in filas]):.0f} ms |")
    L.append(f"| inteligible | transcripcion identica | {sum(f['transcripcion_identica'] for f in filas)}/{n} |")
    L.append(f"| inteligible | WER medio base → contra | {100 * statistics.mean(f['wer_base'] for f in filas):.2f} % → {100 * statistics.mean(f['wer_contra'] for f in filas):.2f} % · diferencia {ic('wer_contra', 'wer_base', 100)} pts |")
    L.append(f"| inteligible | clips exactos base → contra | {sum(f['wer_base'] == 0 for f in filas)} → {sum(f['wer_contra'] == 0 for f in filas)} |")
    L.append(f"| naturalidad | UTMOS medio base → contra | {statistics.mean(f['utmos_base'] for f in filas):.3f} → {statistics.mean(f['utmos_contra'] for f in filas):.3f} · diferencia {ic('utmos_contra', 'utmos_base')} |")
    L.append(f"| velocidad | RTF medio base → contra | {_med([f['rtf_base'] for f in filas]):.3f} → {_med([f['rtf_contra'] for f in filas]):.3f} (medianas) |")
    L.append("")
    L.append("Por voz (los clones primero):\n")
    L.append("| voz | n | MCD dB | cents med | Δ tono st | Δ recorrido st | identidad base → contra | cos directo | WER base → contra | UTMOS base → contra | transcr. idénticas |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    voces = sorted({f["voz"] for f in filas}, key=lambda v: (v not in CLONES, v))
    for v in voces:
        g = [f for f in filas if f["voz"] == v]
        L.append(f"| {v}{' (clon)' if v in CLONES else ''} | {len(g)} | {_med([f['mcd_db'] for f in g]):.2f} | "
                 f"{_med([f['cents_mediana'] for f in g]):.1f} | {statistics.mean(f['tono_st'] for f in g):+.3f} | "
                 f"{statistics.mean(f['d_recorrido_st'] for f in g):+.3f} | "
                 f"{statistics.mean(f['id_base'] for f in g):.4f} → {statistics.mean(f['id_contra'] for f in g):.4f} | "
                 f"{_med([f['cos_directo'] for f in g]):.4f} | "
                 f"{100 * statistics.mean(f['wer_base'] for f in g):.1f} → {100 * statistics.mean(f['wer_contra'] for f in g):.1f} % | "
                 f"{statistics.mean(f['utmos_base'] for f in g):.3f} → {statistics.mean(f['utmos_contra'] for f in g):.3f} | "
                 f"{sum(f['transcripcion_identica'] for f in g)}/{len(g)} |")
    L.append("")
    peores = sorted(filas, key=lambda f: -f["mcd_db"])[:5]
    L.append("Los 5 clips que mas se separan (MCD), para escucharlos:\n")
    for f in peores:
        L.append(f"- `{f['fichero']}` · MCD {f['mcd_db']:.2f} dB · {f['cents_mediana']:.0f} cents · WER {100 * f['wer_base']:.0f} → {100 * f['wer_contra']:.0f} % · "
                 f"«{f['texto_base'][:60]}» / «{f['texto_contra'][:60]}»")
    distintas = [f for f in filas if not f["transcripcion_identica"]]
    if distintas:
        L.append(f"\nTranscripciones que cambian ({len(distintas)}):\n")
        for f in distintas[:12]:
            L.append(f"- `{f['fichero']}` · base «{f['texto_base']}» · contra «{f['texto_contra']}»")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--contra", nargs="+", default=[])
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--salida", default=None, help="carpeta del informe (por defecto, junto a --base)")
    ap.add_argument("--solo", action="store_true",
                    help="solo analiza --base y deja su cache (whisper, huella, UTMOS): para ir "
                         "adelantando variantes segun se generan")
    a = ap.parse_args()
    salida = Path(a.salida or Path(a.base).parent / "informe")
    salida.mkdir(parents=True, exist_ok=True)
    jueces = Jueces(a.whisper)
    if a.solo:
        datos, _, _ = analizar_variante(Path(a.base).name, a.base, jueces, salida)
        print(f"{Path(a.base).name}: {len(datos)} clips analizados en {salida}")
        return
    if not a.contra:
        ap.error("hace falta --contra (o --solo)")
    base_nombre = Path(a.base).name
    base, clips, _ = analizar_variante(base_nombre, a.base, jueces, salida)
    partes = [f"# Banco A/B · base `{base_nombre}` · whisper {a.whisper}\n"]
    for contra_dir in a.contra:
        nombre = Path(contra_dir).name
        contra, _, _ = analizar_variante(nombre, contra_dir, jueces, salida)
        filas = comparar(base_nombre, a.base, base, nombre, contra_dir, contra, clips)
        with open(salida / f"{nombre}_frente_a_{base_nombre}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        partes.append(informe(base_nombre, nombre, filas))
    (salida / "informe.md").write_text("\n".join(partes), encoding="utf-8")
    print("\n".join(partes))
    print(f"informe en {salida / 'informe.md'}")


if __name__ == "__main__":
    main()
