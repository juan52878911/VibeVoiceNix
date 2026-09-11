#!/usr/bin/env python
"""Puntua la NATURALIDAD de los clips de un banco, y compara bancos entre si.

    pkgs/vibevoice/.venv/bin/python scripts/naturalidad.py puntuar --dir banco/pasos6
    pkgs/vibevoice/.venv/bin/python scripts/naturalidad.py comparar \\
        --base banco/pasos6 --contra banco/pasos8 banco/pasos10

POR QUE HACE FALTA
El banco de fidelidad (scripts/fidelidad.py) cierra el circuito texto -> voz ->
whisper -> texto y mide si la voz DICE lo que se le pidio. Lo que no mide es si
suena natural: whisper entiende perfectamente una voz horrible. Asi que los
pasos de difusion, el freno de la guia y el cfg se venian decidiendo por WER o a
oido. Esto pone un numero al lado.

EL JUEZ: UTMOS22 (strong), el predictor de MOS del VoiceMOS Challenge 2022,
via torch.hub (tarepan/SpeechMOS). Entrenado a 16 kHz sobre voz sintetica, que
es justo lo que se juzga aqui. La primera vez torch.hub baja los pesos a
~/.cache/torch/hub (cientos de MB); despues es local. Cuesta ~0,2 s por clip
en CPU.

COMO LEERLO, Y COMO NO
El valor absoluto de UTMOS no es comparable entre jueces ni con MOS humanos de
otro banco. Lo que vale es la DIFERENCIA emparejada dentro del mismo banco:
mismo texto, misma semilla, misma voz, y solo cambia la variante del servicio.
Por eso `comparar` empareja por (frase, semilla) y saca la media de las
diferencias y en cuantos clips gana cada lado, ademas de las medias sueltas.

ENTRADA: la carpeta que deja fidelidad.py --semillas (WAV a 24 kHz y clips.csv
con una fila por clip). Si no hay CSV, se puntuan los WAV sueltos igual.
SALIDA: puntuacion.csv en la misma carpeta (las columnas del CSV de entrada
mas `utmos`), y por pantalla media, minimo y percentil 10.
"""
import argparse
import csv
import os
import statistics
import sys
import wave
from pathlib import Path

RITMO_JUEZ = 16_000


def leer_wav(ruta):
    import numpy as np
    with wave.open(str(ruta), "rb") as w:
        hz = w.getframerate()
        n = w.getnframes()
        crudo = w.readframes(n)
        ancho = w.getsampwidth()
        canales = w.getnchannels()
    if ancho != 2:
        raise ValueError(f"{ruta}: se esperaba PCM de 16 bits, hay {ancho * 8}")
    x = np.frombuffer(crudo, dtype="<i2").astype("float32") / 32768.0
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    return x, hz


def cargar_juez():
    import torch
    # trust_repo=True: sin el, torch.hub pregunta por consola la primera vez y
    # un banco lanzado con nohup se queda colgado esperando una respuesta.
    juez = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                          trust_repo=True)
    juez.eval()
    return juez


def utmos_de(juez, x, hz):
    import librosa
    import torch
    if hz != RITMO_JUEZ:
        x = librosa.resample(x, orig_sr=hz, target_sr=RITMO_JUEZ)
    with torch.no_grad():
        return float(juez(torch.from_numpy(x).float()[None], RITMO_JUEZ).item())


def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta, filas, campos):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def resumen(valores):
    if not valores:
        return "sin datos"
    v = sorted(valores)
    p10 = v[max(0, int(round(0.10 * (len(v) - 1))))]
    return (f"media {statistics.mean(v):.3f} · min {v[0]:.3f} · p10 {p10:.3f} "
            f"· max {v[-1]:.3f} · n {len(v)}")


# ------------------------------------------------------------------ puntuar
def puntuar(args):
    d = Path(args.dir)
    ruta_csv = Path(args.csv) if args.csv else d / "clips.csv"
    filas = leer_csv(ruta_csv) if ruta_csv.exists() else None
    if filas is None:
        wavs = sorted(p.name for p in d.glob("*.wav"))
        if not wavs:
            sys.exit(f"no hay ni {ruta_csv} ni WAV en {d}")
        filas = [{"fichero": w} for w in wavs]
        print(f"sin CSV: se puntuan {len(filas)} WAV sueltos de {d}")
    juez = cargar_juez()
    for f in filas:
        ruta = d / f["fichero"]
        if not ruta.exists():
            f["utmos"] = ""
            print(f"  {f['fichero']:24s} FALTA")
            continue
        x, hz = leer_wav(ruta)
        f["utmos"] = round(utmos_de(juez, x, hz), 4)
        print(f"  {f['fichero']:24s} utmos {f['utmos']:.3f}"
              + (f"  wer {float(f['wer']):.0%}" if f.get("wer") else ""))
    campos = list(filas[0].keys())
    if "utmos" not in campos:
        campos.append("utmos")
    salida = d / "puntuacion.csv"
    escribir_csv(salida, filas, campos)
    valores = [float(f["utmos"]) for f in filas if f.get("utmos") != ""]
    print(f"\nUTMOS: {resumen(valores)}")
    print(f"puntuacion en {salida}")


# ----------------------------------------------------------------- comparar
def _clave(f):
    return (f.get("frase", ""), f.get("semilla", ""))


def _tabla(nombre, filas):
    """Las cifras sueltas de un banco: WER, exactos, UTMOS, RTF."""
    wers = [float(f["wer"]) for f in filas if f.get("wer") not in (None, "")]
    ut = [float(f["utmos"]) for f in filas if f.get("utmos") not in (None, "")]
    rtf = [float(f["rtf"]) for f in filas if f.get("rtf") not in (None, "")]
    exactos = sum(int(f.get("exacto") or 0) for f in filas)
    pasos = sorted({f.get("pasos", "") for f in filas} - {""})
    return {
        "banco": nombre,
        "pasos": ",".join(pasos) if pasos else "-",
        "clips": len(filas),
        "wer_medio": statistics.mean(wers) if wers else None,
        "wer_peor": max(wers) if wers else None,
        "exactos": exactos,
        "utmos_medio": statistics.mean(ut) if ut else None,
        "utmos_min": min(ut) if ut else None,
        "utmos_p10": (sorted(ut)[max(0, int(round(0.10 * (len(ut) - 1))))]
                      if ut else None),
        "rtf_medio": statistics.mean(rtf) if rtf else None,
    }


def _fmt(v, pct=False, dec=3):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.{dec}f}"


def comparar(args):
    base_dir = Path(args.base)
    base = leer_csv(base_dir / "puntuacion.csv")
    por_clave = {_clave(f): f for f in base}
    print(f"base: {base_dir} ({len(base)} clips)\n")
    cabecera = ("| banco | pasos | clips | WER medio | WER peor | exactos | "
                "UTMOS medio | UTMOS min | UTMOS p10 | RTF medio | "
                "ΔUTMOS medio | mejora en |")
    print(cabecera)
    print("|" + "---|" * 12)
    t = _tabla(base_dir.name, base)
    print(f"| {t['banco']} | {t['pasos']} | {t['clips']} | {_fmt(t['wer_medio'], True)} "
          f"| {_fmt(t['wer_peor'], True)} | {t['exactos']}/{t['clips']} "
          f"| {_fmt(t['utmos_medio'])} | {_fmt(t['utmos_min'])} | {_fmt(t['utmos_p10'])} "
          f"| {_fmt(t['rtf_medio'])} | — | — |")
    for otro_dir in args.contra:
        otro_dir = Path(otro_dir)
        otro = leer_csv(otro_dir / "puntuacion.csv")
        t = _tabla(otro_dir.name, otro)
        deltas, mejor = [], 0
        for f in otro:
            b = por_clave.get(_clave(f))
            if b is None or not f.get("utmos") or not b.get("utmos"):
                continue
            d = float(f["utmos"]) - float(b["utmos"])
            deltas.append(d)
            mejor += d > 0
        delta = statistics.mean(deltas) if deltas else None
        print(f"| {t['banco']} | {t['pasos']} | {t['clips']} | {_fmt(t['wer_medio'], True)} "
              f"| {_fmt(t['wer_peor'], True)} | {t['exactos']}/{t['clips']} "
              f"| {_fmt(t['utmos_medio'])} | {_fmt(t['utmos_min'])} | {_fmt(t['utmos_p10'])} "
              f"| {_fmt(t['rtf_medio'])} | {('+' if delta and delta > 0 else '') + _fmt(delta)} "
              f"| {mejor}/{len(deltas)} |")
    print("\nΔUTMOS es la media de las diferencias clip a clip (misma frase, misma "
          "semilla) frente a la base; 'mejora en' cuenta en cuantos clips sube.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="orden", required=True)
    p = sub.add_parser("puntuar", help="UTMOS por clip de una carpeta")
    p.add_argument("--dir", required=True, help="carpeta con los WAV y clips.csv")
    p.add_argument("--csv", default=None, help="otro CSV de entrada")
    p.set_defaults(fn=puntuar)
    c = sub.add_parser("comparar", help="tabla emparejada entre bancos")
    c.add_argument("--base", required=True, help="carpeta de referencia (con puntuacion.csv)")
    c.add_argument("--contra", nargs="+", required=True, help="carpetas a comparar")
    c.set_defaults(fn=comparar)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
