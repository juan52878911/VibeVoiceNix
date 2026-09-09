#!/usr/bin/env python
"""Cuanto vale corregir el tono a la salida: antes y despues, con el mismo juez.

    python scripts/banco_tono.py --carpeta banco/laura --semilla 1 --referencia ref.wav

Toma los clips que dejo banco_semillas.py (o cualquier carpeta de wav de una
voz) y prueba tres cosas sobre CADA clip, sin sintetizar nada:

    sin corregir            lo que sale del motor
    correccion por voz      un solo desplazamiento para la voz: el sesgo
                            medio medido sobre esos clips, con signo cambiado.
                            Es lo que puede aplicar el servidor por voz.
    correccion por clip     el sesgo de ESE clip, con signo cambiado: el tope
                            de lo que se puede ganar corrigiendo el tono.

Y mide cada version contra la referencia: ECAPA, sesgo_st, WER. Si la
correccion por voz acerca el tono sin bajar la identidad ni subir el WER, va
al servidor (voz_stream: tono_st por voz). Si baja la identidad, no.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banco_clonado import Huella, wer  # noqa: E402
from banco_motores import TEXTOS  # noqa: E402
from clonar_voz import leer_audio  # noqa: E402
from tono import RITMO, desplazar_tono, escribir_wav, f0_mediana, leer_wav, sesgo_st  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", required=True)
    ap.add_argument("--semilla", type=int, default=None,
                    help="solo los clips de esa semilla de clonado (<nombre>-clonN-*.wav)")
    ap.add_argument("--referencia", required=True)
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--sin-wer", action="store_true")
    ap.add_argument("--salida", default=None, help="carpeta para los wav corregidos")
    ap.add_argument("--metodo", choices=["psola", "remuestreo"], default="psola")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    patron = f"*-clon{args.semilla}-*.wav" if args.semilla is not None else "*.wav"
    clips = sorted(p for p in carpeta.glob(patron) if "referencia" not in p.name
                   and "-corr" not in p.name)
    if not clips:
        raise SystemExit(f"sin clips en {carpeta} ({patron})")
    ref = leer_audio(args.referencia)
    huella = Huella()
    h_ref = huella(ref)
    f0_ref = f0_mediana(ref)
    whisper = None
    if not args.sin_wer:
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    salida = Path(args.salida) if args.salida else carpeta / "corregidos"
    salida.mkdir(parents=True, exist_ok=True)

    def texto_de(nombre):
        # <nombre>-clonN-<idioma>_<clave>_<semilla>.wav
        cola = nombre.rsplit("-", 1)[-1]
        idioma, clave, _ = cola.rsplit("_", 2)
        return idioma, TEXTOS[idioma][clave]

    def medir(x, idioma, texto, ruta_tmp):
        m = {"ecapa": float(huella(x) @ h_ref), "sesgo_st": sesgo_st(x, f0_ref)}
        if whisper is not None:
            escribir_wav(ruta_tmp, x)
            segs, _ = whisper.transcribe(str(ruta_tmp), language=idioma, beam_size=5)
            m["wer"] = wer(texto, " ".join(s.text for s in segs))
        return m

    # 1. el sesgo de cada clip y el medio de la voz
    xs, sesgos = {}, {}
    for p in clips:
        x, hz = leer_wav(p)
        xs[p] = x
        sesgos[p] = sesgo_st(x, f0_ref)
    validos = [s for s in sesgos.values() if s is not None]
    # mediana, no media: un clip con error de octava (-12 st) no debe decidir
    medio = float(np.median(validos)) if validos else 0.0
    print(f"referencia f0 {f0_ref:.0f} Hz; {len(clips)} clips; sesgo medio {medio:+.2f} st "
          f"(de {min(validos):+.2f} a {max(validos):+.2f})")

    filas = []
    for p in clips:
        idioma, texto = texto_de(p.stem)
        x = xs[p]
        versiones = {
            "sin": x,
            "voz": desplazar_tono(x, -medio, metodo=args.metodo, f0_ref=f0_ref),
            "clip": desplazar_tono(x, -(sesgos[p] or 0.0), metodo=args.metodo, f0_ref=f0_ref),
        }
        for k, y in versiones.items():
            if k != "sin":
                escribir_wav(salida / f"{p.stem}-corr_{k}.wav", y)
            m = medir(y, idioma, texto, salida / "_tmp.wav")
            filas.append({"clip": p.name, "idioma": idioma, "version": k, **m})
            print(f"  {p.name:44s} {k:4s} ECAPA {m['ecapa']:.3f}  st {m['sesgo_st']:+.2f}"
                  + (f"  WER {m['wer']:.3f}" if "wer" in m else ""))

    with open(salida / "banco_tono.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "idioma", "version", "ecapa", "sesgo_st", "wer"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(filas)

    print("\n| version | ECAPA media | ECAPA min | sesgo st medio | |sesgo| medio | WER |")
    print("|---|---|---|---|---|---|")
    for k in ("sin", "voz", "clip"):
        f = [r for r in filas if r["version"] == k]
        e = [r["ecapa"] for r in f]
        s = [r["sesgo_st"] for r in f if r["sesgo_st"] is not None]
        wv = [r["wer"] for r in f if "wer" in r]
        print(f"| {k} | {np.mean(e):.3f} | {min(e):.3f} | {np.mean(s):+.2f} | {np.mean(np.abs(s)):.2f} | "
              f"{np.mean(wv):.3f} |" if wv else
              f"| {k} | {np.mean(e):.3f} | {min(e):.3f} | {np.mean(s):+.2f} | {np.mean(np.abs(s)):.2f} | - |")
    print(f"\ncorreccion por voz ({args.metodo}): {-medio:+.2f} st; wav en {salida}")


if __name__ == "__main__":
    main()
