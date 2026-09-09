#!/usr/bin/env python
"""Elegir la SEMILLA del clonado de una voz midiendo, no a ciegas.

    python scripts/banco_semillas.py --audio nota.opus \\
        --transcripcion "lo que dice" --nombre juan --salida voces/

QUE PASA SIN ESTO
El codificador acustico muestrea (`e.sample`, torch.randn) y clonar_voz.py
nunca fijaba la semilla: dos clones de la MISMA referencia salian distintos.
MEDIDO en el doblaje del video de 4 voces (dobla, 9 de septiembre): con las
mismas referencias, la identidad QC de una voz fue 0,53 en una corrida y 0,39
en la siguiente. Esa varianza tapa cualquier mejora pequena y, peor, decide
por sorteo como suena una persona.

QUE HACE
Fabrica el prefijo con varias semillas de CLONADO (el mismo modelo cargado
una vez), y con cada prefijo sintetiza las mismas frases con las mismas
semillas de SINTESIS. Mide cada clon contra la referencia:

    ECAPA       identidad media y minima (el juez de oido.py); techo al lado
    sesgo_st    12*log2(f0 clon / f0 referencia): el tono; +2,5 st es el
                defecto conocido de las voces graves
    car/s       ritmo sobre el tiempo con voz
    WER         faster-whisper small, en el idioma de cada frase

Ordena las semillas por identidad media, desempata por |sesgo_st| y WER, y
deja <nombre>.pt con la ganadora mas <nombre>.json (semilla, techo y la tabla
entera). La tabla es la parte importante: se ven todos los tonos que puede
sacar el clonado de esa grabacion y cuanto vale cada uno.

DONDE CORRE
Estacion de trabajo (modelo + codificador, ~5,4 GB) o el worker de dobla en
AWS. Coste: semillas_clon x frases x semillas_sintesis generaciones (por
defecto 5 x 4 x 2 = 40, unos 5 min a RTF 1) mas la puntuacion.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banco_duracion import cargar_modelo, construir_prefijo, hablar  # noqa: E402
from banco_motores import TEXTOS, duracion_habla, escribir_wav  # noqa: E402
from clonar_voz import RITMO, a_cpu, avisar_calidad, igualar_volumen, leer_audio  # noqa: E402

FRASES = [("es", "neutro"), ("es", "expresivo"), ("es", "largo"), ("en", "neutro")]


def guardar_prefijo(prefijo, ruta):
    """Como clonar_voz.py: las caches a CPU para que el .pt cargue en cualquier sitio."""
    p = {k: a_cpu(copy.deepcopy(v)) for k, v in prefijo.items()}
    torch.save(p, ruta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", action="append", required=True)
    ap.add_argument("--transcripcion", action="append", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--salida", default="voces")
    ap.add_argument("--semillas-clon", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--semillas-sintesis", type=int, nargs="+", default=[11, 42])
    ap.add_argument("--frases", nargs="+", default=[f"{i}:{k}" for i, k in FRASES],
                    help="idioma:clave de banco_motores.TEXTOS")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--sin-wer", action="store_true", help="sin whisper (mas rapido)")
    args = ap.parse_args()
    if len(args.audio) != len(args.transcripcion):
        raise SystemExit("un --transcripcion por cada --audio, en el mismo orden")
    disp = args.dispositivo
    if disp == "auto":
        disp = "mps" if torch.backends.mps.is_available() else "cpu"
    frases = [tuple(f.split(":", 1)) for f in args.frases]

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    clips = []
    for ruta in args.audio:
        x = leer_audio(ruta)
        avisar_calidad(x, sangria="  ")
        clips.append(x)
    if len(clips) > 1:
        clips = igualar_volumen(clips)
    referencia = np.concatenate(clips)
    escribir_wav(salida / f"{args.nombre}-referencia.wav", referencia)

    # jueces
    from banco_clonado import Huella, wer
    from prosodia import descripcion
    from techo import techo_de_clips
    huella = Huella()
    h_ref = huella(referencia)
    techo = techo_de_clips(clips)
    f0_ref = descripcion(referencia)["hz"]
    whisper = None
    if not args.sin_wer:
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    print(f"referencia: {len(referencia) / RITMO:.1f} s, techo {techo}, f0 {f0_ref:.0f} Hz")

    t0 = time.perf_counter()
    proc, modelo = cargar_modelo(args.modelo, args.cache, disp, args.pasos, con_encoder=True)
    print(f"modelo cargado en {time.perf_counter() - t0:.0f} s ({disp})")

    filas, resumen = [], {}
    for sc in args.semillas_clon:
        torch.manual_seed(sc)             # el sorteo del codificador
        prefijo, n, M = construir_prefijo(proc, modelo, clips, args.transcripcion, disp)
        ruta_pt = salida / f"{args.nombre}-clon{sc}.pt"
        guardar_prefijo(prefijo, ruta_pt)
        medidas = []
        for idioma, clave in frases:
            texto = TEXTOS[idioma][clave]
            for ss in args.semillas_sintesis:
                t1 = time.perf_counter()
                x = hablar(proc, modelo, prefijo, texto, args.cfg, ss, disp)
                dt = time.perf_counter() - t1
                wav = salida / f"{args.nombre}-clon{sc}-{idioma}_{clave}_{ss}.wav"
                escribir_wav(wav, x)
                f0 = descripcion(x)["hz"]
                habla = duracion_habla(x)
                fila = {"semilla_clon": sc, "idioma": idioma, "texto": clave, "semilla": ss,
                        "ecapa": round(float(huella(x) @ h_ref), 3),
                        "sesgo_st": round(12 * math.log2(f0 / f0_ref), 2) if f0 and f0_ref else None,
                        "car_s": round(len(texto) / habla, 1) if habla else None,
                        "dur_s": round(len(x) / RITMO, 2), "rtf": round(dt / (len(x) / RITMO), 2)}
                if whisper is not None:
                    segs, _ = whisper.transcribe(str(wav), language=idioma, beam_size=5)
                    fila["wer"] = round(wer(texto, " ".join(s.text for s in segs)), 3)
                filas.append(fila); medidas.append(fila)
                print(f"  clon {sc} {idioma}_{clave}_{ss}: ECAPA {fila['ecapa']:.3f} "
                      f"st {fila['sesgo_st']:+.1f} {fila['car_s']} car/s"
                      + (f" WER {fila['wer']:.2f}" if 'wer' in fila else ""))
        e = [m["ecapa"] for m in medidas]
        st = [m["sesgo_st"] for m in medidas if m["sesgo_st"] is not None]
        w = [m["wer"] for m in medidas if "wer" in m]
        resumen[sc] = {"ecapa": round(float(np.mean(e)), 3), "ecapa_min": round(min(e), 3),
                       "ecapa_es": round(float(np.mean([m["ecapa"] for m in medidas if m["idioma"] == "es"])), 3),
                       "ecapa_en": round(float(np.mean([m["ecapa"] for m in medidas if m["idioma"] == "en"] or [0])), 3),
                       "sesgo_st": round(float(np.mean(st)), 2) if st else None,
                       "car_s": round(float(np.mean([m["car_s"] for m in medidas if m["car_s"]])), 1),
                       "wer": round(float(np.mean(w)), 3) if w else None,
                       "posiciones": n + M}
        print(f"== semilla de clonado {sc}: ECAPA {resumen[sc]['ecapa']:.3f} (min {resumen[sc]['ecapa_min']:.3f}, "
              f"es {resumen[sc]['ecapa_es']:.3f}, en {resumen[sc]['ecapa_en']:.3f}) "
              f"sesgo {resumen[sc]['sesgo_st']:+.1f} st  WER {resumen[sc]['wer']}")

    def orden(sc):
        r = resumen[sc]
        return (-r["ecapa"], abs(r["sesgo_st"] or 0), r["wer"] or 0)
    ganadora = sorted(resumen, key=orden)[0]

    with open(salida / f"{args.nombre}-semillas.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for f in filas for k in f}))
        wr.writeheader(); wr.writerows(filas)
    (salida / f"{args.nombre}-clon{ganadora}.pt").replace(salida / f"{args.nombre}.pt")
    (salida / f"{args.nombre}.json").write_text(json.dumps({
        "semilla_clon": ganadora, "techo": techo, "f0_ref": round(f0_ref),
        "segundos": round(len(referencia) / RITMO, 1), "fuentes": args.audio,
        "cfg": args.cfg, "pasos": args.pasos,
        "semillas": {str(k): v for k, v in resumen.items()}}, ensure_ascii=False, indent=1))

    print("\n| semilla | ECAPA | min | es | en | sesgo st | car/s | WER |")
    print("|---|---|---|---|---|---|---|---|")
    for sc in sorted(resumen, key=orden):
        r = resumen[sc]
        print(f"| {sc}{' *' if sc == ganadora else ''} | {r['ecapa']:.3f} | {r['ecapa_min']:.3f} | "
              f"{r['ecapa_es']:.3f} | {r['ecapa_en']:.3f} | {r['sesgo_st']:+.1f} | {r['car_s']} | "
              f"{r['wer'] if r['wer'] is not None else '-'} |")
    print(f"\ntecho {techo}; ganadora {ganadora} -> {salida / (args.nombre + '.pt')} "
          f"(+ .json con la tabla)")


if __name__ == "__main__":
    main()
