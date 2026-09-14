#!/usr/bin/env python
"""Perfil vocal de un clip: lo que hace que una voz suene a ESA persona, en números.

    python scripts/perfil_vocal.py clip.wav [otro.wav ...]
    python scripts/perfil_vocal.py --manifiesto dataset/manifiesto.csv --raiz dataset   # agregado por persona

Es la medida común de la fase 1 del plan de personalidad (docs/plan-personalidad-voz.md): la
misma función sobre el audio REAL de una persona y sobre su CLON diciendo lo mismo, para ver en
qué descriptor está lo «plano». Solo numpy y librosa.

  hz, recorrido, desviacion, movimiento   tono con octavas corregidas (prosodia.descripcion)
  silabas_s                               velocidad: grupos vocálicos del texto / duración
                                          (solo si se pasa el texto)
  pausas_min                              pausas internas >= 150 ms por minuto
  rango_db                                dinámica de energía: p95 - p10 de la envolvente activa
  inclinacion_db                          energía 1-8 kHz frente a 50 Hz-1 kHz: más negativo es
                                          una voz más oscura o más suave
  hnr_db                                  parte armónica frente a la percusiva (HPSS de librosa):
                                          aproximación a la «limpieza» de la voz, no el HNR de Praat
  microvariacion_cents                    variación rápida del F0 entre ventanas vecinas donde hay
                                          voz (mediana del salto en cents): lo que se pierde
                                          cuando una voz suena «de máquina» aunque su recorrido
                                          sea el mismo
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prosodia as PR  # noqa: E402

HOP = 240
VOCALES = re.compile(r"[aeiouáéíóúü]+", re.I)


def perfil(x, sr, texto=None):
    import librosa
    x = np.asarray(x, dtype=np.float32)
    dur = len(x) / sr
    d = PR.descripcion(x, sr)
    f0 = PR.corregir_octavas(PR.contorno(x, sr))
    saltos = [abs(1200 * np.log2(f0[i] / f0[i - 1])) for i in range(1, len(f0))
              if not np.isnan(f0[i]) and not np.isnan(f0[i - 1])]
    rms = librosa.feature.rms(y=x, frame_length=2 * HOP, hop_length=HOP)[0]
    db = 20 * np.log10(rms / (rms.max() + 1e-9) + 1e-9)
    activo = db > -35
    rango = float(np.percentile(db[activo], 95) - np.percentile(db[activo], 10)) if activo.sum() > 10 else float("nan")
    pausas, largo = 0, 0
    if activo.any():
        i0, i1 = int(np.argmax(activo)), len(activo) - int(np.argmax(activo[::-1]))
        for a in activo[i0:i1]:
            if not a:
                largo += 1
            else:
                pausas += largo >= 15
                largo = 0
    esp = np.abs(librosa.stft(x, n_fft=1024, hop_length=HOP)) ** 2
    frec = librosa.fft_frequencies(sr=sr, n_fft=1024)
    m = min(esp.shape[1], len(activo))
    sel = esp[:, :m][:, activo[:m]] if activo[:m].sum() > 5 else esp
    media = sel.mean(1) + 1e-12
    inclinacion = 10 * np.log10(media[(frec > 1000) & (frec < 8000)].sum() / media[(frec > 50) & (frec < 1000)].sum())
    armonico, _ = librosa.effects.hpss(x)
    hnr = 10 * np.log10((armonico ** 2).sum() / (((x - armonico) ** 2).sum() + 1e-12))
    return {
        "duracion_s": dur, "hz": d["hz"], "recorrido": d["recorrido"], "desviacion": d["desviacion"],
        "movimiento": d["movimiento"], "tonal": d["tonal"],
        "microvariacion_cents": float(np.median(saltos)) if saltos else float("nan"),
        "silabas_s": (len(VOCALES.findall(texto)) / dur) if texto else float("nan"),
        "pausas_min": pausas / (dur / 60) if dur else float("nan"),
        "rango_db": rango, "inclinacion_db": float(inclinacion), "hnr_db": float(hnr),
    }


CLAVES = ["hz", "recorrido", "desviacion", "movimiento", "microvariacion_cents", "silabas_s",
          "pausas_min", "rango_db", "inclinacion_db", "hnr_db"]


def leer(ruta):
    import soundfile as sf
    x, sr = sf.read(str(ruta), dtype="float32")
    return (x.mean(1) if x.ndim > 1 else x), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--manifiesto", help="CSV con columnas fichero, hablante y texto")
    ap.add_argument("--raiz", default=".", help="carpeta desde la que se leen los ficheros del manifiesto")
    a = ap.parse_args()
    if a.manifiesto:
        por = {}
        for f in csv.DictReader(open(a.manifiesto, encoding="utf-8")):
            x, sr = leer(Path(a.raiz) / f["fichero"])
            por.setdefault(f["hablante"], []).append(perfil(x, sr, f.get("texto")))
        print("persona".ljust(24) + "min".rjust(6) + "".join(k[:11].rjust(12) for k in CLAVES))
        for h, v in sorted(por.items(), key=lambda kv: -sum(p["duracion_s"] for p in kv[1])):
            fila = [statistics.median([p[k] for p in v if p[k] == p[k]] or [float("nan")]) for k in CLAVES]
            print(h.ljust(24) + f"{sum(p['duracion_s'] for p in v) / 60:6.2f}" + "".join(f"{x:12.2f}" for x in fila))
        return
    for ruta in a.wavs:
        x, sr = leer(ruta)
        p = perfil(x, sr)
        print(ruta, " ".join(f"{k}={p[k]:.2f}" for k in CLAVES if p[k] == p[k]))


if __name__ == "__main__":
    main()
