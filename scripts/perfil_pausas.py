#!/usr/bin/env python
"""Mide el perfil de pausas de una persona en su audio REAL y lo guarda en la ficha de su voz.

    # de clips sueltos (WAV 24 kHz mono), con sus textos si se quieren las sílabas/s
    python scripts/perfil_pausas.py --ficha voces/juan.json --audio a.wav --texto "..." --audio b.wav --texto "..."

    # de un vídeo anotado: la pista de voces separada y la anotación del editor de dobla
    python scripts/perfil_pausas.py --ficha voces/charla-h0.json --pista voces24k.wav \
        --anotacion anotacion.json --hablante 0

Es lo que usa `forma` en voz-stream (bloque FORMA de voz_stream.py): cada pausa que el modelo hace
pasa a durar lo que duran las de esa persona. La ficha queda con

    "pausas": {"dist": [segundos...], "n", "pausas_min", "mediana_s", "segundos", "silabas_s"?, "detector"}

y el servicio la lee junto al .pt; dobla también puede mandar `pausas` (la lista `dist`) en cada
petición a /tts/stream. Se mide con el MISMO detector causal con el que se conforma (pausas.py): con
otro, las duraciones no casarían.

Con --anotacion solo entran segmentos del hablante de al menos --min-seg segundos que no se pisen con
otro hablante más de --pureza segundos (la regla de pureza de dobla): una pausa que en realidad es la
otra persona hablando falsearía la distribución.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "pkgs" / "vibevoice-cli"))
import pausas as P  # noqa: E402


def leer_wav(ruta):
    try:
        import soundfile as sf
        x, sr = sf.read(str(ruta), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
    except ImportError:
        with wave.open(str(ruta)) as w:
            if w.getsampwidth() != 2:
                raise SystemExit(f"{ruta}: sin soundfile solo se lee WAV PCM16")
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
            if w.getnchannels() > 1:
                x = x.reshape(-1, w.getnchannels()).mean(1)
    if sr != P.RITMO:
        raise SystemExit(f"{ruta}: está a {sr} Hz y el detector espera {P.RITMO} (convierte antes con ffmpeg -ar 24000 -ac 1)")
    return x.astype(np.float32)


def segmentos_puros(anotacion, hablante, min_seg, pureza):
    segs = anotacion["segmentos"]
    propios = [s for s in segs if str(s["hablante"]) == str(hablante)]
    ajenos = [s for s in segs if str(s["hablante"]) != str(hablante)]
    out = []
    for s in propios:
        if s["fin"] - s["ini"] < min_seg:
            continue
        pisado = sum(max(0.0, min(s["fin"], o["fin"]) - max(s["ini"], o["ini"])) for o in ajenos)
        if pisado <= pureza:
            out.append(s)
    return out, len(propios)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ficha", required=True, help="<voz>.json (se crea si no existe; se conserva lo demás)")
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--texto", action="append", default=[])
    ap.add_argument("--pista", help="pista de voces separada del vídeo (voces24k.wav)")
    ap.add_argument("--anotacion", help="anotación de dobla: {segmentos: [{ini, fin, hablante, texto}]}")
    ap.add_argument("--hablante")
    ap.add_argument("--min-seg", type=float, default=3.0)
    ap.add_argument("--pureza", type=float, default=0.3)
    ap.add_argument("--solo-medir", action="store_true", help="imprime el perfil sin tocar la ficha")
    a = ap.parse_args()

    if a.pista or a.anotacion:
        if not (a.pista and a.anotacion and a.hablante is not None):
            ap.error("con --pista hacen falta --anotacion y --hablante")
        anot = json.load(open(a.anotacion, encoding="utf-8"))
        segs, total = segmentos_puros(anot, a.hablante, a.min_seg, a.pureza)
        pista = leer_wav(a.pista)
        clips = [pista[int(s["ini"] * P.RITMO):int(s["fin"] * P.RITMO)] for s in segs]
        textos = [s.get("texto", "") for s in segs]
        fuente = {"pista": Path(a.pista).name, "anotacion": Path(a.anotacion).name, "hablante": a.hablante,
                  "segmentos": len(segs), "de": total, "min_seg": a.min_seg, "pureza": a.pureza}
        print(f"hablante {a.hablante}: {len(segs)} de {total} segmentos puros y de >= {a.min_seg} s", flush=True)
    else:
        if not a.audio:
            ap.error("hace falta --audio o --pista/--anotacion")
        if a.texto and len(a.texto) != len(a.audio):
            ap.error("un --texto por --audio, o ninguno")
        clips = [leer_wav(r) for r in a.audio]
        textos = a.texto or None
        fuente = {"audios": [Path(r).name for r in a.audio]}

    perfil = P.perfil(clips, textos if textos and all(textos) else None)
    perfil["fuente"] = fuente
    resumen = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in perfil.items() if k != "dist"}
    print(f"perfil: {json.dumps(resumen, ensure_ascii=False)}")
    print(f"  duración de pausa p25/p50/p75/p90: {np.percentile(perfil['dist'], [25, 50, 75, 90]).round(2).tolist()} s")
    if a.solo_medir:
        return
    ruta = Path(a.ficha)
    ficha = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    ficha["pausas"] = perfil
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(ficha, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"guardado en {ruta}")


if __name__ == "__main__":
    main()
