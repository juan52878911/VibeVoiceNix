#!/usr/bin/env python
"""El techo de una voz: lo maximo que CUALQUIER motor puede clonar de esa grabacion.

    python scripts/techo.py nota.opus                 # techo de una grabacion
    python scripts/techo.py a.wav b.wav c.wav         # techo del conjunto (una voz)
    python scripts/techo.py charla.wav --tramos 25    # el mejor tramo de 25 s

QUE ES
La referencia partida en dos mitades, una contra la otra, con la misma huella
ECAPA que juzga los clones (scripts/oido.py). Si la voz no se parece a si
misma, ningun modelo va a parecerse a ella: es el limite superior, y sin ese
numero no se distingue "el clon es malo" (mas referencia, otra semilla) de
"esta grabacion no se deja clonar" (grabar mejor; no hay motor que lo arregle).

MEDIDO (docs/comparativa-motores.md §7): Laura, techo 0,946, clona a 0,65;
Juan Pablo, techo 0,598 -- por debajo del umbral de "misma persona" (0,626) --
no hubo manera ni limpiando el banco. Los dos motores probados sacan un
46-61 % del techo de cada voz: la grabacion manda sobre el motor. Una voz de
trabajo normal (charla de movil separada por demucs) da 0,764.

COMO SE PARTE
- Un solo fichero: primera mitad contra segunda mitad. Es lo que se hizo en
  todas las medidas del repo, y por eso los numeros son comparables.
- Varios ficheros: se ordenan por duracion y se reparten en dos mitades de
  material (la misma regla que `techo_de_voz()` en dobla), huella media de
  cada mitad. Asi un clip corto y sucio no decide el techo el solo.
- --tramos: ventanas deslizantes sobre el material, techo de cada una,
  ordenadas de mejor a peor. Es la forma barata de ELEGIR referencia: el
  tramo con mas techo, no el mas largo ni el primero.

VEREDICTO
    >= 0,85   buena: el clon puede pasar de 0,60
    0,70-0,85 aceptable: es la zona de las voces de trabajo
    0,626-0,70 floja: el clon va a quedar en la zona gris; mejor regrabar
    < 0,626   no se reconoce a si misma: no hay clon que valga
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RITMO = 24000
UMBRAL_MISMA_PERSONA = 0.626      # el de oido.py / banco_clonado.py
_modelo = None


def huella(x, hz=RITMO):
    """Huella ECAPA normalizada (speechbrain, el mismo modelo que oido.py)."""
    global _modelo
    import torch
    import torchaudio
    if _modelo is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _modelo = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
            run_opts={"device": "cpu"})
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    if hz != 16000:
        t = torchaudio.functional.resample(t, hz, 16000)
    with torch.no_grad():
        e = _modelo.encode_batch(t).squeeze()
    return (e / e.norm()).numpy()


def _coseno_medio(A, B):
    ua, ub = np.mean(A, 0), np.mean(B, 0)
    return float(ua @ ub / ((np.linalg.norm(ua) + 1e-9) * (np.linalg.norm(ub) + 1e-9)))


def techo_de_clips(clips, hz=RITMO, minimo_s=2.0):
    """Techo de UNA voz dada como uno o varios clips. None si no hay material."""
    clips = [c for c in clips if len(c) >= minimo_s * hz]
    if not clips:
        return None
    if len(clips) == 1:
        x = clips[0]
        m = len(x) // 2
        return round(float(huella(x[:m], hz) @ huella(x[m:], hz)), 3)
    hs = [huella(c, hz) for c in clips]
    dur = [len(c) / hz for c in clips]
    orden = sorted(range(len(hs)), key=lambda i: -dur[i])
    mitad, acc, A, B = sum(dur) / 2, 0.0, [], []
    for i in orden:
        (A if acc < mitad else B).append(hs[i])
        acc += dur[i]
    if not A or not B:                 # dos clips: uno contra otro
        A, B = [hs[orden[0]]], [hs[orden[-1]]]
    return round(_coseno_medio(A, B), 3)


def techo_por_tramos(x, ventana_s, salto_s, hz=RITMO):
    """[(ini_s, techo)] de cada ventana, de mejor a peor."""
    v, s = int(ventana_s * hz), int(max(salto_s, 0.5) * hz)
    if len(x) <= v:
        return [(0.0, techo_de_clips([x], hz))]
    filas = []
    for ini in range(0, len(x) - v + 1, s):
        filas.append((ini / hz, techo_de_clips([x[ini:ini + v]], hz)))
    return sorted((f for f in filas if f[1] is not None), key=lambda f: -f[1])


def veredicto(techo):
    if techo is None:
        return "sin material suficiente"
    if techo >= 0.85:
        return "buena"
    if techo >= 0.70:
        return "aceptable"
    if techo >= UMBRAL_MISMA_PERSONA:
        return "floja: el clon quedara en la zona gris; mejor regrabar"
    return "no se reconoce a si misma: no hay clon que valga, hace falta otra grabacion"


def informar(techo, sangria="", segundos=None):
    """Una linea para el humano, la misma en clonar_voz.py y clonar_voz_qwen.py."""
    extra = f" ({segundos:.1f} s)" if segundos else ""
    if techo is None:
        print(f"{sangria}techo: sin material suficiente{extra}")
        return
    print(f"{sangria}techo ECAPA {techo:.3f}{extra}: {veredicto(techo)}")
    if techo < 0.70:
        print(f"{sangria}   por debajo de 0,70 el limite lo pone la grabacion, no el "
              "motor: mas referencia o mas semillas no lo van a subir")


def main():
    from clonar_voz import leer_audio
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--tramos", type=float, default=None,
                    help="segundos de ventana: busca el mejor tramo dentro del material")
    ap.add_argument("--salto", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    clips = [leer_audio(r) for r in args.audio]
    total = sum(len(c) for c in clips) / RITMO
    if args.tramos:
        x = np.concatenate(clips)
        filas = techo_por_tramos(x, args.tramos, args.salto)
        if args.json:
            print(json.dumps([{"ini_s": i, "techo": t} for i, t in filas]))
            return
        print(f"{total:.1f} s; ventanas de {args.tramos:g} s cada {args.salto:g} s, "
              f"de mejor a peor:")
        for ini, t in filas[:10]:
            print(f"  {ini:7.1f} s  techo {t:.3f}  {veredicto(t)}")
        return
    t = techo_de_clips(clips)
    if args.json:
        print(json.dumps({"techo": t, "segundos": round(total, 1), "veredicto": veredicto(t)}))
        return
    informar(t, segundos=total)


if __name__ == "__main__":
    main()
