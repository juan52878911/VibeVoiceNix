#!/usr/bin/env python3
"""F3 del plan de mejora: acento sin tocar pesos, solo con el PREFIJO del clon.

Base: la referencia espontanea de F1 (la mejor medida en identidad). Variantes:

  puro         la base tal cual (control)
  mix30_10     la base entera + ~10 s de una voz nativa inglesa (del mismo sexo) con su texto
  mix20_20     ~20 s de la base + ~20 s nativos
  mix10_30     ~10 s de la base + ~30 s nativos
  mix20_20inv  lo mismo que mix20_20 con lo nativo DELANTE (¿importa el orden?)
  traducido    B3: los mismos audios de la base con su transcripcion traducida al ingles

Si el prefijo manda la fonetica, mezclar latentes nativos deberia bajar el PER en ingles; la
pregunta es cuanta identidad cuesta. Clona todo con UNA carga del modelo.

  python3 f3_acento.py --nativos ~/Documents/mejora-modelo/f4b/wav
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

MOTOR = Path.home() / "Documents/GitHub/VibeVoiceNix"
NATIVA = {"carlos-segura": "en-Carter_man", "liliana-morales": "en-Emma_woman"}


def dur(r):
    return sf.info(r["audio"]).duration


def hasta(refs, segundos):
    sel, acum = [], 0.0
    for r in refs:
        if acum >= segundos - 1.0:
            break
        sel.append(r)
        acum += dur(r)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nativos", required=True, help="carpeta con <voz>__harvardNN.wav")
    ap.add_argument("--f1", default=str(Path.home() / "Documents/mejora-modelo/f1"))
    ap.add_argument("--trabajo", default=str(Path.home() / "Documents/mejora-modelo/f3"))
    a = ap.parse_args()
    trabajo = Path(a.trabajo)
    f1 = json.loads((Path(a.f1) / "lote.json").read_text())
    trad = json.loads((trabajo / "traducciones.json").read_text())
    harvard = {p["clave"]: p["texto"] for p in
               json.loads((Path(a.nativos).parent / "peticiones.json").read_text())}
    lote, variantes = [], {}
    for ident, nativa in NATIVA.items():
        base = next(e["refs"] for e in f1 if e["salida"].endswith(f"f1-{ident}-espontanea30.pt"))
        nat = [{"audio": str(Path(a.nativos) / f"{k}.wav"), "transcripcion": t}
               for k, t in sorted(harvard.items()) if k.startswith(f"{nativa}__")]
        v = {
            "puro": base,
            "mix30_10": base + hasta(nat, 10),
            "mix20_20": hasta(base, 20) + hasta(nat, 20),
            "mix10_30": hasta(base, 10) + hasta(nat, 30),
            "mix20_20inv": hasta(nat, 20) + hasta(base, 20),
            "traducido": [{"audio": r["audio"], "transcripcion": t} for r, t in zip(base, trad[ident])],
        }
        variantes[ident] = {}
        for nombre, refs in v.items():
            voz = f"f3-{ident}-{nombre}"
            propios = sum(dur(r) for r in refs if "/f1/" in r["audio"])
            ajenos = sum(dur(r) for r in refs if "/f4b/" in r["audio"])
            variantes[ident][nombre] = {"voz": voz, "segundos_persona": round(propios, 1),
                                        "segundos_nativa": round(ajenos, 1)}
            lote.append({"salida": str(trabajo / "clones" / f"{voz}.pt"), "semilla": 11, "refs": refs})
            print(f"[f3] {voz:38s} persona {propios:5.1f} s · nativa {ajenos:5.1f} s", flush=True)
    (trabajo / "clones").mkdir(parents=True, exist_ok=True)
    (trabajo / "variantes.json").write_text(json.dumps(variantes, ensure_ascii=False, indent=1))
    (trabajo / "lote.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    subprocess.run([sys.executable, str(MOTOR / "scripts/clonar_voz.py"), "--lote", str(trabajo / "lote.json")],
                   check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
