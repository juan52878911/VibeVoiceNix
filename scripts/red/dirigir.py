#!/usr/bin/env python3
"""I2 · Intervencion causal: sumar una direccion (o quitarla) sobre la condicion o el residual de una capa, generando.

    python3 scripts/red/dirigir.py --modelo <ruta> --voces ~/.cache/vibevoice-nix/voces --voz sp-Spk1_man \
        --direcciones red/sondas/direcciones.npz --clave condicion/f0_st --lambdas 0,0.05,0.1,0.2 --rama ambas \
        --corpus scripts/corpus_mejora.json --grupos es --semillas 11,101 --salida red/dirigir

  --clave  <sitio>/<etiqueta> de sondas.py: 'condicion/...' suma en la condicion de la cabeza (dos lineas, cero coste);
           'capaNN/...' suma en el residual de esa capa del tts_lm (entra en la cache de los fotogramas siguientes).
  --rama   pos (solo la condicional: la guia lo amplifica x cfg) | ambas (la guia no lo ve).
  --quitar proyecta la direccion FUERA en vez de sumarla (prueba de causalidad).
  --desde  fotograma a partir del cual se aplica (prueba de tiempo real: 0 = toda la locucion).

Deja un .wav por (frase, semilla, lambda) y medidas.json con los descriptores de perfil_vocal.py por clip. La puerta
(WER, ECAPA, UTMOS) se pasa despues con los jueces de siempre (scripts/juez_lote.py) sobre esa carpeta.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import modelo as MO  # noqa: E402
from bucle import Generador  # noqa: E402
from instrumentar import guardar_wav  # noqa: E402


def enganche_condicion(d, lam, rama, quitar, desde):
    d = d / d.norm()

    def f(c, n, j):
        if j < desde or lam == 0:
            return c, n
        if quitar:
            c2 = c - (c @ d)[:, None] * d
            n2 = n - (n @ d)[:, None] * d if rama == "ambas" else n
            return c2, n2
        e = lam * c.pow(2).mean(-1, keepdim=True).sqrt() * d
        return c + e, (n + e if rama == "ambas" else n)
    return f


class EngancheCapa:
    """Suma la direccion al residual de salida de la capa L, solo en la ultima posicion, en la rama que toque."""

    def __init__(self, gen, capa, d, lam, rama, quitar, desde):
        self.gen, self.d, self.lam, self.rama, self.quitar, self.desde = gen, d / d.norm(), lam, rama, quitar, desde
        self.h = gen.m.model.tts_language_model.layers[capa].register_forward_hook(self)

    def __call__(self, mod, ent, sal):
        g = self.gen
        if g.fotograma < self.desde or self.lam == 0:
            return None
        negativo = g._en_negativo
        if negativo and self.rama != "ambas":
            return None
        h = sal[0] if isinstance(sal, tuple) else sal
        x = h[:, -1]
        if self.quitar:
            x2 = x - (x @ self.d)[:, None] * self.d
        else:
            x2 = x + self.lam * x.pow(2).mean(-1, keepdim=True).sqrt() * self.d
        h = h.clone()
        h[:, -1] = x2
        return (h,) + tuple(sal[1:]) if isinstance(sal, tuple) else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--aleatorio", action="store_true")
    ap.add_argument("--voces", required=True)
    ap.add_argument("--voz", action="append", required=True)
    ap.add_argument("--direcciones", required=True)
    ap.add_argument("--clave", required=True)
    ap.add_argument("--lambdas", default="0,0.05,0.1,0.2")
    ap.add_argument("--rama", choices=["pos", "ambas"], default="ambas")
    ap.add_argument("--quitar", action="store_true")
    ap.add_argument("--desde", type=int, default=0)
    ap.add_argument("--corpus", default=str(AQUI.parent / "corpus_mejora.json"))
    ap.add_argument("--grupos", default="es")
    ap.add_argument("--semillas", default="11,101")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cfg", type=float, default=3.0)
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    m, tok = MO.cargar(a.modelo, aleatorio=a.aleatorio)
    d = torch.tensor(np.load(a.direcciones)[a.clave], dtype=torch.float32)
    sitio = a.clave.split("/")[0]
    corpus = json.loads(Path(a.corpus).read_text())
    medidas = {}
    try:
        import perfil_vocal as PV
        import soundfile as sf
    except Exception:
        PV = None
    for voz in a.voz:
        base = MO.prefijo(Path(a.voces) / f"{voz}.pt")
        for g in a.grupos.split(","):
            for k, texto in enumerate(corpus[g]):
                for s in (int(x) for x in a.semillas.split(",")):
                    for lam in (float(x) for x in a.lambdas.split(",")):
                        nombre = f"{voz}__{g}{k}__s{s}__l{lam:g}{'q' if a.quitar else ''}"
                        if (sal / f"{nombre}.wav").exists():
                            continue
                        torch.manual_seed(s)
                        gen = Generador(m, base, MO.fichas(texto, tok), cfg_scale=a.cfg,
                                        al_condicion=enganche_condicion(d, lam, a.rama, a.quitar, a.desde) if sitio == "condicion" else None)
                        eng = EngancheCapa(gen, int(sitio[4:]), d, lam, a.rama, a.quitar, a.desde) if sitio.startswith("capa") else None
                        onda = gen.correr()
                        if eng:
                            eng.h.remove()
                        guardar_wav(sal / f"{nombre}.wav", onda)
                        if PV is not None:
                            x, hz = sf.read(str(sal / f"{nombre}.wav"), dtype="float32")
                            medidas[nombre] = PV.perfil(x, hz, texto)
                        print(nombre, f"{onda.shape[0] / 24000:.1f} s", flush=True)
    json.dump(medidas, open(sal / "medidas.json", "w"), indent=1)


if __name__ == "__main__":
    main()
