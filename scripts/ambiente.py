#!/usr/bin/env python3
"""Ambientes de fondo sintetizados con codigo (sin muestras ni licencias de terceros), para mezclar bajo
la voz: teclado, lluvia, oficina (murmullo) y cafeteria.

Por que asi y no desde el modelo: el plan de mejora (docs/plan-mejora-modelo.md, F6) midio que pedirle
ambiente al TTS cuesta lo mismo por fotograma que el habla, no se controla y ensucia la voz; y F5
(docs/bancos/2026-09-23-f2-f5-texto-y-no-verbales.md) que el modelo lee las marcas en vez de
interpretarlas. Un fondo procedural cuesta milisegundos por minuto y se controla del todo.

El murmullo de oficina y cafeteria se hace con voces de serie del propio VibeVoice, dadas la vuelta y
filtradas: suena a gente hablando lejos y no se entiende ninguna palabra.

  python3 ambiente.py teclado 30 fondo.wav            # 30 s de teclado
  python3 ambiente.py mezclar voz.wav oficina salida.wav --nivel -22    # dB bajo la voz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 24000


def _filtro(x, bajo=None, alto=None, orden=2):
    if bajo and alto:
        b, a = signal.butter(orden, [bajo / (SR / 2), alto / (SR / 2)], "band")
    elif bajo:
        b, a = signal.butter(orden, bajo / (SR / 2), "high")
    else:
        b, a = signal.butter(orden, alto / (SR / 2), "low")
    return signal.lfilter(b, a, x)


def rosa(n, rng):
    """Ruido rosa (1/f) por filtrado del blanco."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, rng.normal(0, 1, n))


def teclado(dur, rng):
    n = int(dur * SR)
    x = np.zeros(n)
    t = 0.3
    while t < dur - 0.2:
        # rafagas de tecleo: 4-9 teclas por segundo, con pausas de pensar
        if rng.random() < 0.12:
            t += rng.uniform(0.6, 1.8)
            continue
        largo = int(rng.uniform(0.012, 0.03) * SR)
        clic = rng.normal(0, 1, largo) * np.exp(-np.arange(largo) / (0.004 * SR))
        clic = _filtro(clic, 1800, 7000) * rng.uniform(0.5, 1.0)
        if rng.random() < 0.08:        # barra espaciadora: mas grave y mas larga
            largo = int(0.05 * SR)
            clic = _filtro(rng.normal(0, 1, largo) * np.exp(-np.arange(largo) / (0.012 * SR)), 300, 2500) * 0.9
        i = int(t * SR)
        x[i:i + len(clic)] += clic[:n - i]
        t += rng.uniform(0.09, 0.24)
    return x + 0.02 * _filtro(rosa(n, rng), None, 400)       # algo de sala


def lluvia(dur, rng):
    n = int(dur * SR)
    x = _filtro(rng.normal(0, 1, n), 800, 9000) * 0.25 + _filtro(rosa(n, rng), None, 1200) * 0.15
    for _ in range(int(dur * 60)):                       # gotas
        i = rng.integers(0, n - 800)
        d = rng.normal(0, 1, 600) * np.exp(-np.arange(600) / 90)
        x[i:i + 600] += _filtro(d, 2000, 8000) * rng.uniform(0.2, 0.6)
    return x


def murmullo(dur, rng, voces, n_voces=6):
    """Varias voces de serie, dadas la vuelta, desplazadas y filtradas: habla lejana e ininteligible."""
    n = int(dur * SR)
    x = np.zeros(n)
    for k in range(n_voces):
        v = voces[k % len(voces)][::-1]
        if len(v) < n:
            v = np.tile(v, n // len(v) + 1)
        ini = rng.integers(0, len(v) - n + 1)
        x += np.roll(v[ini:ini + n], rng.integers(0, n)) * rng.uniform(0.5, 1.0)
    return _filtro(x, 250, 3000) + 0.05 * _filtro(rosa(n, rng), None, 300)


def cafeteria(dur, rng, voces):
    n = int(dur * SR)
    x = murmullo(dur, rng, voces, 10)
    for _ in range(int(dur * 0.8)):                      # tazas y cucharillas
        i = rng.integers(0, n - 2400)
        f = rng.uniform(2500, 4500)
        tt = np.arange(2400) / SR
        x[i:i + 2400] += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.03) * rng.uniform(0.3, 0.8) * np.abs(x).max()
    return x


def generar(tipo, dur, semilla=0, voces_dir=None):
    rng = np.random.default_rng(semilla)
    voces = []
    if tipo in ("oficina", "cafeteria"):
        for w in sorted(Path(voces_dir).glob("en-*.wav"))[:12]:     # solo voces de serie, nunca personas
            v, hz = sf.read(str(w), dtype="float32")
            voces.append(v)
    x = {"teclado": lambda: teclado(dur, rng), "lluvia": lambda: lluvia(dur, rng),
         "oficina": lambda: murmullo(dur, rng, voces) + 0.3 * teclado(dur, rng),
         "cafeteria": lambda: cafeteria(dur, rng, voces)}[tipo]()
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)      # RMS 1: el nivel lo pone la mezcla


def mezclar(voz, fondo, nivel_db):
    """El fondo queda `nivel_db` por debajo del nivel de la voz cuando habla (RMS de lo activo)."""
    activa = voz[np.abs(voz) > 0.02]
    rms_voz = np.sqrt(np.mean(activa ** 2)) if len(activa) else 0.05
    fondo = np.resize(fondo, len(voz)) * rms_voz * 10 ** (nivel_db / 20)
    y = voz + fondo
    return (y / max(1.0, np.abs(y).max() / 0.98)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c", nargs="?")
    ap.add_argument("--nivel", type=float, default=-22.0)
    ap.add_argument("--voces", default=str(Path.home() / "Documents/mejora-modelo/f4b/wav"))
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    if a.accion == "mezclar":
        voz, hz = sf.read(a.a, dtype="float32")
        fondo = generar(a.b, len(voz) / hz + 1, a.semilla, a.voces)
        sf.write(a.c, mezclar(voz, fondo, a.nivel), hz, subtype="PCM_16")
    else:
        sf.write(a.b, generar(a.accion, float(a.a), a.semilla, a.voces) * 0.1, SR, subtype="PCM_16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
