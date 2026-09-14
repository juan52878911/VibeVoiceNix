#!/usr/bin/env python
"""Banco bit a bit: 8 frases con semilla fija por /tts/stream, md5 del audio, RTF y /crono.

    VOZ_TOKEN=... python scripts/banco_md5.py --url http://127.0.0.1:8082 --etiqueta base --salida b.json
    python scripts/banco_md5.py comparar base1.json variante1.json base2.json variante2.json

Es la puerta de los cambios que no deben tocar un bit del audio (fase 1 del plan de rendimiento,
docs/plan-rendimiento.md): mismo md5 frase a frase y ronda a ronda. El RTF de cada ronda es el reloj
de pared de las 8 peticiones entre la duracion del audio; la ronda 0 de cada proceso NO cuenta (paginas
frias y asignaciones unicas). Solo usa la biblioteca estandar: corre con cualquier python de la VM.
Con --pid lee VmRSS y VmHWM del proceso servidor al acabar.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

FRASES = [
    "El backup de anoche terminó sin errores y los tres servicios responden con normalidad.",
    "¿Has visto el correo de esta mañana? Dice que la reunión pasa al jueves a las once.",
    "¡No me lo puedo creer! Eso sí que no me lo esperaba para nada.",
    "La factura número cuatro mil trescientos veinte vence el quince de octubre.",
    "Vale, ahora mismo lo miro.",
    "Hecho.",
    "La temperatura del servidor ha subido a ochenta y ocho grados durante el banco de pruebas.",
    "El problema no es el precio, es que nadie te explica lo que estás comprando. Te enseñan una "
    "tabla con veinte filas, te sonríen, y cuando preguntas por la letra pequeña te dicen que eso ya "
    "lo veremos más adelante.",
]
CABECERA_WAV = 44
RITMO = 24000


def pedir(url, token, cuerpo=None, tiempo=900):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    r = urllib.request.Request(url, data=datos, method="POST" if datos else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(r, timeout=tiempo) as resp:
        return resp.read()


def memoria(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            c = dict(linea.split(":", 1) for linea in f if linea.startswith(("VmRSS", "VmHWM")))
        return {k: int(v.split()[0]) // 1024 for k, v in c.items()}
    except (OSError, KeyError, ValueError):
        return {}


def correr(a):
    token = os.environ.get("VOZ_TOKEN", "")
    pedir(f"{a.url}/crono?reset=true", token)
    rondas = []
    for ronda in range(a.rondas):
        clips, pared, audio = [], 0.0, 0.0
        for i, frase in enumerate(FRASES):
            cuerpo = {"texto": frase, "semilla": a.semilla}
            if a.voz:
                cuerpo["voz"] = a.voz
            t = time.perf_counter()
            wav = pedir(f"{a.url}/tts/stream", token, cuerpo)
            dt = time.perf_counter() - t
            dur = (len(wav) - CABECERA_WAV) / 2 / RITMO
            pared += dt
            audio += dur
            clips.append({"frase": i, "md5": hashlib.md5(wav).hexdigest(), "audio_s": round(dur, 3),
                          "pared_s": round(dt, 3)})
            if a.wav:
                os.makedirs(a.wav, exist_ok=True)
                with open(os.path.join(a.wav, f"{a.etiqueta}_r{ronda}_f{i}.wav"), "wb") as f:
                    f.write(wav)
        crono = json.loads(pedir(f"{a.url}/crono?reset=true", token))
        r = {"ronda": ronda, "rtf": round(pared / audio, 4), "audio_s": round(audio, 2), "clips": clips,
             "reparto": crono.get("reparto_ms_por_fotograma")}
        rondas.append(r)
        print(f"[{a.etiqueta}] ronda {ronda}: RTF {r['rtf']:.4f} ({audio:.1f} s de audio)  "
              f"reparto {json.dumps(r['reparto'])}", flush=True)
    salida = {"etiqueta": a.etiqueta, "semilla": a.semilla, "url": a.url, "rondas": rondas,
              "memoria": memoria(a.pid) if a.pid else {}}
    if a.salida:
        json.dump(salida, open(a.salida, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"etiqueta": a.etiqueta, "memoria": salida["memoria"],
                      "rtf_validas": [r["rtf"] for r in rondas[1:]]}), flush=True)


def comparar(rutas):
    bancos = [json.load(open(r)) for r in rutas]
    ref = bancos[0]["rondas"][0]["clips"]
    todo_igual = True
    for b in bancos:
        for r in b["rondas"]:
            iguales = sum(c["md5"] == ref[c["frase"]]["md5"] for c in r["clips"])
            todo_igual &= iguales == len(ref)
            print(f"{b['etiqueta']:<12} ronda {r['ronda']}: {iguales}/{len(ref)} md5 como la primera; "
                  f"RTF {r['rtf']:.4f}; memoria {b.get('memoria')}")
    por_etiqueta = {}
    for b in bancos:
        por_etiqueta.setdefault(b["etiqueta"].split("-")[0], []).extend(r["rtf"] for r in b["rondas"][1:])
    for e, v in por_etiqueta.items():
        if not v:
            print(f"{e:<12} sin rondas validas (solo la ronda 0): no hay RTF que comparar")
            continue
        print(f"{e:<12} RTF valido (sin ronda 0): mediana {statistics.median(v):.4f}  n={len(v)}  {v}")
    print("MD5 IDENTICO EN TODO" if todo_igual else "HAY MD5 DISTINTOS")
    return 0 if todo_igual else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "comparar":
        sys.exit(comparar(sys.argv[2:]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--etiqueta", required=True)
    ap.add_argument("--rondas", type=int, default=3)
    ap.add_argument("--semilla", type=int, default=101)
    ap.add_argument("--voz")
    ap.add_argument("--salida")
    ap.add_argument("--wav", help="carpeta donde dejar los WAV")
    ap.add_argument("--pid", type=int)
    correr(ap.parse_args())


if __name__ == "__main__":
    main()
