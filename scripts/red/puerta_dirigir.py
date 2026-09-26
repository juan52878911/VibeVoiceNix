#!/usr/bin/env python3
"""I2 · La puerta de un barrido de dirigir.py, pareada por (voz, frase, semilla) contra lambda = 0.

    python3 scripts/red/puerta_dirigir.py lote <carpeta de dirigir o de instrumentar> [--corpus a.json,b.json]
        -> <carpeta>/lote.json para scripts/juez_lote.py (clave = nombre del wav; identidad = la voz)
    python3 scripts/red/puerta_dirigir.py puerta <carpeta de dirigir> --descriptor hz [--signo +1]
        -> lee <carpeta>/medidas.json (perfil_vocal, de dirigir.py) y <carpeta>/juez.json (juez_lote.py)

Puerta (plan de la red §4.3, fijada antes de medir), por voz y por lambda, todo pareado contra lambda = 0:
  - el descriptor objetivo se mueve en la direccion pedida con IC 95 % > 0 y monotono en lambda
    (hz se compara en semitonos: 12·log2(hz/hz0); los demas en su unidad)
  - ECAPA contra el audio real >= -0,005 (media de la diferencia)
  - UTMOS: IC inferior de la diferencia >= -0,02
  - WER <= +0,5 puntos (media de la diferencia, en puntos porcentuales)
  - ningun clip con WER > 25 % si su base (lambda 0) tenia <= 10 %
Pasa la lambda MAXIMA que cumple todo. Se imprime la tabla y se guarda <carpeta>/puerta.json.
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

NOMBRE = re.compile(r"^(?P<voz>.+?)__(?P<grupo>[a-z_]+?)(?P<k>\d+)__s(?P<s>\d+)(?:__l(?P<lam>[0-9.eE+-]+)(?P<q>q?))?$")
IDIOMA = {"es": "es", "es_numeros": "es", "en": "en", "en_numeros": "en", "no_verbales": "es", "largas": "es", "cortas": "es"}


def partes(nombre):
    m = NOMBRE.match(nombre)
    if not m:
        return None
    d = m.groupdict()
    d["k"], d["s"], d["lam"] = int(d["k"]), int(d["s"]), float(d["lam"]) if d["lam"] is not None else None
    return d


def ic(v, n=4000):
    if not v:
        return (float("nan"),) * 3
    r = random.Random(0)
    b = sorted(sum(r.choices(v, k=len(v))) / len(v) for _ in range(n))
    return sum(v) / len(v), b[int(0.025 * n)], b[min(n - 1, int(0.975 * n))]


def lote(carpeta, corpus):
    todos = {}
    for c in corpus.split(","):
        todos.update(json.loads(Path(c).read_text()))
    corpus = todos
    sal = []
    for w in sorted(Path(carpeta).glob("*.wav")):
        p = partes(w.stem)
        if not p:
            continue
        sal.append(dict(clave=w.stem, audio=str(w), texto=corpus[p["grupo"]][p["k"]], idioma=IDIOMA.get(p["grupo"], "es"),
                        identidad=p["voz"]))
    Path(carpeta, "lote.json").write_text(json.dumps(sal, ensure_ascii=False, indent=1))
    print(f"{len(sal)} clips en {carpeta}/lote.json")


def puerta(carpeta, descriptor, signo, umbral_ecapa, umbral_utmos, umbral_wer):
    carpeta = Path(carpeta)
    perfil = json.loads((carpeta / "medidas.json").read_text())
    juez = json.loads((carpeta / "juez.json").read_text()) if (carpeta / "juez.json").exists() else {}
    por = defaultdict(dict)                    # (voz, grupo, k, s) -> lam -> registro
    for nombre, m in perfil.items():
        p = partes(nombre)
        if not p or p["lam"] is None:
            continue
        j = juez.get(nombre, {})
        por[(p["voz"], p["grupo"], p["k"], p["s"])][p["lam"]] = dict(desc=m.get(descriptor), ecapa=j.get("ecapa"),
                                                                    utmos=j.get("utmos"), wer=j.get("wer_norm", j.get("wer")))
    lambdas = sorted({lam for v in por.values() for lam in v} - {0.0})
    voces = sorted({k[0] for k in por})
    res = {}
    print(f"descriptor {descriptor} (signo pedido {signo:+d}); pareado contra lambda 0; n = parejas")
    print(f"{'voz':14s} {'lambda':>6s} {'n':>3s} | {'desc dif (IC 95 %)':>28s} | {'ECAPA dif':>9s} | {'UTMOS dif (IC inf)':>20s} | "
          f"{'WER dif pts':>11s} | {'rotos':>5s} | veredicto")
    for voz in voces + ["TODAS"]:
        filas = {}
        for lam in lambdas:
            dd, de, du, dw, rotos, n_wer = [], [], [], [], 0, 0
            for clave, v in por.items():
                if voz != "TODAS" and clave[0] != voz:
                    continue
                b, x = v.get(0.0), v.get(lam)
                if not b or not x:
                    continue
                if b["desc"] is not None and x["desc"] is not None:
                    if descriptor == "hz":
                        if b["desc"] > 0 and x["desc"] > 0:
                            dd.append(12 * math.log2(x["desc"] / b["desc"]))
                    else:
                        dd.append(x["desc"] - b["desc"])
                if b["ecapa"] is not None and x["ecapa"] is not None:
                    de.append(x["ecapa"] - b["ecapa"])
                if b["utmos"] is not None and x["utmos"] is not None:
                    du.append(x["utmos"] - b["utmos"])
                if b["wer"] is not None and x["wer"] is not None:
                    dw.append(100 * (x["wer"] - b["wer"]))
                    n_wer += 1
                    if b["wer"] <= 0.10 and x["wer"] > 0.25:
                        rotos += 1
            if not dd:
                continue
            m_d, lo_d, hi_d = ic([signo * x for x in dd])
            m_e = ic(de)[0] if de else None
            m_u, lo_u, _ = ic(du) if du else (None, None, None)
            m_w = ic(dw)[0] if dw else None
            filas[lam] = dict(n=len(dd), desc=(m_d, lo_d, hi_d), ecapa=m_e, utmos=(m_u, lo_u), wer=m_w, rotos=rotos,
                              mueve=lo_d > 0,
                              ecapa_ok=m_e is None or m_e >= umbral_ecapa,
                              utmos_ok=lo_u is None or lo_u >= umbral_utmos,
                              wer_ok=m_w is None or m_w <= umbral_wer,
                              rotos_ok=rotos == 0)
        # monotono: la media del descriptor no baja al subir lambda (con el signo pedido)
        medias = [filas[l]["desc"][0] for l in sorted(filas)]
        monotono = all(b >= a - 1e-9 for a, b in zip(medias, medias[1:]))
        pasa_max = None
        for lam in sorted(filas):
            f = filas[lam]
            f["monotono"] = monotono
            f["pasa"] = f["mueve"] and monotono and f["ecapa_ok"] and f["utmos_ok"] and f["wer_ok"] and f["rotos_ok"]
            if f["pasa"]:
                pasa_max = lam
            fallos = [k for k in ("mueve", "monotono", "ecapa_ok", "utmos_ok", "wer_ok", "rotos_ok") if not f[k]]
            ecapa = "-" if f["ecapa"] is None else f"{f['ecapa']:+.4f}"
            utmos = "-" if f["utmos"][0] is None else f"{f['utmos'][0]:+.3f} ({f['utmos'][1]:+.3f})"
            wer = "-" if f["wer"] is None else f"{f['wer']:+.2f}"
            d = f["desc"]
            print(f"{voz:14s} {lam:6g} {f['n']:3d} | {d[0]:+7.3f} [{d[1]:+7.3f},{d[2]:+7.3f}] | {ecapa:>9s} | {utmos:>20s} | "
                  f"{wer:>11s} | {f['rotos']:5d} | {'PASA' if f['pasa'] else 'no: ' + ','.join(fallos)}")
        res[voz] = dict(filas={str(l): f for l, f in filas.items()}, pasa_lambda_max=pasa_max, monotono=monotono)
        print(f"{voz:14s} -> lambda maxima que pasa: {pasa_max}")
    (carpeta / "puerta.json").write_text(json.dumps(dict(descriptor=descriptor, signo=signo, umbrales=dict(
        ecapa=umbral_ecapa, utmos_ic_inf=umbral_utmos, wer_pts=umbral_wer), por_voz=res), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["lote", "puerta"])
    ap.add_argument("carpeta")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent / "corpus_mejora.json"),
                    help="uno o varios json separados por comas (corpus_mejora.json,red/corpus_duraciones.json)")
    ap.add_argument("--descriptor", default="hz", help="clave de perfil_vocal.py: hz, rango_db, silabas_s, pausas_min...")
    ap.add_argument("--signo", type=int, default=1, help="+1 si lambda > 0 debe SUBIR el descriptor, -1 si debe bajarlo")
    ap.add_argument("--ecapa", type=float, default=-0.005)
    ap.add_argument("--utmos", type=float, default=-0.02)
    ap.add_argument("--wer", type=float, default=0.5)
    a = ap.parse_args()
    if a.modo == "lote":
        lote(a.carpeta, a.corpus)
    else:
        puerta(a.carpeta, a.descriptor, a.signo, a.ecapa, a.utmos, a.wer)


if __name__ == "__main__":
    main()
