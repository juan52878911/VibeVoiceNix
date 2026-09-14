#!/usr/bin/env python
"""Elige la mejor referencia de clonado de una persona dentro de TODO su audio de un vídeo anotado.

    pkgs/vibevoice/.venv/bin/python scripts/mejor_referencia.py --pista voces24k.wav --anotacion anotacion.json \
        --hablante 0 --salida trabajo/sebastian --duraciones 30 60

POR QUÉ
El banco de dobla guarda ~30 s por persona elegidos por SNR y duración. Con una charla larga hay decenas de
minutos de la misma voz, y lo que manda en un clon no es la cantidad sino que la referencia sea CONSISTENTE
consigo misma (docs/clonado-de-voz.md; memoria techo-de-voz: un solape destruye el techo, 0,019 -> 0,890 al
quitarlo; tres tramos consistentes dieron 0,490 frente a 0,443 del banco en una voz difícil). Esto aprovecha
todo el audio: huella ECAPA de cada segmento puro, centroide de la persona, y referencias de la duración pedida
con los segmentos más parecidos a ese centroide.

QUÉ DEJA en --salida:
  apartados.json          segmentos que NO pueden entrar en ninguna referencia: son la vara para evaluar el
                          clon contra audio real que no ha oído (una fracción repartida por toda la charla)
  seleccion-<D>s.json     los segmentos elegidos para ~D segundos, con su consistencia y el techo estimado
  seleccion-<D>s/ref-XX.wav + lote.json   listos para clonar_voz.py --lote (audio + transcripción literal)
  resumen.json            pausas reales, consistencia por segmento, techo de cada selección

EL TECHO que se imprime es el de la selección partida en dos mitades por clips alternos (coseno entre las
huellas medias de cada mitad). Es la misma idea que scripts/techo.py pero sobre segmentos ya puros.

Datos biométricos: la salida es la voz de una persona. Solo para identidades con consentimiento registrado
(dobla, voces/identidades.json) y fuera del repo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent / "pkgs" / "vibevoice-cli"))
import pausas as PZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pista", required=True)
    ap.add_argument("--anotacion", required=True)
    ap.add_argument("--hablante", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--duraciones", nargs="+", type=float, default=[30.0, 60.0])
    ap.add_argument("--min-seg", type=float, default=3.0)
    ap.add_argument("--max-seg", type=float, default=15.0, help="los segmentos más largos se parten en trozos de este tope")
    ap.add_argument("--pureza", type=float, default=0.3)
    ap.add_argument("--apartar", type=float, default=0.25, help="fracción de segmentos para evaluar, nunca referencia")
    ap.add_argument("--criterio", choices=("consistencia", "limpieza"), default="consistencia",
                    help="limpieza: por relación señal/ruido estimada (p90-p10 de la energía en ventanas de 10 ms), "
                         "entre los segmentos con consistencia >= la mediana. MEDIDO con Sebastián: la consistencia "
                         "sola (techo 0,89-0,90) dio peor clon que el banco de dobla, que elige por SNR")
    a = ap.parse_args()
    import librosa
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    anot = json.load(open(a.anotacion, encoding="utf-8"))
    segs = anot["segmentos"]
    pista, sr = sf.read(a.pista, dtype="float32")
    assert sr == PZ.RITMO, f"la pista tiene que ir a {PZ.RITMO} Hz"

    # ---------------------------------------------------------- segmentos puros
    propios = [s for s in segs if str(s["hablante"]) == str(a.hablante)]
    ajenos = [s for s in segs if str(s["hablante"]) != str(a.hablante)]
    puros = []
    for s in propios:
        if s["fin"] - s["ini"] < a.min_seg:
            continue
        pisado = sum(max(0.0, min(s["fin"], o["fin"]) - max(s["ini"], o["ini"])) for o in ajenos)
        if pisado > a.pureza:
            continue
        # los largos se parten: una referencia de 40 s de un solo segmento no deja elegir
        ini, fin = s["ini"], s["fin"]
        n = max(1, int(np.ceil((fin - ini) / a.max_seg)))
        paso = (fin - ini) / n
        for k in range(n):
            puros.append({"ini": round(ini + k * paso, 3), "fin": round(ini + (k + 1) * paso, 3),
                          "texto": s.get("texto", "") if n == 1 else "", "entero": n == 1})
    puros.sort(key=lambda s: s["ini"])
    print(f"hablante {a.hablante}: {len(propios)} segmentos, {len(puros)} tramos puros de >= {a.min_seg} s", flush=True)

    # apartados repartidos por toda la charla (uno de cada 1/apartar), nunca referencia
    cada = max(2, int(round(1 / a.apartar)))
    apartados = [s for i, s in enumerate(puros) if i % cada == cada // 2]
    pool = [s for i, s in enumerate(puros) if i % cada != cada // 2]
    json.dump({"hablante": a.hablante, "segmentos": apartados}, open(sal / "apartados.json", "w"),
              ensure_ascii=False, indent=1)

    # ---------------------------------------------------------- huellas ECAPA
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"),
                                           run_opts={"device": "cpu"})

    def huella(x):
        x16 = librosa.resample(x, orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(x16).float()[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)

    def trozo(s):
        return pista[int(s["ini"] * sr):int(s["fin"] * sr)]

    for s in puros:
        s["huella"] = huella(trozo(s))
    centro = np.mean([s["huella"] for s in pool], 0)
    centro /= np.linalg.norm(centro)
    for s in puros:
        s["consistencia"] = float(np.dot(s["huella"], centro))
        x = trozo(s)
        n = len(x) // PZ.VENTANA
        rms = np.sqrt(np.mean(x[:n * PZ.VENTANA].reshape(n, PZ.VENTANA).astype(np.float64) ** 2, 1)) + 1e-7
        db = 20 * np.log10(rms)
        s["snr_db"] = float(np.percentile(db, 90) - np.percentile(db, 10))
    cons_pool = np.array([s["consistencia"] for s in pool])
    print(f"consistencia con el centroide (pool): p10 {np.percentile(cons_pool, 10):.3f} · "
          f"mediana {np.median(cons_pool):.3f} · p90 {np.percentile(cons_pool, 90):.3f}", flush=True)

    def techo(sel):
        if len(sel) < 2:
            return None
        m1 = np.mean([s["huella"] for s in sel[0::2]], 0)
        m2 = np.mean([s["huella"] for s in sel[1::2]], 0)
        return float(np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2)))

    # ---------------------------------------------------------- selecciones
    resumen = {"hablante": a.hablante, "tramos_puros": len(puros), "apartados": len(apartados),
               "consistencia_pool": {"p10": float(np.percentile(cons_pool, 10)), "mediana": float(np.median(cons_pool)),
                                     "p90": float(np.percentile(cons_pool, 90))},
               "pausas": {k: v for k, v in PZ.perfil([trozo(s) for s in pool]).items() if k != "dist"},
               "selecciones": {}}
    enteros = [s for s in pool if s["entero"] and s["texto"].strip()]
    if a.criterio == "limpieza":
        corte = float(np.median([s["consistencia"] for s in enteros]))
        candidatos = sorted([s for s in enteros if s["consistencia"] >= corte], key=lambda s: -s["snr_db"])
    else:
        candidatos = sorted(enteros, key=lambda s: -s["consistencia"])
    sufijo = "" if a.criterio == "consistencia" else "-limpia"
    for D in a.duraciones:
        sel, acum = [], 0.0
        for s in candidatos:
            if acum >= D:
                break
            sel.append(s)
            acum += s["fin"] - s["ini"]
        sel.sort(key=lambda s: s["ini"])
        carpeta = sal / f"seleccion{sufijo}-{int(D)}s"
        carpeta.mkdir(exist_ok=True)
        refs = []
        for i, s in enumerate(sel):
            ruta = carpeta / f"ref-{i:02d}.wav"
            sf.write(str(ruta), trozo(s), sr, subtype="PCM_16")
            refs.append({"audio": str(ruta), "transcripcion": s["texto"].strip()})
        json.dump(refs, open(carpeta / "refs.json", "w"), ensure_ascii=False, indent=1)
        t = techo(sel)
        info = {"segundos": acum, "clips": len(sel), "techo_mitades": t,
                "consistencia_media": float(np.mean([s["consistencia"] for s in sel])),
                "snr_db_medio": float(np.mean([s["snr_db"] for s in sel])),
                "segmentos": [{k: s[k] for k in ("ini", "fin", "texto", "consistencia", "snr_db")} for s in sel]}
        json.dump(info, open(sal / f"seleccion{sufijo}-{int(D)}s.json", "w"), ensure_ascii=False, indent=1)
        resumen["selecciones"][f"{a.criterio}-{int(D)}s"] = {k: v for k, v in info.items() if k != "segmentos"}
        print(f"  seleccion {a.criterio} {int(D)} s: {len(sel)} clips, {acum:.1f} s, consistencia media "
              f"{info['consistencia_media']:.3f}, SNR {info['snr_db_medio']:.1f} dB, "
              f"techo por mitades {t if t is None else round(t, 3)}", flush=True)
    json.dump(resumen, open(sal / f"resumen{sufijo}.json", "w"), ensure_ascii=False, indent=1)
    print(f"pausas reales: {resumen['pausas']}")


if __name__ == "__main__":
    main()
