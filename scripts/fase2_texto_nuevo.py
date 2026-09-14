#!/usr/bin/env python
"""¿El juez de la fase 2b sirve con frases que la persona NUNCA dijo?

    python scripts/fase2_texto_nuevo.py --dataset /var/lib/taller/dataset-voces --clones /var/lib/taller/fase1/clon \
        --trabajo /var/lib/taller/fase2b --voces /var/lib/taller/fase1/voces --salida /var/lib/taller/texto_nuevo

Hasta ahora el juez solo vio clones diciendo el MISMO texto que el clip real. La fase 3 generará frases
nuevas, así que hay que saber si el juez juzga la voz o se apoya en el texto. Por persona de la
validación, cuatro grupos:

  real        sus clips apartados (audio real)
  clon        el clon de la fase 1 diciendo esos textos (decir.py, torch)
  mismo       el clon por voz-stream (motor de producción) diciendo esos textos: control del motor
  nuevo       el clon por voz-stream diciendo FRASES_NUEVAS, que no están en su audio

PUERTA (fijada antes de medir): el juez generaliza si AUC(real, nuevo) > 0,7 -- sigue viendo el clon
con texto nuevo -- y AUC(mismo, nuevo) < 0,75 -- el texto no mueve su veredicto --, y la salida de
persona acierta con el texto nuevo por encima del azar.

Usa el servicio en marcha (no hace falta modo taller): copia los .pt de la fase 1 como taller-<persona>
en la carpeta de voces del servicio y los borra al acabar.
"""
import argparse
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fase2_estilo as F2  # noqa: E402

FRASES_NUEVAS = [
    "Oye, ¿te acuerdas del restaurante que abrieron en la esquina? Fui el sábado y la verdad me pareció carísimo para lo que sirven.",
    "Mañana salgo temprano para Medellín, así que si necesitas algo del proyecto, escríbeme antes de las siete.",
    "No sé si llueva esta tarde, pero por si acaso lleva sombrilla, que el cielo está bastante nublado.",
    "Estuve leyendo sobre cómo cuidar las plantas del apartamento y resulta que las estaba regando demasiado.",
    "¿Viste el partido de anoche? Íbamos ganando hasta el minuto ochenta y al final se nos fue de las manos.",
    "Lo que más me gusta de viajar en bus es mirar por la ventana y pensar en todo lo que tengo pendiente.",
]


def palabras(t):
    return set(re.findall(r"\w+", t.lower()))


def pedir_wav(url, token, texto, voz, cfg, semilla, pasos):
    cuerpo = json.dumps({"texto": texto, "voz": voz, "cfg_scale": cfg, "semilla": semilla, "pasos": pasos,
                         "formato": "wav"}).encode()
    pet = urllib.request.Request(f"{url}/tts/stream", data=cuerpo, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(pet, timeout=600) as r:
        datos = r.read()
    i = datos.find(b"data")
    return np.frombuffer(datos[i + 8:], dtype="<i2").astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clones", required=True)
    ap.add_argument("--trabajo", required=True, help="carpeta con estilo.pt, informe y caracteristicas de la fase 2b")
    ap.add_argument("--voces", required=True, help="los .pt de clonar_voz.py de la fase 1")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--token-env", default="/var/lib/voz/token.env")
    ap.add_argument("--voces-servicio", default="/run/voz-stream/voces")
    ap.add_argument("--semilla", type=int, default=101)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--pasos", type=int, default=6)
    a = ap.parse_args()
    import soundfile as sf
    import torch
    torch.set_num_threads(2)  # el servicio sigue sirviendo

    tr_dir, ds, sal = Path(a.trabajo), Path(a.dataset), Path(a.salida)
    inf = json.load(open(tr_dir / "informe_fase2.json"))
    personas, nm = inf["personas"], inf["normalizacion"]
    d = np.load(tr_dir / "caracteristicas.npz", allow_pickle=True)
    C, P, R = d["C"], d["P"], d["R"]
    _, va = F2.particion(C, P, R)
    val = sorted({(str(C[i]), int(P[i])) for i in np.where(va)[0] if R[i] == 1})
    import csv
    filas = list(csv.DictReader(open(ds / "manifiesto.csv", encoding="utf-8")))
    texto_de = {Path(f["fichero"]).stem: f["texto"] for f in filas}
    evaluadas = sorted({personas[p] for _, p in val})
    print(f"personas: {evaluadas} · {len(val)} clips apartados", flush=True)

    print("== novedad de las frases (maximo de palabras compartidas con cualquier clip del dataset)")
    for f in FRASES_NUEVAS:
        w = palabras(f)
        comun = max(len(w & palabras(t)) / len(w | palabras(t)) for t in texto_de.values())
        print(f"   Jaccard {comun:.2f} · {f[:60]}", flush=True)

    token = next(l.split("=", 1)[1].strip() for l in open(a.token_env) if l.startswith("VOZ_TOKEN="))
    copiadas = []
    try:
        for persona in evaluadas:
            destino = Path(a.voces_servicio) / f"taller-{persona}.pt"
            shutil.copyfile(Path(a.voces) / f"{persona}.pt", destino)
            shutil.chown(destino, "voz-stream", "voz-stream")
            copiadas.append(destino)
        print("== generando con voz-stream", flush=True)
        for persona in evaluadas:
            (sal / persona).mkdir(parents=True, exist_ok=True)
            trabajo = [(f"mismo-{c}", texto_de[c]) for c, p in val if personas[p] == persona]
            trabajo += [(f"nuevo-{i}", t) for i, t in enumerate(FRASES_NUEVAS)]
            for nombre, texto in trabajo:
                fichero = sal / persona / f"{nombre}.wav"
                if fichero.exists():
                    continue
                y = pedir_wav(a.url, token, texto, f"taller-{persona}", a.cfg, a.semilla, a.pasos)
                sf.write(str(fichero), y, F2.SR, subtype="PCM_16")
                print(f"   {persona} {nombre} {len(y) / F2.SR:.1f} s", flush=True)
    finally:
        for c in copiadas:
            c.unlink(missing_ok=True)

    modelo = F2.construir_modelo(len(personas), len(F2.DESCRIPTORES))
    modelo.load_state_dict(torch.load(tr_dir / "estilo.pt"))
    modelo.eval()

    def puntuar(fichero):
        x, sr = sf.read(str(fichero), dtype="float32")
        frs = [f for f in F2.fragmentos(x) if np.sqrt(np.mean(f ** 2)) >= 1e-3]
        if not frs:
            return None
        mel = (np.stack([F2.mel(f) for f in frs]) - nm["mel_media"]) / nm["mel_desv"]
        with torch.no_grad():
            _, _, per, real = modelo(torch.from_numpy(mel.astype(np.float32)))
        return {"p_real": float(torch.sigmoid(real).mean()),
                "oida": personas[int(torch.softmax(per, -1).mean(0).argmax())]}

    grupos = {g: [] for g in ("real", "clon", "mismo", "nuevo")}
    for persona in evaluadas:
        rutas = {"real": [ds / persona / f"{c}.wav" for c, p in val if personas[p] == persona],
                 "clon": [Path(a.clones) / persona / f"{c}.wav" for c, p in val if personas[p] == persona],
                 "mismo": sorted((sal / persona).glob("mismo-*.wav")),
                 "nuevo": sorted((sal / persona).glob("nuevo-*.wav"))}
        for g, lista in rutas.items():
            for r in lista:
                s = puntuar(r) if r.exists() else None
                if s:
                    grupos[g].append({"esperada": persona, "fichero": r.name, **s})

    def auc_de(pos, neg):
        p, n = [x["p_real"] for x in grupos[pos]], [x["p_real"] for x in grupos[neg]]
        return F2.auc([1] * len(p) + [0] * len(n), p + n)

    acierto = {g: float(np.mean([x["oida"] == x["esperada"] for x in v])) if v else float("nan")
               for g, v in grupos.items()}
    res = {"auc_real_nuevo": auc_de("real", "nuevo"), "auc_mismo_nuevo": auc_de("mismo", "nuevo"),
           "auc_real_mismo": auc_de("real", "mismo"), "auc_clon_mismo": auc_de("clon", "mismo"),
           "persona_acierto": acierto, "azar_persona": 1 / len(evaluadas), "grupos": grupos}
    res["pasa"] = bool(res["auc_real_nuevo"] > 0.7 and res["auc_mismo_nuevo"] < 0.75
                       and acierto["nuevo"] > res["azar_persona"])
    json.dump(res, open(sal / "informe_texto_nuevo.json", "w"), ensure_ascii=False, indent=1)

    print("\n== P(real) media por grupo y persona")
    for persona in evaluadas:
        fila = []
        for g in grupos:
            v = [x["p_real"] for x in grupos[g] if x["esperada"] == persona]
            fila.append(f"{g} {np.mean(v):.3f} (n={len(v)})" if v else f"{g} -")
        print(f"   {persona:18s} " + " · ".join(fila))
    print("   persona acertada: " + " · ".join(f"{g} {v:.2f}" for g, v in acierto.items())
          + f" (azar {res['azar_persona']:.2f})")
    print(f"   AUC real/nuevo {res['auc_real_nuevo']:.3f} (puerta > 0,7) · mismo/nuevo {res['auc_mismo_nuevo']:.3f} "
          f"(puerta < 0,75) · real/mismo {res['auc_real_mismo']:.3f} · clon/mismo {res['auc_clon_mismo']:.3f}")
    print(f"   {'PUERTA PASA' if res['pasa'] else 'PUERTA NO PASA'} · informe en {sal / 'informe_texto_nuevo.json'}")


if __name__ == "__main__":
    main()
