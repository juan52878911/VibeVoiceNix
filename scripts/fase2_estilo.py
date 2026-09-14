#!/usr/bin/env python
"""Fase 2 del plan de personalidad: un codificador de estilo que oye fragmentos de 2 s.

    python scripts/fase2_estilo.py --dataset /var/lib/taller/dataset-voces --clones /var/lib/taller/fase1/clon \
        --trabajo /var/lib/taller/fase2

Red convolucional pequeña sobre mel (~0,5 M de parámetros) con tres salidas (docs/plan-personalidad-voz.md):

  estilo      8 descriptores de scripts/perfil_vocal.py sobre el fragmento (tono en log, recorrido,
              desviación, movimiento, microvariación, rango de energía, inclinación, armonicidad).
              Pausas y velocidad no: en 2 s no significan nada.
  persona     quién habla, entre las identidades con consentimiento. Solo con audio REAL.
  real/clon   si el fragmento es la persona o su clon de la fase 1 (solo donde hay clones). Es el
              juez de «suena a bot».

LA TRAMPA DEL CANAL. El audio real viene de vídeos separados con demucs; el clon es sintético y
limpio. Una red puede aprender a distinguir el CANAL y no la personalidad. Por eso: aumentos a los
dos lados (ganancia, ruido, inclinación de ecualización) y validación por CLIP entero, nunca por
fragmento — los fragmentos de un mismo clip se parecen demasiado entre sí.

PUERTA (fijada antes de medir), sobre los clips apartados: R² > 0 en la mayoría de descriptores;
persona por clip por encima del azar; real/clon con AUC por clip > 0,7.

Y CON --idavuelta (el control de canal midió que sin ella el juez oía el CÓDEC: AUC 0,996 entre el audio
real y el mismo audio pasado por el códec de VibeVoice), el audio real de ida y vuelta entra etiquetado
como REAL, y la puerta del juez pasa a ser: AUC(ida y vuelta, clon) > 0,7 -- sigue separando con el
mismo códec a los dos lados -- y AUC(real, ida y vuelta) < 0,75 -- ya no se apoya en el códec.

Las características de cada fragmento se calculan una vez y se guardan (caracteristicas.npz): el
perfil vocal lleva HPSS y es lo lento. Relanzar reutiliza la caché.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import perfil_vocal as PV  # noqa: E402

VENTANA_S, SALTO_S = 2.0, 1.0
SR = 24000
N_MELS, HOP, NFFT = 64, 240, 1024
DESCRIPTORES = ["hz", "recorrido", "desviacion", "movimiento", "microvariacion_cents", "rango_db",
                "inclinacion_db", "hnr_db"]


# ------------------------------------------------------------------ datos
def fragmentos(x):
    n, s = int(VENTANA_S * SR), int(SALTO_S * SR)
    if len(x) < n:
        return []
    return [x[i:i + n] for i in range(0, len(x) - n + 1, s)]


def mel(x):
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=N_MELS, fmin=50, fmax=11000)
    return np.log(m + 1e-6).astype(np.float32)


def construir(a):
    import soundfile as sf
    import librosa
    cache = Path(a.trabajo) / "caracteristicas.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    filas = list(csv.DictReader(open(Path(a.dataset) / "manifiesto.csv", encoding="utf-8")))
    personas = sorted({f["hablante"] for f in filas})
    X, Y, P, R, C, K = [], [], [], [], [], []   # mel, descriptores, persona, real(1)/clon(0), clip, codec(1)
    t0 = time.time()
    for n, f in enumerate(filas):
        clip = Path(f["fichero"]).stem
        fuentes = [(Path(a.dataset) / f["fichero"], 1, 0)]
        clon = Path(a.clones) / f["hablante"] / Path(f["fichero"]).name
        if a.clones and clon.exists():
            fuentes.append((clon, 0, 1))
        ida = Path(a.idavuelta) / f["fichero"]
        if a.idavuelta and ida.exists():
            fuentes.append((ida, 1, 1))
        for ruta, real, codec in fuentes:
            x, sr = sf.read(str(ruta), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != SR:
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            for fr in fragmentos(x):
                if np.sqrt(np.mean(fr ** 2)) < 1e-3:
                    continue
                p = PV.perfil(fr, SR)
                y = [math.log(p["hz"]) if p["hz"] > 0 else float("nan")] + [p[k] for k in DESCRIPTORES[1:]]
                X.append(mel(fr)); Y.append(y); P.append(personas.index(f["hablante"])); R.append(real); C.append(clip)
                K.append(codec)
        if (n + 1) % 20 == 0:
            print(f"  caracteristicas: {n + 1}/{len(filas)} clips · {len(X)} fragmentos · {(time.time() - t0) / 60:.1f} min", flush=True)
    d = {"X": np.stack(X), "Y": np.array(Y, np.float32), "P": np.array(P), "R": np.array(R), "C": np.array(C),
         "K": np.array(K), "personas": np.array(personas)}
    np.savez_compressed(cache, **d)
    return d


def particion(C, P, R, frac=0.2, semilla=0):
    """Clips apartados por persona (los de un mismo clip, real y clon, van juntos)."""
    rng = np.random.default_rng(semilla)
    val = set()
    for p in np.unique(P):
        clips = sorted(set(C[P == p]))
        k = int(round(len(clips) * frac))
        if len(clips) >= 3 and k >= 1:
            val.update(rng.choice(clips, k, replace=False).tolist())
    es_val = np.array([c in val for c in C])
    return ~es_val, es_val


# ------------------------------------------------------------------ modelo
def construir_modelo(n_personas, n_desc):
    import torch
    import torch.nn as nn

    def bloque(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    class Estilo(nn.Module):
        def __init__(self):
            super().__init__()
            self.cuerpo = nn.Sequential(bloque(1, 24), bloque(24, 48), bloque(48, 96), bloque(96, 128))
            self.proy = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
            self.estilo = nn.Linear(128, n_desc)
            self.persona = nn.Linear(128, n_personas)
            self.real = nn.Linear(128, 1)

        def forward(self, m):
            h = self.cuerpo(m[:, None])                    # [B, 128, F', T']
            h = h.mean(2)                                   # a lo largo de las bandas
            h = torch.cat([h.mean(-1), h.std(-1)], 1)       # estadísticos en el tiempo
            e = self.proy(h)
            return e, self.estilo(e), self.persona(e), self.real(e).squeeze(1)

    return Estilo()


def aumentar(m, rng):
    """Canal al azar sobre el log-mel: ganancia, inclinación de ecualización y ruido."""
    b = m.shape[0]
    ganancia = rng.uniform(-6, 6, (b, 1, 1)) * math.log(10) / 20 * 2
    rampa = np.linspace(-1, 1, m.shape[1])[None, :, None] * rng.uniform(-0.7, 0.7, (b, 1, 1))
    ruido = np.log(np.exp(m + ganancia + rampa) + np.exp(rng.uniform(-9, -5, (b, 1, 1))))
    return ruido.astype(np.float32)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clones", default="")
    ap.add_argument("--idavuelta", default="", help="clips reales pasados por el codec (fase2_idavuelta.py)")
    ap.add_argument("--trabajo", required=True)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--hilos", type=int, default=6)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F

    Path(a.trabajo).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.hilos)
    torch.manual_seed(0)
    d = construir(a)
    X, Y, P, R, C = d["X"], d["Y"], d["P"], d["R"], d["C"]
    K = d["K"] if "K" in d else (R == 0).astype(int)
    personas = [str(p) for p in d["personas"]]
    tr, va = particion(C, P, R)
    print(f"fragmentos {len(X)} (reales {int(R.sum())}, clones {int((R == 0).sum())}) · entreno {int(tr.sum())} · "
          f"validacion {int(va.sum())} ({len(set(C[va]))} clips) · personas {personas}", flush=True)

    media = np.nanmean(Y[tr], 0)
    desv = np.nanstd(Y[tr], 0) + 1e-6
    Yz = (Y - media) / desv
    mel_media, mel_desv = X[tr].mean(), X[tr].std() + 1e-6
    Xn = (X - mel_media) / mel_desv

    # persona: solo reales, con pesos por clase (Carlos es el 76 %)
    cuenta = np.bincount(P[tr & (R == 1)], minlength=len(personas)).astype(np.float32)
    peso_persona = torch.tensor(np.where(cuenta > 0, cuenta.sum() / (len(personas) * np.maximum(cuenta, 1)), 0.0))
    hay_clones = (R == 0).any()
    con_clon = np.isin(P, np.unique(P[R == 0])) if hay_clones else np.zeros(len(P), bool)

    modelo = construir_modelo(len(personas), len(DESCRIPTORES))
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    idx_tr = np.where(tr)[0]
    mejor, paciencia, historial = float("inf"), 0, []

    def perdidas(xb, yb, pb, rb, cb, entrenando):
        e, est, per, rea = modelo(torch.from_numpy(xb))
        y = torch.from_numpy(yb)
        mascara = ~torch.isnan(y)
        l_est = F.mse_loss(est[mascara], y[mascara]) if mascara.any() else torch.tensor(0.0)
        reales = torch.from_numpy(rb == 1)
        l_per = F.cross_entropy(per[reales], torch.from_numpy(pb[reales.numpy()]), weight=peso_persona) if reales.any() else torch.tensor(0.0)
        conc = torch.from_numpy(cb)
        l_rea = F.binary_cross_entropy_with_logits(rea[conc], torch.from_numpy(rb[cb].astype(np.float32))) if conc.any() else torch.tensor(0.0)
        return l_est + l_per + l_rea, (l_est.item(), l_per.item(), l_rea.item()), (est, per, rea)

    t0 = time.time()
    for ep in range(a.epocas):
        modelo.train()
        rng.shuffle(idx_tr)
        suma = np.zeros(3)
        for i in range(0, len(idx_tr), a.lote):
            b = idx_tr[i:i + a.lote]
            xb = aumentar(Xn[b], rng)
            total, partes, _ = perdidas(xb, Yz[b], P[b], R[b], con_clon[b], True)
            opt.zero_grad(); total.backward(); opt.step()
            suma += np.array(partes) * len(b)
        modelo.eval()
        with torch.no_grad():
            iv = np.where(va)[0]
            _, pv, _ = perdidas(Xn[iv], Yz[iv], P[iv], R[iv], con_clon[iv], False)
        v = sum(pv)
        historial.append({"epoca": ep, "entreno": (suma / len(idx_tr)).tolist(), "validacion": list(pv)})
        print(f"  epoca {ep + 1:2d} · entreno estilo {suma[0] / len(idx_tr):.3f} persona {suma[1] / len(idx_tr):.3f} "
              f"real {suma[2] / len(idx_tr):.3f} · valid {pv[0]:.3f} {pv[1]:.3f} {pv[2]:.3f} · {(time.time() - t0) / 60:.1f} min", flush=True)
        if v < mejor - 1e-3:
            mejor, paciencia = v, 0
            torch.save(modelo.state_dict(), Path(a.trabajo) / "estilo.pt")
        else:
            paciencia += 1
            if paciencia >= 8:
                print("  parada temprana", flush=True)
                break

    # ---------------------------------------------------------------- evaluacion por CLIP
    modelo.load_state_dict(torch.load(Path(a.trabajo) / "estilo.pt"))
    modelo.eval()
    with torch.no_grad():
        iv = np.where(va)[0]
        e, est, per, rea = modelo(torch.from_numpy(Xn[iv]))
    est = est.numpy() * desv + media
    per, rea = per.numpy(), torch.sigmoid(rea).numpy()
    claves = sorted({(C[i], int(R[i]), int(K[i])) for i in iv})
    por_clip = {k: [j for j, i in enumerate(iv) if (C[i], int(R[i]), int(K[i])) == k] for k in claves}

    r2 = {}
    for j, nombre in enumerate(DESCRIPTORES):
        verd = np.array([np.nanmean(Y[iv[js], j]) for js in por_clip.values()])
        pred = np.array([est[js, j].mean() for js in por_clip.values()])
        ok = ~np.isnan(verd)
        base = np.nanmean(Y[tr, j])
        r2[nombre] = float(1 - ((verd[ok] - pred[ok]) ** 2).sum() / max(((verd[ok] - base) ** 2).sum(), 1e-9))
    reales = [k for k in claves if k[1] == 1 and k[2] == 0]
    aciertos = [int(np.argmax(per[por_clip[k]].mean(0)) == P[iv[por_clip[k][0]]]) for k in reales]
    personas_val = sorted({personas[P[iv[por_clip[k][0]]]] for k in reales})
    con = [k for k in claves if con_clon[iv[por_clip[k][0]]]]
    puntos = {k: rea[por_clip[k]].mean() for k in con}
    def auc_entre(pos, neg):
        return auc([1] * len(pos) + [0] * len(neg), [puntos[k] for k in pos] + [puntos[k] for k in neg])
    g_real = [k for k in con if k[1] == 1 and k[2] == 0]
    g_ida = [k for k in con if k[1] == 1 and k[2] == 1]
    g_clon = [k for k in con if k[1] == 0]
    auc_clip = auc_entre(g_real, g_clon)
    auc_ida_clon = auc_entre(g_ida, g_clon) if g_ida else float("nan")
    auc_real_ida = auc_entre(g_real, g_ida) if g_ida else float("nan")

    puerta = {"estilo_r2_positivos": sum(v > 0 for v in r2.values()), "estilo_de": len(r2),
              "persona_acierto_clip": float(np.mean(aciertos)) if aciertos else float("nan"),
              "persona_azar": 1 / len(personas_val) if personas_val else float("nan"),
              "real_clon_auc_clip": auc_clip, "idavuelta_clon_auc_clip": auc_ida_clon,
              "real_idavuelta_auc_clip": auc_real_ida}
    juez_ok = ((auc_ida_clon > 0.7 and auc_real_ida < 0.75) if g_ida
               else (auc_clip > 0.7 if auc_clip == auc_clip else False))
    puerta["juez_valido"] = bool(juez_ok)
    puerta["pasa"] = bool(puerta["estilo_r2_positivos"] > len(r2) / 2
                          and puerta["persona_acierto_clip"] > puerta["persona_azar"] and juez_ok)
    informe = {"r2_por_descriptor": r2, "puerta": puerta, "personas_en_validacion": personas_val,
               "clips_validacion": len(claves), "historial": historial,
               "normalizacion": {"media": media.tolist(), "desv": desv.tolist(),
                                 "mel_media": float(mel_media), "mel_desv": float(mel_desv)},
               "personas": personas, "descriptores": DESCRIPTORES}
    json.dump(informe, open(Path(a.trabajo) / "informe_fase2.json", "w"), ensure_ascii=False, indent=1)
    print("\n== validacion por clip")
    for k, v in r2.items():
        print(f"   R² {k:22s} {v:+.3f}")
    print(f"   persona: {puerta['persona_acierto_clip']:.2f} de acierto (azar {puerta['persona_azar']:.2f}) en {personas_val}")
    print(f"   real/clon: AUC por clip {auc_clip:.3f} ({len(g_real)} reales, {len(g_clon)} clones)")
    if g_ida:
        print(f"   ida y vuelta/clon: AUC {auc_ida_clon:.3f} (puerta > 0,7) · real/ida y vuelta: AUC {auc_real_ida:.3f} (puerta < 0,75)")
    print(f"   PUERTA {'PASA' if puerta['pasa'] else 'NO PASA'} · informe en {Path(a.trabajo) / 'informe_fase2.json'}")


if __name__ == "__main__":
    main()
