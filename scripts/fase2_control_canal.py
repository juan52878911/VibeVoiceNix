#!/usr/bin/env python
"""¿El juez real/clon de la fase 2 oye la PERSONALIDAD o el CANAL?

    python scripts/fase2_control_canal.py --dataset /var/lib/taller/dataset-voces --clones /var/lib/taller/fase1/clon \
        --trabajo /var/lib/taller/fase2

Dio AUC 1,000 separando audio real de clones, y eso es sospechoso: el real sale de vídeos separados con
demucs y el clon es sintético y limpio. Este control pasa el audio REAL de los clips apartados por el
propio códec de VibeVoice (codificador acústico -> latentes -> decodificador): la misma persona, con su
misma prosodia, pausas y energía, pero con el sonido sintético del modelo.

  AUC(real, ida-y-vuelta) ~ 1   y  AUC(ida-y-vuelta, clon) ~ 0,5  -> el juez oye el CÓDEC: no sirve
  AUC(real, ida-y-vuelta) ~ 0,5 y  AUC(ida-y-vuelta, clon) ~ 1    -> oye lo que la GENERACIÓN pierde

Además, cuánto mueven los descriptores la ida y vuelta (lo que el códec cambia por sí solo).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fase2_estilo as F2  # noqa: E402
import perfil_vocal as PV  # noqa: E402


def cargar_codec(modelo_dir, cache):
    import torch
    from auditar_encoder import encoder_comunitario
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        modelo_dir, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    m = modelo.model
    m.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(cache)).items()}, strict=True)
    return m


def ida_y_vuelta(m, x):
    import torch
    with torch.no_grad():
        torch.manual_seed(0)
        e = m.acoustic_tokenizer.encode(torch.from_numpy(x)[None, None].float())
        z, _ = e.sample(dist_type=m.acoustic_tokenizer.std_dist_type)
        try:
            y = m.acoustic_tokenizer.decode(z)                     # [B, T, 64]
        except RuntimeError:
            y = m.acoustic_tokenizer.decode(z.permute(0, 2, 1))    # [B, 64, T]
    return y.reshape(-1).numpy()[: len(x)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clones", required=True)
    ap.add_argument("--trabajo", required=True)
    ap.add_argument("--modelo", default=__import__("os").environ.get("VIBEVOICE_MODELO"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    import soundfile as sf
    import torch
    torch.set_num_threads(6)

    tr_dir = Path(a.trabajo)
    inf = json.load(open(tr_dir / "informe_fase2.json"))
    d = np.load(tr_dir / "caracteristicas.npz", allow_pickle=True)
    C, P, R = d["C"], d["P"], d["R"]
    _, va = F2.particion(C, P, R)
    personas = inf["personas"]
    val_clips = sorted({(str(C[i]), int(P[i])) for i in np.where(va)[0] if R[i] == 1})
    print(f"{len(val_clips)} clips apartados", flush=True)

    modelo = F2.construir_modelo(len(personas), len(F2.DESCRIPTORES))
    modelo.load_state_dict(torch.load(tr_dir / "estilo.pt"))
    modelo.eval()
    nm = inf["normalizacion"]

    def puntuar(x):
        frs = [f for f in F2.fragmentos(x) if np.sqrt(np.mean(f ** 2)) >= 1e-3]
        if not frs:
            return None
        mel = (np.stack([F2.mel(f) for f in frs]) - nm["mel_media"]) / nm["mel_desv"]
        with torch.no_grad():
            _, _, _, real = modelo(torch.from_numpy(mel.astype(np.float32)))
        return float(torch.sigmoid(real).mean())

    codec = cargar_codec(a.modelo, a.cache)
    filas = []
    desvios = {k: [] for k in PV.CLAVES if k not in ("silabas_s", "pausas_min")}
    for clip, p in val_clips:
        persona = personas[p]
        fichero = Path(a.dataset) / persona / f"{clip}.wav"
        xr, sr = sf.read(str(fichero), dtype="float32")
        xv = ida_y_vuelta(codec, xr)
        clon = Path(a.clones) / persona / f"{clip}.wav"
        xc = sf.read(str(clon), dtype="float32")[0] if clon.exists() else None
        fila = {"clip": clip, "persona": persona, "real": puntuar(xr), "ida_vuelta": puntuar(xv),
                "clon": puntuar(xc) if xc is not None else None}
        pr, pv = PV.perfil(xr, sr), PV.perfil(xv, sr)
        for k in desvios:
            if pr[k] == pr[k] and pv[k] == pv[k]:
                desvios[k].append(pv[k] - pr[k])
        filas.append(fila)
        print(f"  {persona:18s} {clip:30s} P(real): real {fila['real']:.3f} · ida y vuelta {fila['ida_vuelta']:.3f} · "
              f"clon {fila['clon'] if fila['clon'] is None else round(fila['clon'], 3)}", flush=True)
        sf.write(str(tr_dir / f"idavuelta-{clip}.wav"), xv, sr, subtype="PCM_16") if len(filas) <= 3 else None

    def auc_de(pos, neg):
        return F2.auc([1] * len(pos) + [0] * len(neg), pos + neg)

    reales = [f["real"] for f in filas if f["real"] is not None]
    idas = [f["ida_vuelta"] for f in filas if f["ida_vuelta"] is not None]
    clones = [f["clon"] for f in filas if f["clon"] is not None]
    res = {"auc_real_vs_clon": auc_de(reales, clones), "auc_real_vs_ida_vuelta": auc_de(reales, idas),
           "auc_ida_vuelta_vs_clon": auc_de(idas, clones),
           "p_real_media": {"real": float(np.mean(reales)), "ida_vuelta": float(np.mean(idas)),
                            "clon": float(np.mean(clones)) if clones else None},
           "desvio_descriptores_ida_vuelta": {k: float(np.median(v)) for k, v in desvios.items() if v},
           "clips": filas}
    json.dump(res, open(tr_dir / "control_canal.json", "w"), ensure_ascii=False, indent=1)
    print("\n== control de canal")
    print(f"   P(real) media: real {res['p_real_media']['real']:.3f} · ida y vuelta {res['p_real_media']['ida_vuelta']:.3f} · clon {res['p_real_media']['clon']}")
    print(f"   AUC real vs clon          {res['auc_real_vs_clon']:.3f}")
    print(f"   AUC real vs ida y vuelta  {res['auc_real_vs_ida_vuelta']:.3f}   (~1 = oye el codec)")
    print(f"   AUC ida y vuelta vs clon  {res['auc_ida_vuelta_vs_clon']:.3f}   (~1 = oye lo que pierde la generacion)")
    print("   desvio de descriptores por la ida y vuelta (mediana):",
          {k: round(v, 2) for k, v in res["desvio_descriptores_ida_vuelta"].items()})


if __name__ == "__main__":
    main()
