#!/usr/bin/env python
"""Fase 3, paso 2: adaptador por voz sobre la condición de la cabeza de difusión.

    python scripts/fase3_adaptador.py --condiciones /var/lib/taller/fase3/condiciones --fase2 /var/lib/taller/fase2b \
        --trabajo /var/lib/taller/fase3

Con la condición de cada fotograma real (fase3_condiciones.py), la cabeza de difusión congelada y un adaptador
pequeño por persona delante de ella:

    c' = c + rms(c) · (β + γ ⊙ ĉ + U Vᵀ ĉ),   ĉ = c / rms(c)      (β, γ ∈ R^896; U, V ∈ R^{896×r}; U = 0 al empezar)

Arranca siendo la identidad. Va escalado por la norma de la condición porque los estados del LM tienen normas
de cientos. Por persona y no desde el vector del codificador de estilo: con cinco identidades una red que
lleve del vector de estilo al ajuste no aprende nada que generalice a una voz nueva; lo que sí se puede
comprobar es si ajustar a UNA voz con su audio real la acerca a esa voz. En producción serían constantes por
voz, dentro del grafo de difusión, sin coste.

Pérdida de la cabeza tal como se entrenó (config: v_prediction, coseno, 1000 pasos): x_t = α_t·x0 + σ_t·ε,
objetivo v = α_t·ε − σ_t·x0, t uniforme.

ANTES DE ENTRENAR, dos comprobaciones de los datos:
  forzado   pérdida de la cabeza base con la condición de SU fotograma frente a la de un fotograma al azar
            de otro clip. Si el forzado está bien, la emparejada es claramente menor.
  latente   pérdida base con la media del codificador frente a una muestra (media + std·ε): dice qué vio la
            cabeza al entrenarse, y se entrena con la de menor pérdida.

PUERTA 3a (fijada antes de medir, docs/plan-personalidad-voz.md): en los clips que la fase 2b apartó, para
cada persona con al menos 5, la pérdida con adaptador es menor que sin él con el IC 95 % por bootstrap sobre
clips entero por debajo de 0. Base y adaptador ven el mismo ruido y los mismos t (semilla por clip). Para
elegir el paso de entrenamiento se usa una parte INTERNA de los clips de entrenamiento, nunca los apartados.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fase2_estilo as F2  # noqa: E402


def ic95(d, n=2000):
    d = np.asarray(d, float)
    medias = np.random.default_rng(0).choice(d, (n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--condiciones", required=True)
    ap.add_argument("--fase2", required=True, help="carpeta de la fase 2b: su particion decide los clips apartados")
    ap.add_argument("--trabajo", required=True)
    ap.add_argument("--modelo", default=__import__("os").environ.get("VIBEVOICE_MODELO"))
    ap.add_argument("--rango", type=int, default=4)
    ap.add_argument("--pasos", type=int, default=1500)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--k", type=int, default=2, help="t por latente en el entrenamiento")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--decaimiento", type=float, default=0.01)
    ap.add_argument("--hilos", type=int, default=6)
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    torch.set_num_threads(a.hilos)
    torch.manual_seed(0)
    trabajo = Path(a.trabajo)
    trabajo.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ datos y particion
    clips = []
    for f in sorted(Path(a.condiciones).glob("*/*.npz")):
        d = np.load(f)
        clips.append({"persona": f.parent.name, "clip": f.stem, "cond": torch.from_numpy(d["cond"].astype(np.float32)),
                      "media": torch.from_numpy(d["media"].astype(np.float32)),
                      "std": torch.from_numpy(np.asarray(d["std"], np.float32))})
    inf2 = json.load(open(Path(a.fase2) / "informe_fase2.json"))
    d2 = np.load(Path(a.fase2) / "caracteristicas.npz", allow_pickle=True)
    _, va = F2.particion(d2["C"], d2["P"], d2["R"])
    apartados = {(inf2["personas"][int(d2["P"][i])], str(d2["C"][i])) for i in np.where(va)[0]}
    personas = sorted({c["persona"] for c in clips})
    for c in clips:
        c["grupo"] = "apartado" if (c["persona"], c["clip"]) in apartados else "entreno"
    rng = np.random.default_rng(1)
    for p in personas:
        ent = [c for c in clips if c["persona"] == p and c["grupo"] == "entreno"]
        if len(ent) >= 7:
            for i in rng.choice(len(ent), max(1, round(len(ent) * 0.15)), replace=False):
                ent[i]["grupo"] = "interno"
    for p in personas:
        cuenta = {g: sum(1 for c in clips if c["persona"] == p and c["grupo"] == g) for g in ("entreno", "interno", "apartado")}
        fot = sum(len(c["media"]) for c in clips if c["persona"] == p and c["grupo"] == "entreno")
        print(f"  {p:22s} clips {cuenta} · {fot} fotogramas de entreno", flush=True)

    # ------------------------------------------------------------ cabeza congelada
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        a.modelo, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    cabeza = modelo.model.prediction_head
    sched = modelo.model.noise_scheduler
    alfa, sigma = sched.alpha_t.float(), sched.sigma_t.float()
    for q in cabeza.parameters():
        q.requires_grad_(False)
    del modelo.model.language_model, modelo.model.tts_language_model, modelo.model.acoustic_tokenizer

    def perdida(x0, c, t, eps):
        at, st = alfa[t].unsqueeze(-1), sigma[t].unsqueeze(-1)
        pred = cabeza(at * x0 + st * eps, t.float(), condition=c)
        return ((pred - (at * eps - st * x0)) ** 2).mean(-1)

    class Adaptador(nn.Module):
        def __init__(self, dim, rango):
            super().__init__()
            self.beta = nn.Parameter(torch.zeros(dim))
            self.gamma = nn.Parameter(torch.zeros(dim))
            self.U = nn.Parameter(torch.zeros(dim, rango))
            self.V = nn.Parameter(torch.randn(dim, rango) * 0.02)

        def forward(self, c):
            rms = c.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            h = c / rms
            return c + rms * (self.beta + self.gamma * h + (h @ self.V) @ self.U.T)

    def perdida_clip(clip, objetivo, adaptador=None, k=16):
        """Media por fotograma con ruido y t fijos por clip: base y adaptador ven lo mismo."""
        g = torch.Generator().manual_seed(zlib.crc32(f"{clip['persona']}/{clip['clip']}".encode()))
        T = len(clip["media"])
        x0 = clip["media"].repeat(k, 1)
        if objetivo == "muestra":
            x0 = x0 + clip["std"] * torch.randn(x0.shape, generator=g)
        t = torch.randint(0, len(alfa), (T * k,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        c = clip["cond"].repeat(k, 1)
        with torch.no_grad():
            if adaptador is not None:
                c = adaptador(c)
            return float(perdida(x0, c, t, eps).mean())

    # ------------------------------------------------------------ comprobaciones
    t0 = time.time()
    muestra_clips = [clips[i] for i in np.random.default_rng(2).choice(len(clips), min(30, len(clips)), replace=False)]
    todas_cond = torch.cat([c["cond"] for c in clips])
    emparejada, cruzada, con_media, con_muestra = [], [], [], []
    for c in muestra_clips:
        con_media.append(perdida_clip(c, "media"))
        con_muestra.append(perdida_clip(c, "muestra"))
        otra = dict(c)
        otra["cond"] = todas_cond[torch.randint(0, len(todas_cond), (len(c["cond"]),),
                                                generator=torch.Generator().manual_seed(3))]
        emparejada.append(con_media[-1])
        cruzada.append(perdida_clip(otra, "media"))
    comprobaciones = {"forzado_emparejada": float(np.mean(emparejada)), "forzado_cruzada": float(np.mean(cruzada)),
                      "latente_media": float(np.mean(con_media)), "latente_muestra": float(np.mean(con_muestra))}
    objetivo = "media" if comprobaciones["latente_media"] <= comprobaciones["latente_muestra"] else "muestra"
    comprobaciones["objetivo"] = objetivo
    print(f"== comprobaciones ({time.time() - t0:.0f} s)")
    print(f"   forzado: condicion emparejada {comprobaciones['forzado_emparejada']:.4f} · cruzada {comprobaciones['forzado_cruzada']:.4f}")
    print(f"   latente: media {comprobaciones['latente_media']:.4f} · muestra {comprobaciones['latente_muestra']:.4f} -> {objetivo}", flush=True)
    if comprobaciones["forzado_emparejada"] >= 0.9 * comprobaciones["forzado_cruzada"]:
        print("   EL FORZADO NO SE NOTA: la condicion no ayuda a la cabeza; no se entrena", flush=True)
        json.dump({"comprobaciones": comprobaciones, "pasa": False}, open(trabajo / "informe_fase3a.json", "w"), indent=1)
        return

    # ------------------------------------------------------------ entrenamiento
    adaptadores = nn.ModuleDict({p.replace("-", "_"): Adaptador(896, a.rango) for p in personas})
    opt = torch.optim.AdamW(adaptadores.parameters(), lr=a.lr, weight_decay=a.decaimiento)
    pools = {}
    for p in personas:
        ent = [c for c in clips if c["persona"] == p and c["grupo"] == "entreno"]
        pools[p] = (torch.cat([c["media"] for c in ent]), torch.cat([c["cond"] for c in ent]),
                    torch.cat([c["std"].expand_as(c["media"]) for c in ent]))
    internos = {p: [c for c in clips if c["persona"] == p and c["grupo"] == "interno"] for p in personas}
    base_interno = {p: [perdida_clip(c, objetivo) for c in internos[p]] for p in personas}

    def mejora_interna():
        rel = []
        for p in personas:
            if internos[p]:
                ad = adaptadores[p.replace("-", "_")]
                con = [perdida_clip(c, objetivo, ad) for c in internos[p]]
                rel.append(np.mean(con) / np.mean(base_interno[p]) - 1)
        return float(np.mean(rel)) if rel else float("nan")

    mejor, historial = (float("inf"), -1), []
    t0 = time.time()
    for paso in range(1, a.pasos + 1):
        total = 0.0
        for p in personas:
            X0, C, S = pools[p]
            idx = torch.randint(0, len(X0), (a.lote,))
            x0 = X0[idx].repeat(a.k, 1)
            if objetivo == "muestra":
                x0 = x0 + S[idx].repeat(a.k, 1) * torch.randn_like(x0)
            c = adaptadores[p.replace("-", "_")](C[idx].repeat(a.k, 1))
            t = torch.randint(0, len(alfa), (len(x0),))
            total = total + perdida(x0, c, t, torch.randn_like(x0)).mean()
        opt.zero_grad()
        total.backward()
        opt.step()
        if paso % 100 == 0 or paso == a.pasos:
            m = mejora_interna()
            historial.append({"paso": paso, "perdida": float(total) / len(personas), "mejora_interna": m})
            print(f"  paso {paso} · perdida {float(total) / len(personas):.4f} · interna {m * 100:+.2f} % · "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if m == m and m < mejor[0]:
                mejor = (m, paso)
                torch.save(adaptadores.state_dict(), trabajo / "adaptador.pt")
    if mejor[1] < 0:
        torch.save(adaptadores.state_dict(), trabajo / "adaptador.pt")
    adaptadores.load_state_dict(torch.load(trabajo / "adaptador.pt"))
    print(f"  mejor paso por la parte interna: {mejor[1]} ({mejor[0] * 100:+.2f} %)", flush=True)

    # ------------------------------------------------------------ puerta 3a
    resultado = {}
    for p in personas:
        ap_clips = [c for c in clips if c["persona"] == p and c["grupo"] == "apartado"]
        if not ap_clips:
            continue
        ad = adaptadores[p.replace("-", "_")]
        base = [perdida_clip(c, objetivo) for c in ap_clips]
        con = [perdida_clip(c, objetivo, ad) for c in ap_clips]
        with torch.no_grad():
            cc = torch.cat([c["cond"] for c in ap_clips])
            mov = float(((ad(cc) - cc).norm(dim=-1) / cc.norm(dim=-1)).mean())
        media, lo, hi = ic95(np.array(con) - np.array(base))
        resultado[p] = {"clips": len(ap_clips), "base": float(np.mean(base)), "adaptador": float(np.mean(con)),
                        "dif": media, "ic": [lo, hi], "relativa": float(np.mean(con) / np.mean(base) - 1),
                        "movimiento_condicion": mov, "evaluable": len(ap_clips) >= 5}
    evaluables = [p for p, r in resultado.items() if r["evaluable"]]
    pasa = bool(evaluables) and all(resultado[p]["ic"][1] < 0 for p in evaluables)
    informe = {"comprobaciones": comprobaciones, "argumentos": vars(a), "mejor_paso": mejor[1],
               "historial": historial, "apartados": resultado, "evaluables": evaluables, "pasa": pasa}
    json.dump(informe, open(trabajo / "informe_fase3a.json", "w"), ensure_ascii=False, indent=1)
    print("\n== puerta 3a: perdida de difusion en los clips apartados")
    for p, r in resultado.items():
        print(f"   {p:22s} n={r['clips']:2d} · base {r['base']:.4f} · adaptador {r['adaptador']:.4f} · "
              f"{r['relativa'] * 100:+.2f} % · IC [{r['ic'][0]:+.4f}, {r['ic'][1]:+.4f}] · "
              f"mueve la condicion {r['movimiento_condicion'] * 100:.1f} %{'' if r['evaluable'] else ' (no evaluable)'}")
    print(f"   {'PUERTA 3a PASA' if pasa else 'PUERTA 3a NO PASA'} · evaluables {evaluables}")


if __name__ == "__main__":
    main()
