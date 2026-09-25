#!/usr/bin/env python3
"""DPO, paso 3 (plan de preferencias, 24-09): Diffusion-DPO (Wallace et al., 2023) en v-prediccion sobre las
propias salidas del modelo, juzgadas (dpo_generar.py + juez_lote.py + dpo_puntuar.py).

  python3 dpo_entrenar.py --datos d1/ --salida d2/b2000 --beta 2000 [--pasos 600 --acumular 4 --lr 5e-5]

Por par (ganador w, perdedor l; misma voz y texto, semillas distintas), en cada fotograma, con el MISMO t y
el MISMO ruido para el modelo y la referencia:
  e_th(x)  = || v - v_head(x_t, t | cond_th) ||^2       cond de forzado.estados con el LoRA puesto
  e_ref(x) = || v - v_head(x_t, t | cond_ref) ||^2      el mismo modelo con el LoRA apagado (sin gradiente)
  D = [e_th(w) - e_ref(w)] - [e_th(l) - e_ref(l)]        (media por fotograma)
  L = -log sigmoid(-beta D) + lam * L_sft(w)             L_sft = difusion + 0,15 negativa (forzado.pasada)
La cabeza queda congelada: el gradiente llega a los LoRA de los dos LM por la condicion. La condicion de la
referencia no cambia en todo el entrenamiento: se calcula una vez por muestra y se guarda.
Los latentes son los que genero la cabeza; un perdedor cortado por el fin se fuerza tal cual (cortar=True).

Validacion DPO en pares de lectores apartados (t y ruido fijos por par): acierto = fraccion de pares con
D < 0 (el modelo prefiere al ganador mas que la referencia). Base: 0,5 por construccion (D = 0). Se guarda
el mejor por acierto; la puerta la pone la generacion (D3), no esta medida.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402
import lora as LR  # noqa: E402

TOPE_T = 400          # fotogramas como mucho por muestra (53 s): el prefijo de un desbocado basta (causal)


class Muestras:
    """Carga perezosa de latentes y condiciones de referencia (LoRA apagado) por clave."""

    def __init__(self, carpetas, dispositivo):
        self.d = dispositivo
        self.meta, self.ref, self.ruta = {}, {}, {}
        for c in carpetas:
            c = Path(c)
            hs = json.loads((c / "hablantes.json").read_text())
            for f in sorted(c.glob("lote.*.json")):
                for m in json.loads(f.read_text()):
                    h = hs[m["identidad"]]
                    self.meta[m["clave"]] = {"txt": m["texto"], "ref_txt": h["ref_txt"], "identidad": m["identidad"]}
                    self.ruta[m["clave"]] = c
            for ident in hs:
                self.ref[ident] = torch.load(c / "hablantes" / ident / "ref_lat.pt", map_location="cpu").float()
        self.lat, self.cond_ref = {}, {}

    def ejemplo(self, clave):
        if clave not in self.lat:
            lat = torch.load(self.ruta[clave] / "lat" / f"{clave}.pt", map_location="cpu")["lat"].float()
            self.lat[clave] = lat[:TOPE_T]
        m = self.meta[clave]
        return {"ref_lat": self.ref[m["identidad"]], "ref_txt": m["ref_txt"], "lat": self.lat[clave], "txt": m["txt"]}


def ruido(T, programa, d, gen=None):
    t = torch.randint(0, programa.n, (T,), generator=gen)
    eps = torch.randn(T, 64, generator=gen)
    return t.to(d), eps.to(d)


def error(modelo, cond, lat, t, eps, programa):
    xt, v = programa.ruido(lat, t, eps)
    pred = modelo.model.prediction_head(xt, t.float(), condition=cond)
    return ((pred - v) ** 2).mean(), xt, v


def cond_ref(modelo, tok, mues, clave):
    """Condicion de la referencia (LoRA apagado), una sola vez por muestra."""
    if clave not in mues.cond_ref:
        LR.activar(modelo, False)
        with torch.no_grad():
            r = FZ.estados(modelo, tok, mues.ejemplo(clave), cortar=True)
        LR.activar(modelo, True)
        mues.cond_ref[clave] = None if r is None else r[1].detach()
    return mues.cond_ref[clave]


def delta_par(modelo, tok, mues, par, programa, gen=None, lam=0.0, frac_neg=0.15):
    """(D, L_sft del ganador, e_th(w), e_th(l)) de un par, o None si no se puede forzar."""
    d = next(modelo.parameters()).device
    res = []
    for clave in (par["ganador"], par["perdedor"]):
        cr = cond_ref(modelo, tok, mues, clave)
        if cr is None:
            return None
        r = FZ.estados(modelo, tok, mues.ejemplo(clave), cortar=True)
        if r is None:
            return None
        lat, cond, cond_neg = r[0], r[1], r[2]
        t, eps = ruido(lat.shape[0], programa, d, gen)
        e_th, xt, v = error(modelo, cond, lat, t, eps, programa)
        with torch.no_grad():
            e_ref, _, _ = error(modelo, cr, lat, t, eps, programa)
        res.append((e_th, e_ref, xt, v, t, cond_neg))
    (ew, rw, xt, v, t, cneg), (el, rl, *_) = res
    D = (ew - rw) - (el - rl)
    l_sft = ew
    if lam > 0 and frac_neg > 0:
        sel = torch.rand(xt.shape[0], device=d) < frac_neg
        if sel.any():
            pred_n = modelo.model.prediction_head(xt[sel], t[sel].float(), condition=cneg[sel])
            l_sft = l_sft + frac_neg * F.mse_loss(pred_n, v[sel])
    return D, l_sft, ew, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpetas de dpo_generar/dpo_puntuar, separadas por comas")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--beta", type=float, default=2000.0)
    ap.add_argument("--lam", type=float, default=0.1, help="peso del ancla SFT sobre el ganador")
    ap.add_argument("--pasos", type=int, default=600)
    ap.add_argument("--acumular", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--alfa", type=int, default=32)
    ap.add_argument("--ramas", default="language_model,tts_language_model")
    ap.add_argument("--cada", type=int, default=50)
    ap.add_argument("--val", type=float, default=0.1, help="fraccion de lectores apartados para validar")
    ap.add_argument("--val-max", type=int, default=80)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    a = ap.parse_args()
    random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    from entrenar import cargar_modelo
    proc, modelo = cargar_modelo(a.modelo, d)
    tok = proc.tokenizer
    LR.poner(modelo, a.rango, a.alfa, 0.0, tuple(a.ramas.split(",")))   # sin abandono: D no debe llevar ruido extra
    params = LR.congelar_salvo_lora(modelo)
    carpetas = a.datos.split(",")
    mues = Muestras(carpetas, d)
    pares = [json.loads(l) for c in carpetas for l in open(Path(c) / "pares.jsonl")]
    pares = [p for p in pares if p["ganador"] in mues.meta and p["perdedor"] in mues.meta]
    idents = sorted({p["identidad"] for p in pares})
    random.Random(1).shuffle(idents)
    val_i = set(idents[: max(1, round(len(idents) * a.val))])
    val = [p for p in pares if p["identidad"] in val_i]
    random.Random(2).shuffle(val)
    val = val[: a.val_max]
    grupos = defaultdict(list)                     # se muestrea por grupo: un grupo con 6 pares no pesa 6 veces
    for p in pares:
        if p["identidad"] not in val_i:
            grupos[p["grupo"]].append(p)
    lista_g = sorted(grupos)
    motivos = defaultdict(int)
    for g in lista_g:
        for p in grupos[g]:
            motivos[p["motivo"]] += 1
    print(f"[dpo] {sum(len(v) for v in grupos.values())} pares de entrenamiento en {len(lista_g)} grupos "
          f"({dict(motivos)}); {len(val)} de validacion ({len(val_i)} lectores apartados); beta {a.beta}, "
          f"lam {a.lam}, lr {a.lr}", flush=True)
    (sal / "config.json").write_text(json.dumps({**vars(a), "val_lectores": sorted(val_i)}, indent=1))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda p: min(1.0, (p + 1) / 50) * max(0.1, 1 - p / a.pasos))
    programa = FZ.Programa(dispositivo=d)

    def validar():
        modelo.eval()
        Ds, sft = [], []
        with torch.no_grad():
            for i, p in enumerate(val):
                gen = torch.Generator().manual_seed(1000 + i)
                r = delta_par(modelo, tok, mues, p, programa, gen)
                if r is not None:
                    Ds.append(float(r[0]))
                    sft.append(float(r[2]))
        modelo.train()
        if not Ds:
            return {}
        return {"acierto": round(sum(x < 0 for x in Ds) / len(Ds), 4), "D": round(sum(Ds) / len(Ds), 6),
                "absD": round(sum(abs(x) for x in Ds) / len(Ds), 6),
                "dpo": round(sum(float(F.softplus(torch.tensor(a.beta * x))) for x in Ds) / len(Ds), 4),
                "sft_w": round(sum(sft) / len(sft), 5), "n": len(Ds)}

    registro = open(sal / "registro.jsonl", "a")
    base = validar()
    print(f"[dpo] validacion antes de entrenar: {base}", flush=True)
    registro.write(json.dumps({"paso": 0, "val": base}) + "\n")
    mejor = (0.5, 0.0)                 # (acierto, -D): hay que ganar al azar de la base
    modelo.train()
    t0 = time.time()
    media = defaultdict(float)
    for paso in range(1, a.pasos + 1):
        for _ in range(a.acumular):
            p = random.choice(grupos[random.choice(lista_g)])
            r = delta_par(modelo, tok, mues, p, programa, lam=a.lam)
            if r is None:
                continue
            D, l_sft, ew, el = r
            l_dpo = F.softplus(a.beta * D)            # = -log sigmoid(-beta D)
            ((l_dpo + a.lam * l_sft) / a.acumular).backward()
            media["dpo"] += float(l_dpo)
            media["sft"] += float(l_sft)
            media["D"] += float(D)
            media["absD"] += abs(float(D))
            media["acierto"] += float(D < 0)
            media["n"] += 1
        norma = torch.nn.utils.clip_grad_norm_(params, 1.0)
        media["norma"] += float(norma)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if paso % 10 == 0:
            n = max(1, media["n"])
            print(f"[dpo] paso {paso}/{a.pasos} · dpo {media['dpo'] / n:.4f} · sft {media['sft'] / n:.4f} · "
                  f"D {media['D'] / n:+.2e} · |D| {media['absD'] / n:.2e} · beta|D| {a.beta * media['absD'] / n:.3f} · "
                  f"acierto {media['acierto'] / n:.2f} · norma {media['norma'] / 10:.2f} · lr {sched.get_last_lr()[0]:.1e} · "
                  f"{(time.time() - t0) / paso:.2f} s/paso", flush=True)
            registro.write(json.dumps({"paso": paso, "ent": {k: v / (n if k != "norma" else 10) for k, v in media.items()}}) + "\n")
            media = defaultdict(float)
        if paso % a.cada == 0 or paso == a.pasos:
            v = validar()
            print(f"[dpo] validacion paso {paso}: {v}", flush=True)
            registro.write(json.dumps({"paso": paso, "val": v}) + "\n")
            registro.flush()
            torch.save(LR.estado(modelo), sal / f"lora_{paso}.pt")
            if v and (v["acierto"], -v["D"]) > mejor:
                mejor = (v["acierto"], -v["D"])
                torch.save(LR.estado(modelo), sal / "lora_mejor.pt")
                (sal / "mejor.json").write_text(json.dumps({"paso": paso, **v}))
    print(f"[dpo] fin: {a.pasos} pasos en {(time.time() - t0) / 60:.1f} min; mejor acierto de validacion "
          f"{mejor[0]:.3f} (base 0,5)", flush=True)


if __name__ == "__main__":
    main()
