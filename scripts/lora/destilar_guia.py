#!/usr/bin/env python3
"""Destila la guia sin clasificador (CFG) en la cabeza de difusion: menos RTF sin tocar el resto.

POR QUE. Cada fotograma pasa DOS veces por el tts_lm (la rama con texto y la negativa <|image_pad|> del
CFG) y la cabeza calcula las dos ramas en cada paso del solver: en produccion son 43 de los 111 ms del
fotograma el backbone (2,11 pasadas) y 16 la cabeza (motor.py, crono del 13-09). Si la cabeza aprende a
dar, con SOLO la condicion positiva, lo que hoy da la guia completa (cfg 3,0 + freno 0,75, lo de
produccion), la rama negativa sobra entera: ~1 pasada de backbone y media cabeza menos por fotograma.

QUE SE ENTRENA. Una copia de la cabeza (el alumno), entera; el maestro es la cabeza original con las dos
ramas, la guia y el freno EXACTOS de voz_stream.py (sample_speech_tokens de frenar_guia). La rama
positiva del backbone no cambia: sin la negativa, lo que ve el alumno es bit a bit lo que veia el maestro.

DONDE SE ENTRENA. En los puntos que el solver visita de verdad (DPM-Solver multipaso, 6 pasos):
  - trayectorias del MAESTRO desde ruido, con su objetivo guiado en cada paso;
  - trayectorias del ALUMNO (lo que visitara al generar), etiquetadas por el maestro;
  - latentes reales con ruido a un paso del solver (el anclaje a datos).
Las condiciones salen de condiciones.py (pasadas forzadas sobre CML-TTS / LibriTTS-R con prefijo de voz).

  python3 destilar_guia.py --condiciones condiciones/ --salida guia1/ [--pasos 4000] [--lr 5e-5]
Guarda cabeza_<paso>.pt y cabeza_mejor.pt (state_dict de prediction_head) y registro.jsonl.
La puerta NO es la perdida: es evaluar.py --cabeza ... --cfg 1 generando, y el banco en la VM.
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))


def guiado(cabeza, x, t, c, cn, cfg, freno):
    """La salida guiada de produccion (voz_stream.frenar_guia): CFG y freno de energia."""
    v = cabeza(torch.cat([x, x]), torch.cat([t, t]), condition=torch.cat([c, cn]))
    vc, vu = v.chunk(2)
    g = vu + cfg * (vc - vu)
    if freno > 0:
        g = freno * (g * (vc.std(dim=-1, keepdim=True) / (g.std(dim=-1, keepdim=True) + 1e-8))) + (1 - freno) * g
    return g


def trayectoria(programa, n_pasos, x, paso_fn):
    """Recorre el solver desde x con paso_fn(x, t) -> v; devuelve [(x, t)] visitados y el final."""
    sch = copy.deepcopy(programa)
    sch.set_timesteps(n_pasos)
    visitados = []
    for t in sch.timesteps:
        tt = t.repeat(x.shape[0]).to(x)
        visitados.append((x, tt))
        x = sch.step(paso_fn(x, tt), t, x).prev_sample
    return visitados, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condiciones", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--pasos", type=int, default=4000)
    ap.add_argument("--lote", type=int, default=256, help="fotogramas por paso (cada uno aporta ~13 puntos)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--freno", type=float, default=0.75)
    ap.add_argument("--pasos-solver", type=int, default=6)
    ap.add_argument("--cada", type=int, default=250)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    from entrenar import cargar_modelo
    _, modelo = cargar_modelo(a.modelo, d)
    maestro = modelo.model.prediction_head.eval()
    for p in maestro.parameters():
        p.requires_grad_(False)
    alumno = copy.deepcopy(maestro).train()
    for p in alumno.parameters():
        p.requires_grad_(True)
    programa = modelo.model.noise_scheduler
    n_param = sum(p.numel() for p in alumno.parameters())
    # fotogramas: condiciones y latentes en la GPU, validacion por hablante
    filas = []
    for f in sorted(Path(a.condiciones).glob("*.pt")):
        filas += torch.load(f, map_location="cpu")
    hablantes = sorted({(r["idioma"], r["hablante"]) for r in filas})
    random.Random(1).shuffle(hablantes)
    val_h = set(hablantes[: max(2, len(hablantes) // 30)])

    def apilar(sel):
        return (torch.cat([r["cond"] for r in sel]).to(d).float(), torch.cat([r["cond_neg"] for r in sel]).to(d).float(),
                torch.cat([r["lat"] for r in sel]).to(d).float())
    C, CN, L = apilar([r for r in filas if (r["idioma"], r["hablante"]) not in val_h])
    VC, VCN, VL = apilar([r for r in filas if (r["idioma"], r["hablante"]) in val_h])
    VC, VCN = VC[:2048], VCN[:2048]
    print(f"[guia] alumno {n_param / 1e6:.1f} M parametros; {C.shape[0]} fotogramas de entrenamiento, "
          f"{VC.shape[0]} de validacion ({len(val_h)} lectores apartados)", flush=True)
    (sal / "config.json").write_text(json.dumps({**vars(a), "parametros_M": round(n_param / 1e6, 2),
                                                 "fotogramas": C.shape[0]}, indent=1))
    alfa_bar = programa.alphas_cumprod.to(d)
    sch6 = copy.deepcopy(programa)
    sch6.set_timesteps(a.pasos_solver)
    t_solver = sch6.timesteps.to(d)

    def maestro_v(x, t, c, cn):
        with torch.no_grad():
            return guiado(maestro, x, t, c, cn, a.cfg, a.freno)

    ruido_val = torch.Generator(device=d).manual_seed(123)
    x0_val = torch.randn(VC.shape[0], L.shape[1], device=d, generator=ruido_val)

    def validar():
        """Final del solver del alumno (solo condicion positiva) frente al del maestro (guiado), mismo ruido.
        Error relativo ||a - m||^2 / ||m||^2; como referencia, el de la rama positiva sin guia (cfg 1)."""
        alumno.eval()
        with torch.no_grad():
            _, fin_m = trayectoria(programa, a.pasos_solver, x0_val, lambda x, t: maestro_v(x, t, VC, VCN))
            _, fin_a = trayectoria(programa, a.pasos_solver, x0_val, lambda x, t: alumno(x, t, condition=VC))
            _, fin_1 = trayectoria(programa, a.pasos_solver, x0_val, lambda x, t: maestro(x, t, condition=VC))
        alumno.train()
        den = (fin_m ** 2).sum()
        return {"err_alumno": round(float(((fin_a - fin_m) ** 2).sum() / den), 5),
                "err_sin_guia": round(float(((fin_1 - fin_m) ** 2).sum() / den), 5),
                "cos_alumno": round(float(torch.nn.functional.cosine_similarity(fin_a, fin_m).mean()), 5)}

    opt = torch.optim.AdamW(alumno.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda p: min(1.0, (p + 1) / 100) * max(0.05, 1 - p / a.pasos))
    registro = open(sal / "registro.jsonl", "a")
    v0 = validar()
    print(f"[guia] antes: {v0}", flush=True)
    registro.write(json.dumps({"paso": 0, "val": v0}) + "\n")
    mejor = v0["err_alumno"]
    t0, media = time.time(), []
    for paso in range(1, a.pasos + 1):
        i = torch.randint(0, C.shape[0], (a.lote,), device=d)
        c, cn, x0 = C[i], CN[i], L[i]
        xs, ts, cs, cns = [], [], [], []
        # 1. trayectoria del maestro desde ruido
        with torch.no_grad():
            vis, _ = trayectoria(programa, a.pasos_solver, torch.randn_like(x0), lambda x, t: maestro_v(x, t, c, cn))
            # 2. trayectoria del alumno (lo que visitara al generar)
            alumno.eval()
            vis_a, _ = trayectoria(programa, a.pasos_solver, torch.randn_like(x0), lambda x, t: alumno(x, t, condition=c))
            alumno.train()
        for x, t in vis + vis_a:
            xs.append(x); ts.append(t); cs.append(c); cns.append(cn)
        # 3. latentes reales con ruido en un paso del solver
        k = t_solver[torch.randint(0, len(t_solver), (a.lote,), device=d)]
        ab = alfa_bar[k.long()][:, None]
        xs.append(ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)); ts.append(k.float()); cs.append(c); cns.append(cn)
        X, T, CC, CCN = torch.cat(xs), torch.cat(ts), torch.cat(cs), torch.cat(cns)
        objetivo = maestro_v(X, T, CC, CCN)
        perdida = torch.nn.functional.mse_loss(alumno(X, T, condition=CC), objetivo)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(alumno.parameters(), 1.0)
        opt.step()
        sched.step()
        media.append(float(perdida))
        if paso % 50 == 0:
            print(f"[guia] paso {paso}/{a.pasos} · perdida {sum(media) / len(media):.5f} · lr {sched.get_last_lr()[0]:.1e} · "
                  f"{(time.time() - t0) / paso:.2f} s/paso", flush=True)
            media = []
        if paso % a.cada == 0 or paso == a.pasos:
            v = validar()
            print(f"[guia] validacion paso {paso}: {v}", flush=True)
            registro.write(json.dumps({"paso": paso, "val": v}) + "\n")
            registro.flush()
            torch.save(alumno.state_dict(), sal / f"cabeza_{paso}.pt")
            if v["err_alumno"] < mejor:
                mejor = v["err_alumno"]
                torch.save(alumno.state_dict(), sal / "cabeza_mejor.pt")
    print(f"[guia] fin: {a.pasos} pasos en {(time.time() - t0) / 60:.1f} min; mejor error {mejor} "
          f"(sin guia {v0['err_sin_guia']})", flush=True)


if __name__ == "__main__":
    main()
