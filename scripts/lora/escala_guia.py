#!/usr/bin/env python3
"""Vara para leer el error de destilar_guia.py: cuanto se aleja el final del solver del de produccion
(cfg 3 + freno 0,75, mismo ruido) con cambios que produccion YA hace o admite: otro cfg (la rampa de
arranque usa 4,5; las peticiones aceptan 0,5-5) y el freno apagado. Si el alumno queda por debajo de
esos, su diferencia es del tamano de un ajuste que ya se da por bueno.

  python3 escala_guia.py --condiciones condiciones/ [--cabeza guia1/cabeza_mejor.pt]
"""
import argparse
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
from destilar_guia import guiado, trayectoria  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condiciones", required=True)
    ap.add_argument("--cabeza", default=None)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    d = "cuda" if torch.cuda.is_available() else "cpu"
    from entrenar import cargar_modelo
    _, modelo = cargar_modelo(a.modelo, d)
    cabeza = modelo.model.prediction_head.eval()
    prog = modelo.model.noise_scheduler
    filas = []
    for f in sorted(Path(a.condiciones).glob("*.pt")):
        filas += torch.load(f, map_location="cpu")
    C = torch.cat([r["cond"] for r in filas]).float()
    CN = torch.cat([r["cond_neg"] for r in filas]).float()
    i = torch.randperm(C.shape[0], generator=torch.Generator().manual_seed(7))[: a.n]
    c, cn = C[i].to(d), CN[i].to(d)
    x0 = torch.randn(a.n, 64, device=d, generator=torch.Generator(device=d).manual_seed(123))

    def fin(fn):
        with torch.no_grad():
            return trayectoria(prog, 6, x0, fn)[1]
    ref = fin(lambda x, t: guiado(cabeza, x, t, c, cn, 3.0, 0.75))
    den = (ref ** 2).sum()

    def err(y):
        return round(float(((y - ref) ** 2).sum() / den), 4)
    filas_sal = []
    for cfg, fr in ((2.5, 0.75), (3.5, 0.75), (4.5, 0.75), (3.0, 0.0), (1.0, 0.75)):
        filas_sal.append((f"cfg {cfg} freno {fr}", err(fin(lambda x, t: guiado(cabeza, x, t, c, cn, cfg, fr)))))
    if a.cabeza:
        alumno = type(cabeza)(cabeza.config).to(d).eval()
        alumno.load_state_dict(torch.load(a.cabeza, map_location=d))
        filas_sal.append(("alumno (solo positiva)", err(fin(lambda x, t: alumno(x, t, condition=c)))))
    for nombre, e in filas_sal:
        print(f"{nombre:28s} error relativo frente a produccion: {e}", flush=True)


if __name__ == "__main__":
    main()
