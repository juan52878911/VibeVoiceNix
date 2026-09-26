"""La verosimilitud del propio modelo como senal de calidad (plan de la red, §5.5).

Para cada fotograma generado, cuanto le sorprende a la cabeza de difusion el latente que ella misma eligio, dada
la condicion que lo produjo: perdida v (v-prediction, programa coseno de 1000 pasos, como forzado.py) promediada
en varios t con ruido de semilla fija. Una racha alta es el aviso temprano de musica inventada o alucinacion; en
lote, la media por clip se contrasta con el WER de whisper (a medir).

    s = por_fotograma(modelo, registro)     # registro: lista de dicts con 'cond' [896] y 'lat' [64] (bucle.Generador)
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lora"))
from forzado import Programa  # noqa: E402


@torch.no_grad()
def por_fotograma(modelo, registro, n_t=8, semilla=0):
    d = next(modelo.parameters()).device
    prog = Programa(dispositivo=d)
    g = torch.Generator(device="cpu").manual_seed(semilla)
    t = torch.linspace(50, 950, n_t).long().to(d)
    sal = []
    for r in registro:
        lat, cond = r["lat"][None].to(d).expand(n_t, -1), r["cond"][None].to(d).expand(n_t, -1)
        eps = torch.randn(lat.shape, generator=g).to(d)
        xt, v = prog.ruido(lat, t, eps)
        pred = modelo.model.prediction_head(xt, t.float(), condition=cond)
        sal.append(F.mse_loss(pred, v).item())
    return torch.tensor(sal)
