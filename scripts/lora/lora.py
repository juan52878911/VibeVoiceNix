"""LoRA minimo (sin peft) para los dos transformadores de VibeVoice-Realtime-0.5B.

Cada nn.Linear elegido pasa a W x + (B A x) * alfa / r, con B a cero al empezar: el modelo arranca
IDENTICO al original (puerta 1: md5 y huella intactos con el LoRA a cero). Solo se guardan A y B.
Para produccion se FUNDE (W += B A * alfa / r) y el modelo vuelve a ser un Qwen2 normal, asi que la
conversion a OpenVINO (int4/int8) y el RTF no cambian.
"""
import math

import torch
import torch.nn as nn

OBJETIVOS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LinealLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alfa=32, abandono=0.05):
        super().__init__()
        self.base = base
        self.r, self.escala = r, alfa / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.abandono = nn.Dropout(abandono)
        self.activo = True

    def forward(self, x):
        y = self.base(x)
        if self.activo:
            y = y + (self.abandono(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.escala
        return y


def poner(modelo, r=16, alfa=32, abandono=0.05, ramas=("language_model", "tts_language_model")):
    """Envuelve las proyecciones de atencion y MLP de las ramas pedidas. Devuelve los modulos LoRA."""
    m = modelo.model
    envueltos = []
    for rama in ramas:
        for nombre, mod in list(getattr(m, rama).named_modules()):
            for hijo in OBJETIVOS:
                lin = getattr(mod, hijo, None)
                if isinstance(lin, nn.Linear):
                    nuevo = LinealLoRA(lin, r, alfa, abandono).to(lin.weight.device)
                    setattr(mod, hijo, nuevo)
                    envueltos.append((f"{rama}.{nombre}.{hijo}", nuevo))
    return envueltos


def congelar_salvo_lora(modelo, extra=()):
    for p in modelo.parameters():
        p.requires_grad_(False)
    entrenables = []
    for n, p in modelo.named_parameters():
        if n.endswith(".A") or n.endswith(".B") or any(n.startswith(e) for e in extra):
            p.requires_grad_(True)
            entrenables.append(p)
    return entrenables


def estado(modelo, extra=()):
    return {n: p.detach().cpu() for n, p in modelo.named_parameters()
            if n.endswith(".A") or n.endswith(".B") or any(n.startswith(e) for e in extra)}


def cargar(modelo, ruta):
    sd = torch.load(ruta, map_location="cpu")
    faltan = modelo.load_state_dict(sd, strict=False)
    return len(sd), faltan


def activar(modelo, si=True):
    for mod in modelo.modules():
        if isinstance(mod, LinealLoRA):
            mod.activo = si


@torch.no_grad()
def fundir(modelo):
    """W += B A * escala y se quita el envoltorio: el modelo vuelve a su forma original."""
    for padre in list(modelo.modules()):
        for hijo, mod in list(padre.named_children()):
            if isinstance(mod, LinealLoRA):
                mod.base.weight += (mod.B @ mod.A).to(mod.base.weight.dtype) * mod.escala
                setattr(padre, hijo, mod.base)
    return modelo
