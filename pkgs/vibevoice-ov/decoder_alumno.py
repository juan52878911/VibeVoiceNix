"""Decodificador acustico ALUMNO: la misma arquitectura que decoder_manual.Decodificador (mismas piezas,
misma interfaz de streaming con estado explicito, mismo conversor a OpenVINO), con los canales y la
profundidad como parametros. Es el destino de la destilacion del decodificador (plan de optimizacion
tras F7): el original tiene ~340 M parametros (268 M en los 8 bloques de 2048 canales de la etapa 0) y
es el 37 % del tiempo de cada fotograma en produccion.

  DecodificadorAlumno(escala=0.5, profundidades=(8, 3, 3, 3, 3, 3, 3))  -> la mitad de canales en todo

Las formas del estado siguen el mismo orden que el original, asi que motor.AcusticoOV lo carga igual.
"""
import torch.nn as nn

from decoder_manual import RATIOS, Bloque, ConvCausal, SubidaTr

DIMS_ORIGINAL = [2048, 1024, 512, 256, 128, 64, 32]


class DecodificadorAlumno(nn.Module):
    def __init__(self, escala=0.5, profundidades=(8, 3, 3, 3, 3, 3, 3)):
        super().__init__()
        self.dims = [max(8, int(d * escala)) for d in DIMS_ORIGINAL]
        self.profundidades = list(profundidades)
        d = self.dims
        self.stem = ConvCausal(64, d[0])
        self.subidas = nn.ModuleList(SubidaTr(d[i], d[i + 1], RATIOS[i]) for i in range(6))
        self.etapas = nn.ModuleList(nn.ModuleList(Bloque(d[i]) for _ in range(self.profundidades[i]))
                                    for i in range(7))
        self.head = ConvCausal(d[6], 1)
        self.formas_estado = [(64, 6)]
        for i in range(7):
            if i > 0:
                self.formas_estado.append((d[i - 1], RATIOS[i - 1] * 2 - 1))
            self.formas_estado += [(d[i], 6)] * self.profundidades[i]
        self.formas_estado.append((d[6], 6))
        self.iniciar()

    def iniciar(self):
        """Pesos pequenos y gammas de capa (layer scale) como el original (1e-6 -> aqui 1e-2 para que aprenda)."""
        for n, p in self.named_parameters():
            if n.endswith("gamma") or n.endswith("gamma_"):
                nn.init.constant_(p, 1e-2)
            elif n.endswith("norm_w") or n.endswith("ffn_norm_w"):
                nn.init.ones_(p)
            elif p.dim() > 1:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(self, lat, *estados):
        nuevos = []
        j = 0

        def paso(mod, x):
            nonlocal j
            y, ne = mod(x, estados[j])
            nuevos.append(ne)
            j += 1
            return y

        x = paso(self.stem, lat)
        for i in range(7):
            if i > 0:
                x = paso(self.subidas[i - 1], x)
            for blq in self.etapas[i]:
                x = paso(blq, x)
        audio = paso(self.head, x)
        return (audio, *nuevos)
