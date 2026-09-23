#!/usr/bin/env python
"""El bucle de difusion ENTERO -> un solo IR de OpenVINO (int8 como la cabeza).

Por fotograma acustico, voz_stream.py hace `pasos` llamadas a la cabeza de difusion y, entre
cada dos, en torch: la guia (cfg), el freno de guia y un paso del solver DPM. Aqui va todo en
un grafo: entradas condition [2,896] (condicional + negativa), speech [2,64] (el ruido, que
sigue saliendo de torch.randn para que la semilla consuma lo mismo), cfg_scale y freno;
salida el latente [1,64].

Se puede trazar porque, con los pasos fijos, lo que el solver hace en Python (indice del
paso, primer orden al principio y al final, segundo orden en medio) es aritmetica sobre
constantes. MEDIDO en la VM (i7-8700T, 6 hilos): el camino de siempre 19,8 ms por fotograma,
este grafo 13,9 ms; en torch fp32 el bucle desplegado da diferencia 0,0 frente al de
produccion, y el int8 queda a 67,9 dB (el mismo calculo, otro orden de redondeo).

Los pasos van en el nombre del fichero (difusion_p6_int8.xml): el grafo solo vale para esos.

    VIBEVOICE_MODELO=... VIBEVOICE_IR=... VIBEVOICE_PASOS=6 python convertir_difusion.py

GUIA DESTILADA (scripts/lora/destilar_guia.py): con VIBEVOICE_CABEZA_GUIA=cabeza.pt se convierte la
cabeza que ya da la salida guiada con la condicion POSITIVA sola. El grafo tiene UNA fila: entradas
condition [1,896] y speech [1,64], sin cfg ni freno (van dentro de los pesos). Sale como
difusion_guia_p6_int8.xml; voz_stream.py lo usa pasada la rampa de arranque (VIBEVOICE_GUIA_DESTILADA).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import convertir_cabeza

MODELO = os.environ["VIBEVOICE_MODELO"]
SALIDA = os.environ["VIBEVOICE_IR"]
PASOS = int(os.environ.get("VIBEVOICE_PASOS", "6"))
CABEZA_GUIA = os.environ.get("VIBEVOICE_CABEZA_GUIA", "")


def planificador(cfg):
    from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler(num_train_timesteps=cfg["ddpm_num_steps"],
                                      beta_schedule=cfg["ddpm_beta_schedule"],
                                      prediction_type=cfg["prediction_type"])


class BucleDifusion(torch.nn.Module):
    """Copia de frenar_guia.sample_speech_tokens (voz_stream.py) con el ruido como entrada.
    Si ese bucle cambia, este tambien."""

    def __init__(self, cabeza, cfg, pasos):
        super().__init__()
        self.cabeza, self.cfg, self.pasos = cabeza, cfg, pasos

    def forward(self, condition, speech, cfg_scale, freno):
        sched = planificador(self.cfg)
        sched.set_timesteps(self.pasos)
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.cabeza(combined, t.repeat(combined.shape[0]).to(combined), condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            std_cond = cond_eps.std(dim=-1, keepdim=True)
            std_guiado = half_eps.std(dim=-1, keepdim=True)
            half_eps = freno * (half_eps * (std_cond / (std_guiado + 1e-8))) + (1.0 - freno) * half_eps
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = sched.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]


class BucleGuia(torch.nn.Module):
    """El bucle con la cabeza destilada: una fila, sin guia ni freno."""

    def __init__(self, cabeza, cfg, pasos):
        super().__init__()
        self.cabeza, self.cfg, self.pasos = cabeza, cfg, pasos

    def forward(self, condition, speech):
        sched = planificador(self.cfg)
        sched.set_timesteps(self.pasos)
        for t in sched.timesteps:
            v = self.cabeza(speech, t.repeat(speech.shape[0]).to(speech), condition)
            speech = sched.step(v, t, speech).prev_sample
        return speech


def main():
    import nncf
    import openvino as ov

    cfg = json.load(open(f"{MODELO}/config.json"))["diffusion_head_config"]
    cabeza = convertir_cabeza.cargar_cabeza()
    if CABEZA_GUIA:
        cabeza.load_state_dict(torch.load(CABEZA_GUIA, map_location="cpu"))
        bucle = BucleGuia(cabeza, cfg, PASOS).eval()
        ej = (torch.randn(1, 896), torch.randn(1, 64))
        base = f"{SALIDA}/difusion_guia_p{PASOS}"
    else:
        bucle = BucleDifusion(cabeza, cfg, PASOS).eval()
        ej = (torch.randn(2, 896), torch.randn(2, 64), torch.tensor(3.5), torch.tensor(0.75))
        base = f"{SALIDA}/difusion_p{PASOS}"
    with torch.no_grad():
        mo = ov.convert_model(bucle, example_input=ej)
    ov.save_model(mo, base + "_fp16.xml")
    m8 = nncf.compress_weights(ov.Core().read_model(base + "_fp16.xml"),
                               mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(m8, base + "_int8.xml")
    print("guardado", base + "_int8.xml", flush=True)


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    main()
