#!/usr/bin/env python3
"""Destila el decodificador acustico de VibeVoice en uno mas pequeno (plan de optimizacion tras F7).

POR QUE. El decodificador (latentes a 7,5 Hz -> audio a 24 kHz) es el 37 % del tiempo de cada fotograma
en produccion (~41 ms) y tiene ~340 M parametros. Medido en la VM (banco_decoder_alumno.py, pesos
aleatorios): la misma arquitectura con la mitad de canales tarda 10,8 ms y tiene 86 M parametros.
Si el alumno suena igual que el maestro, el RTF baja de ~0,885 a ~0,66 (estimado) sin tocar el modelo
de lenguaje, la cabeza ni las voces.

DATOS. Latentes de audio real (los de datos.py: CML-TTS y LibriTTS-R) pasados a la escala del
decodificador. El objetivo es la salida del MAESTRO sobre los mismos latentes, no el audio real: se
aprende a imitar al decodificador, que es lo que la puerta compara.

PERDIDA. STFT multirresolucion (convergencia espectral y log-magnitud), mel L1 y un poco de L1 en la onda.

  python3 destilar_decoder.py --datos datos/ --salida dec1/ [--escala 0.5] [--pasos 20000]
Guarda alumno_<paso>.pt y alumno_mejor.pt (state_dict + escala + profundidades) y registro.jsonl.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent / "pkgs" / "vibevoice-ov"))

MUESTRAS = 3200


def stft_mag(x, n_fft, hop):
    w = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-7)


def perdida_stft(y, t):
    total = 0.0
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        a, b = stft_mag(y, n_fft, hop), stft_mag(t, n_fft, hop)
        sc = torch.linalg.norm(b - a, dim=(1, 2)) / torch.linalg.norm(b, dim=(1, 2)).clamp(min=1e-7)
        total = total + sc.mean() + F.l1_loss(a.log(), b.log())
    return total / 3


class Mel:
    def __init__(self, dispositivo, n=80):
        import librosa
        self.fb = torch.tensor(librosa.filters.mel(sr=24000, n_fft=1024, n_mels=n, fmin=0, fmax=12000),
                               dtype=torch.float32, device=dispositivo)

    def __call__(self, x):
        return (self.fb @ stft_mag(x, 1024, 256) ** 2).clamp(min=1e-7).log()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--escala", type=float, default=0.5)
    ap.add_argument("--profundidades", default="8,3,3,3,3,3,3")
    ap.add_argument("--pasos", type=int, default=20000)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--latentes", type=int, default=24, help="latentes por recorte (24 = 3,2 s)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cada", type=int, default=1000)
    a = ap.parse_args()
    d = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    import json as _j
    from decoder_alumno import DecodificadorAlumno
    from decoder_manual import cargar
    cfg = _j.load(open(f"{a.modelo}/config.json"))
    # escala de los latentes: los de datos.py estan en la de la difusion ((z + sesgo) * escala)
    from safetensors import safe_open
    with safe_open(f"{a.modelo}/model.safetensors", framework="pt") as f:
        claves = set(f.keys())
        sesgo = float(f.get_tensor("model.speech_bias_factor")) if "model.speech_bias_factor" in claves else None
        escala_lat = float(f.get_tensor("model.speech_scaling_factor")) if "model.speech_scaling_factor" in claves else None
    print(f"[dec] escala de latentes: sesgo {sesgo} escala {escala_lat}", flush=True)
    maestro = cargar(f"{a.modelo}/model.safetensors").to(d).eval()
    for p in maestro.parameters():
        p.requires_grad_(False)
    prof = tuple(int(x) for x in a.profundidades.split(","))
    alumno = DecodificadorAlumno(a.escala, prof).to(d).train()
    n_al = sum(p.numel() for p in alumno.parameters())
    n_ma = sum(p.numel() for p in maestro.parameters())
    print(f"[dec] maestro {n_ma / 1e6:.0f} M, alumno {n_al / 1e6:.0f} M (escala {a.escala}, profundidades {prof})", flush=True)
    # latentes: lista de [T, 64] en escala del decodificador; validacion por hablante
    ejemplos = []
    for f in sorted(Path(a.datos).glob("*.pt")):
        ejemplos += torch.load(f, map_location="cpu")
    secuencias = []
    for e in ejemplos:
        for clave in ("lat", "ref_lat"):
            z = e[clave].float() / escala_lat - sesgo
            if z.shape[0] >= a.latentes:
                secuencias.append(((e["idioma"], e["hablante"]), z))
    hablantes = sorted({h for h, _ in secuencias})
    random.Random(1).shuffle(hablantes)
    val_h = set(hablantes[: max(2, len(hablantes) // 30)])
    ent = [z for h, z in secuencias if h not in val_h]
    val = [z for h, z in secuencias if h in val_h][:64]
    print(f"[dec] {len(ent)} secuencias de entrenamiento, {len(val)} de validacion "
          f"({sum(z.shape[0] for z in ent) / 7.5 / 3600:.1f} h)", flush=True)
    mel = Mel(d)

    def estados(m, b):
        return [torch.zeros(b, c, k, device=d) for c, k in m.formas_estado]

    def recortes(fuente, n):
        out = []
        for _ in range(n):
            z = random.choice(fuente)
            i = random.randint(0, z.shape[0] - a.latentes)
            out.append(z[i:i + a.latentes])
        return torch.stack(out).transpose(1, 2).to(d)            # [B, 64, T]

    def pares(z):
        with torch.no_grad():
            t = maestro(z, *estados(maestro, z.shape[0]))[0][:, 0]
        return t

    def perdida(y, t):
        return perdida_stft(y, t) + F.l1_loss(mel(y), mel(t)) + 0.1 * F.l1_loss(y, t)

    val_z = torch.stack([z[:a.latentes] for z in val]).transpose(1, 2).to(d)
    val_t = torch.cat([pares(val_z[i:i + 8]) for i in range(0, val_z.shape[0], 8)])

    def validar():
        alumno.eval()
        with torch.no_grad():
            ys = torch.cat([alumno(val_z[i:i + 8], *estados(alumno, val_z[i:i + 8].shape[0]))[0][:, 0]
                            for i in range(0, val_z.shape[0], 8)])
            r = {"total": float(perdida(ys, val_t)), "mel_l1": float(F.l1_loss(mel(ys), mel(val_t)))}
        alumno.train()
        return {k: round(v, 5) for k, v in r.items()}

    opt = torch.optim.AdamW(alumno.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda p: min(1.0, (p + 1) / 500) * max(0.05, 1 - p / a.pasos))
    registro = open(sal / "registro.jsonl", "a")
    mejor, t0, media = 9e9, time.time(), []
    for paso in range(1, a.pasos + 1):
        z = recortes(ent, a.lote)
        t = pares(z)
        y = alumno(z, *estados(alumno, z.shape[0]))[0][:, 0]
        loss = perdida(y, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(alumno.parameters(), 1.0)
        opt.step()
        sched.step()
        media.append(float(loss))
        if paso % 100 == 0:
            print(f"[dec] paso {paso}/{a.pasos} · perdida {sum(media) / len(media):.4f} · "
                  f"{(time.time() - t0) / paso:.2f} s/paso", flush=True)
            media = []
        if paso % a.cada == 0 or paso == a.pasos:
            v = validar()
            print(f"[dec] validacion paso {paso}: {v}", flush=True)
            registro.write(json.dumps({"paso": paso, "val": v}) + "\n")
            registro.flush()
            estado = {"escala": a.escala, "profundidades": prof, "estado": alumno.state_dict()}
            torch.save(estado, sal / f"alumno_{paso}.pt")
            if v["total"] < mejor:
                mejor = v["total"]
                torch.save(estado, sal / "alumno_mejor.pt")
    print(f"[dec] fin: {a.pasos} pasos en {(time.time() - t0) / 60:.1f} min; mejor validacion {mejor:.4f}", flush=True)


if __name__ == "__main__":
    main()
