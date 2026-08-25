#!/usr/bin/env python
"""Audita el encoder acustico comunitario del Realtime-0.5B.

    python scripts/auditar_encoder.py

LA PREGUNTA
El 0.5B no trae encoder acustico: 276 claves del decoder y ninguna del
encoder (docs/clonado-de-voz.md §1). Sin el no hay forma de convertir un audio
en los latentes `z` que necesita un prefijo de voz, y por eso el clonado desde
audio esta marcado como NO RESUELTO.

Existe un tercero que dice haberlo entrenado:
mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder. Esto mide si es
verdad, sin creerse la ficha del modelo.

COMO SE MIDE, Y POR QUE ASI
La prueba es el ciclo `audio -> encoder -> z -> decoder(0.5B) -> audio'`. Si el
encoder vive en el espacio latente del 0.5B, `audio'` se parece a `audio`. Si
no, el decoder recibe basura y saca practicamente silencio -- que es justo lo
que le pasa al encoder del 1.5B (§4.2 del doc: correlacion +0,0063 y RMS 0,0083
frente a 0,0591).

No hace falta recuperar la `z` verdadera de ninguna voz oficial: el ciclo se
cierra solo. Se corren TRES brazos sobre el mismo audio:

  comunitario  enc(comunitario) -> dec(0.5B)     <- lo que se audita
  negativo     enc(1.5B)        -> dec(0.5B)     <- el callejon conocido
  positivo     enc(1.5B)        -> dec(1.5B)     <- el arnes contra si mismo

El brazo positivo es el que da derecho a creerse los otros dos: el doc lo midio
en +0,9799 de correlacion de onda. Si aqui no sale eso, el fallo esta en este
script y no en el encoder que se audita.

EL AUDIO
`ejemplos-voces/sp-*.wav`, que son 24 kHz mono generados por el propio 0.5B con
las seis voces espanolas. Audio dentro de la distribucion del decoder, que es
lo que hace la prueba justa: si fallara con audio de microfono aun podria ser
culpa del audio, pero fallar con su propia salida no tiene excusa.

DE DONDE SALEN LOS PESOS
Del safetensors comunitario se piden por `Range` solo los 276 tensores del
encoder: ocupan los primeros 1311 MB del fichero y ningun otro tensor cae en
ese rango, asi que es UNA peticion y no 3,18 GB. El del 1.5B sale de la cache
local del hub.
"""
import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch

URL_COM = ("https://huggingface.co/mohammed-bahumaish/"
           "vibevoice-realtime-0.5b-with-encoder/resolve/main/model.safetensors")
RITMO = 24000


# --------------------------------------------------------------- utilidades --
def cabecera_safetensors(ruta):
    """(dict de tensores, offset donde empiezan los datos)."""
    with open(ruta, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cab = json.loads(f.read(n))
    cab.pop("__metadata__", None)
    return cab, n + 8


def cabecera_remota(url):
    def rango(a, b):
        return subprocess.run(["curl", "-sL", "-H", f"Range: bytes={a}-{b}", url],
                              capture_output=True).stdout
    n = int.from_bytes(rango(0, 7), "little")
    cab = json.loads(rango(8, 7 + n))
    cab.pop("__metadata__", None)
    return cab, n + 8


def a_f32(crudo, dtype):
    if dtype == "BF16":  # ascenso exacto: bf16 son los 16 bits altos de un f32
        return (np.frombuffer(crudo, "<u2").astype("<u4") << 16).view("<f4")
    if dtype == "F32":
        return np.frombuffer(crudo, "<f4")
    raise ValueError(f"dtype no contemplado: {dtype}")


def tensores_locales(ficheros, filtro):
    """{clave sin prefijo: tensor f32} leyendo solo lo que hace falta."""
    fuera = {}
    for r in ficheros:
        cab, base = cabecera_safetensors(r)
        with open(r, "rb") as f:
            for k, v in cab.items():
                if filtro not in k:
                    continue
                a, b = v["data_offsets"]
                f.seek(base + a)
                x = a_f32(f.read(b - a), v["dtype"]).reshape(v["shape"])
                fuera[k.split(filtro)[1].lstrip(".")] = torch.from_numpy(x.copy())
    return fuera


def encoder_comunitario(cache: Path):
    """Los 276 tensores del encoder, por Range. Se cachean en disco."""
    crudo = cache / "encoder-comunitario.bin"
    meta = cache / "encoder-comunitario.json"
    if not (crudo.exists() and meta.exists()):
        cache.mkdir(parents=True, exist_ok=True)
        cab, base = cabecera_remota(URL_COM)
        enc = {k: v for k, v in cab.items() if "acoustic_tokenizer.encoder" in k}
        lo = min(v["data_offsets"][0] for v in enc.values())
        hi = max(v["data_offsets"][1] for v in enc.values())
        ajenos = [k for k, v in cab.items()
                  if "acoustic_tokenizer.encoder" not in k and v["data_offsets"][0] < hi]
        assert not ajenos, f"el rango del encoder no es limpio: {len(ajenos)} tensores ajenos"
        print(f"  bajando {len(enc)} tensores ({(hi-lo)/2**20:.0f} MB) en una peticion...")
        r = subprocess.run(["curl", "-sL", "-H",
                            f"Range: bytes={base+lo}-{base+hi-1}", URL_COM,
                            "-o", str(crudo)])
        assert r.returncode == 0 and crudo.stat().st_size == hi - lo, "descarga incompleta"
        meta.write_text(json.dumps(enc))
    enc = json.loads(meta.read_text())
    buf = np.memmap(crudo, dtype=np.uint8, mode="r")
    fuera = {}
    for k, v in enc.items():
        a, b = v["data_offsets"]
        x = a_f32(buf[a:b].tobytes(), v["dtype"]).reshape(v["shape"])
        fuera[k.split("acoustic_tokenizer.encoder")[1].lstrip(".")] = torch.from_numpy(x.copy())
    return fuera


def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        assert w.getframerate() == RITMO and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return x


def escribir_wav(ruta, x):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def tono(x, ritmo=RITMO):
    """Tono fundamental por autocorrelacion. Copiado de scripts/sondeo_voz.py
    para que este script no dependa del resto del arbol."""
    ventana, salto = 1024, 256
    minimo, maximo = ritmo // 400, ritmo // 60
    f0 = []
    for i in range(0, len(x) - ventana, salto):
        t = x[i:i + ventana]
        if np.sqrt(np.mean(t ** 2)) < 0.01:
            continue
        t = t - t.mean()
        r = np.correlate(t, t, mode="full")[ventana - 1:]
        if r[0] <= 0:
            continue
        pico = int(np.argmax(r[minimo:maximo])) + minimo
        if r[pico] / r[0] > 0.3:
            f0.append(ritmo / pico)
    if len(f0) < 5:
        return 0.0
    return float(np.median(np.array(f0)))


# ------------------------------------------------------------------ modelo --
def construir_tokenizador(config_json, encoder, decoder, dispositivo="cpu"):
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel

    cfg = VibeVoiceAcousticTokenizerConfig(**config_json)
    modelo = VibeVoiceAcousticTokenizerModel(cfg)
    faltan_e = modelo.encoder.load_state_dict(encoder, strict=True)
    faltan_d = modelo.decoder.load_state_dict(decoder, strict=True)
    assert not faltan_e.missing_keys and not faltan_d.missing_keys
    return modelo.to(dispositivo).eval().float()


@torch.no_grad()
def ciclo(modelo, x, dispositivo="cpu"):
    """audio -> z -> audio'. Devuelve (audio', z)."""
    a = torch.from_numpy(x)[None, None, :].to(dispositivo).float()
    z = modelo.encode(a).mean          # la media, sin el ruido de sample()
    y = modelo.decode(z).squeeze().cpu().numpy()
    return y, z.squeeze(0).cpu().numpy()


def comparar(x, y):
    n = min(len(x), len(y))
    a, b = x[:n], y[:n]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return {"corr": corr,
            "rms_ent": float(np.sqrt(np.mean(a ** 2))),
            "rms_sal": float(np.sqrt(np.mean(b ** 2))),
            "hz_ent": tono(a), "hz_sal": tono(b)}


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    ap.add_argument("--audio", default="ejemplos-voces")
    ap.add_argument("--guardar", default="", help="directorio donde dejar los wav del ciclo")
    ap.add_argument("--dispositivo", default="cpu")
    args = ap.parse_args()

    modelo_dir = Path(args.modelo)
    cfg = json.loads((modelo_dir / "config.json").read_text())["acoustic_tokenizer_config"]
    pesos05 = [str(modelo_dir / "model.safetensors")]
    pesos15 = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                                   "models--microsoft--VibeVoice-1.5B/snapshots/*/model-*.safetensors")))
    if not pesos15:
        print("falta el 1.5B en la cache del hub: sin el no hay controles", file=sys.stderr)
        return 2

    print("== pesos ==")
    dec05 = tensores_locales(pesos05, "acoustic_tokenizer.decoder")
    print(f"  decoder 0.5B        {len(dec05):3d} tensores")
    enc15 = tensores_locales(pesos15, "acoustic_tokenizer.encoder")
    dec15 = tensores_locales(pesos15, "acoustic_tokenizer.decoder")
    print(f"  encoder/decoder 1.5B {len(enc15):3d} / {len(dec15):3d} tensores")
    encCom = encoder_comunitario(Path(args.cache))
    print(f"  encoder comunitario {len(encCom):3d} tensores")

    n_par = sum(v.numel() for v in encCom.values())
    identicos = sum(1 for k in encCom if k in enc15
                    and torch.equal(encCom[k], enc15[k]))
    difs = np.array([float((encCom[k] - enc15[k]).norm() / (enc15[k].norm() + 1e-30))
                     for k in encCom if k in enc15])
    tam = np.array([encCom[k].numel() for k in encCom if k in enc15], float)
    print(f"\n== el encoder comunitario contra el del 1.5B ==")
    print(f"  parametros                  {n_par/1e6:.1f} M")
    print(f"  identicos byte a byte       {identicos}/{len(encCom)}")
    print(f"  dif. relativa (mediana)     {np.median(difs):.3f}")
    print(f"  dif. relativa (ponderada)   {float(difs @ (tam/tam.sum())):.3f}")
    print("  -> 0 identicos y todo movido = NO es el del 1.5B copiado")

    brazos = [
        ("comunitario  enc(com) -> dec(0.5B)", encCom, dec05),
        ("negativo     enc(1.5B) -> dec(0.5B)", enc15, dec05),
        ("positivo     enc(1.5B) -> dec(1.5B)", enc15, dec15),
    ]
    wavs = sorted(Path(args.audio).glob("sp-*.wav"))
    if not wavs:
        print(f"no hay audio en {args.audio}", file=sys.stderr)
        return 2
    print(f"\n== ciclo sobre {len(wavs)} voces espanolas ==")

    for nombre, enc, dec in brazos:
        tok = construir_tokenizador(cfg, enc, dec, args.dispositivo)
        print(f"\n{nombre}")
        print(f"  {'voz':18} {'corr':>8} {'RMS ent':>9} {'RMS sal':>9} {'Hz ent':>7} {'Hz sal':>7}")
        acumulado = []
        for w in wavs:
            x = leer_wav(w)
            y, _ = ciclo(tok, x, args.dispositivo)
            m = comparar(x, y)
            acumulado.append(m["corr"])
            print(f"  {w.stem:18} {m['corr']:8.4f} {m['rms_ent']:9.4f} {m['rms_sal']:9.4f}"
                  f" {m['hz_ent']:7.1f} {m['hz_sal']:7.1f}")
            if args.guardar:
                d = Path(args.guardar); d.mkdir(parents=True, exist_ok=True)
                escribir_wav(d / f"{w.stem}--{nombre.split()[0]}.wav", y)
        print(f"  {'MEDIA':18} {np.mean(acumulado):8.4f}")
        del tok

    print("\nReferencia publicada (docs/clonado-de-voz.md §4.2):")
    print("  positivo enc(1.5B)->dec(1.5B)  corr +0,9799")
    print("  negativo enc(1.5B)->dec(0.5B)  corr +0,0063, RMS 0,0083 frente a 0,0591")
    return 0


if __name__ == "__main__":
    sys.exit(main())
