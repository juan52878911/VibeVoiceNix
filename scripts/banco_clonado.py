#!/usr/bin/env python
"""Banco de pruebas para afinar una voz clonada.

    python scripts/banco_clonado.py --voz santiago --referencia santi.wav
    python scripts/banco_clonado.py --voz isis --referencia isis.wav \
        --cfg 3.0 3.5 --semillas 11 42 101 --csv banco.csv

QUE RESPONDE
Con una voz ya clonada, cual es la mejor configuracion de inferencia para ELLA.
No compara voces entre si: los numeros de identidad NO son comparables entre
locutores, porque cada uno tiene su propio techo (la auto-similitud de su
grabacion consigo misma), y ese techo va de 0,655 a 0,862 en las tres voces
reales medidas hasta ahora. Cada voz se juzga contra el suyo.

POR QUE SEMILLAS FIJAS, Y POR QUE VARIAS
El ruido de la difusion se sortea en cada generacion y no es un detalle: entre
tres generaciones de la misma voz con el mismo texto se han medido diferencias
de varios semitonos de recorrido tonal. Esa varianza es del ORDEN del efecto
que se busca al mover cfg. Un banco con una sola semilla por celda mide loteria.

QUE MIDE, Y QUE NO
  identidad   coseno ECAPA contra la grabacion de referencia (speechbrain, el
              mismo modelo y umbrales que scripts/oido.py). Mide TIMBRE.
  prosodia    tono, recorrido, desviacion y movimiento de scripts/prosodia.py,
              con las octavas corregidas. Son DESCRIPTORES, no una distancia:
              la metrica de distancia de melodia se probo y NO valida (ver el
              docstring de prosodia.py). Aqui se listan para ver la tendencia,
              no para declarar un ganador por si solos.
  WER         faster-whisper contra el texto pedido. Es el guardarrail: una
              configuracion que mejora el timbre y sube el WER no vale.

CALIBRACION AUTOMATICA
Antes de nada parte la grabacion de referencia por la mitad y mide una mitad
contra la otra. Ese numero es el techo realista de esa voz en esas condiciones,
y sale impreso arriba del todo para que las cifras de abajo se lean contra el.

LO QUE ESTE BANCO NO TOCA
`VIBEVOICE_FRENO_GUIA`, `CFG_ARRANQUE` y el aplazamiento del EOS son parches
que voz_stream.py aplica al modelo al cargarlo; aqui el modelo va SIN ellos.
O sea que estas medidas son del modelo desnudo, y no predicen exactamente lo
que hara el servicio de produccion. Para medir esos factores hay que instrumentar
voz_stream.py, no este banco.

COSTE
RTF ~1 en un M4. cada celda = una generacion. voces x textos x cfg x semillas.
Lo de por defecto (1 voz, 2 textos, 2 cfg, 3 semillas = 12 generaciones de ~6 s)
son unos 2 minutos mas 40 s de carga del modelo.
"""
import argparse
import copy
import csv
import os
import re
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prosodia import descripcion, leer_wav  # noqa: E402

RITMO = 24000

# CUIDADO CON LOS SIGNOS INVERTIDOS DEL ESPANOL
# La primera version del texto expresivo empezaba por "¿En serio? ¡No me lo
# puedo creer!" y el modelo se atragantaba. MEDIDO sobre la misma frase con y
# sin los signos de apertura, misma voz, misma semilla y mismo cfg:
#
#     voz         con ¿ ¡     sin ellos
#     santiago      0,722       ~0,00
#     isis          0,444        0,000
#     juan          0,222        0,000
#
# Es coherente con lo que se sabe del modelo: se entreno sobre todo en ingles y
# chino, y "¿" y "¡" son tokens que apenas ha visto. Quitar solo los de
# APERTURA basta -- los de cierre "?" y "!" no dan problema y son los que
# llevan la entonacion. Si escribes textos de prueba propios, evita "¿" y "¡".
TEXTOS = {
    "neutro": "El backup de anoche termino sin errores y los tres servicios responden con normalidad.",
    "expresivo": "En serio? No me lo puedo creer! Eso si que no me lo esperaba para nada, de verdad.",
}


def escribir_wav(ruta, x):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def normalizar(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", t).split()


def wer(ref, hip):
    r, h = normalizar(ref), normalizar(hip)
    if not r:
        return float("nan")
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1][-1] / len(r)


class Huella:
    """ECAPA bajo demanda: cargarlo cuesta ~6 s y no siempre hace falta."""

    def __init__(self):
        self._m = None

    def __call__(self, x, hz=RITMO):
        if self._m is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._m = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
                run_opts={"device": "cpu"})
        import torchaudio
        t = torch.from_numpy(np.ascontiguousarray(x))[None]
        if hz != 16000:
            t = torchaudio.functional.resample(t, hz, 16000)
        with torch.no_grad():
            e = self._m.encode_batch(t).squeeze()
        return (e / e.norm()).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz", required=True, help="nombre del .pt instalado")
    ap.add_argument("--referencia", required=True,
                    help="WAV de 24 kHz del que salio la voz; da la calibracion")
    ap.add_argument("--cfg", type=float, nargs="+", default=[3.0, 3.5])
    ap.add_argument("--semillas", type=int, nargs="+", default=[11, 42, 101])
    ap.add_argument("--textos", nargs="+", default=list(TEXTOS),
                    help=f"claves de {list(TEXTOS)} o frases literales")
    ap.add_argument("--pasos", type=int, default=10)
    ap.add_argument("--salida", default="banco", help="directorio para los WAV")
    ap.add_argument("--csv", default="", help="fichero CSV con todas las filas")
    ap.add_argument("--sin-wer", action="store_true", help="salta la transcripcion")
    ap.add_argument("--voces", default=os.environ.get(
        "VIBEVOICE_VOCES", str(Path.home() / ".cache/vibevoice-nix/voces")))
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    args = ap.parse_args()

    textos = {k: TEXTOS.get(k, k) for k in args.textos}
    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
    huella = Huella()

    # ---------------------------------------------------------- calibracion --
    ref, hz_ref = leer_wav(args.referencia)
    m = len(ref) // 2
    techo = float(huella(ref[:m], hz_ref) @ huella(ref[m:], hz_ref))
    h_ref = huella(ref, hz_ref)
    d_ref = descripcion(ref, hz_ref)
    print(f"REFERENCIA {args.referencia}")
    print(f"  {len(ref)/hz_ref:.1f} s | tono {d_ref['hz']:.1f} Hz | recorrido "
          f"{d_ref['recorrido']:.1f} st | desviacion {d_ref['desviacion']:.2f}")
    print(f"  TECHO de esta voz (sus dos mitades entre si): {techo:.4f}")
    print(f"  umbral generico de oido.py: mismo locutor >= 0,626\n")

    # --------------------------------------------------------------- modelo --
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"cargando el modelo en {disp}...", flush=True)
    proc = VibeVoiceStreamingProcessor.from_pretrained(args.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.modelo, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    modelo.model.tts_language_model.embed_tokens = modelo.model.language_model.embed_tokens
    modelo.set_ddpm_inference_steps(args.pasos)
    modelo.to(disp)
    prefijo = torch.load(Path(args.voces) / f"{args.voz}.pt",
                         weights_only=False, map_location=disp)

    transcribe = None
    if not args.sin_wer:
        try:
            from faster_whisper import WhisperModel
            w = WhisperModel("base", device="cpu", compute_type="int8")
            def transcribe(ruta):
                seg, _ = w.transcribe(str(ruta), language="es", beam_size=5)
                return " ".join(s.text for s in seg).strip()
        except ImportError:
            print("[aviso] sin faster-whisper: no habra WER")

    filas = []
    print(f"\n{'texto':10} {'cfg':>4} {'semilla':>8} {'dur':>6} {'ECAPA':>7} "
          f"{'tono':>7} {'recorr':>7} {'desv':>6} {'WER':>6}")
    for clave, texto in textos.items():
        for cfg in args.cfg:
            for semilla in args.semillas:
                torch.manual_seed(semilla)
                t = texto if texto.endswith("\n") else texto + "\n"
                ent = proc.process_input_with_cached_prompt(
                    text=t, cached_prompt=copy.deepcopy(prefijo), padding=True,
                    return_tensors="pt", return_attention_mask=True)
                ent = {k: (v.to(disp) if torch.is_tensor(v) else v) for k, v in ent.items()}
                with torch.no_grad():
                    s = modelo.generate(
                        **ent, max_new_tokens=None, cfg_scale=cfg, tokenizer=proc.tokenizer,
                        generation_config={"do_sample": False}, verbose=False,
                        return_speech=True, all_prefilled_outputs=copy.deepcopy(prefijo))
                x = s.speech_outputs[0].detach().float().cpu().numpy().reshape(-1)
                ruta = salida / f"{args.voz}-{clave}-cfg{cfg}-s{semilla}.wav"
                escribir_wav(ruta, x)
                d = descripcion(x)
                ec = float(huella(x) @ h_ref)
                e = wer(texto, transcribe(ruta)) if transcribe else float("nan")
                filas.append(dict(voz=args.voz, texto=clave, cfg=cfg, semilla=semilla,
                                  duracion=len(x) / RITMO, ecapa=ec, wer=e, **d))
                print(f"{clave:10} {cfg:4.1f} {semilla:8d} {len(x)/RITMO:6.2f} {ec:7.4f} "
                      f"{d['hz']:7.1f} {d['recorrido']:7.1f} {d['desviacion']:6.2f} "
                      f"{e:6.3f}", flush=True)

    # --------------------------------------------------------------- resumen --
    print(f"\nRESUMEN por cfg (media de {len(args.semillas)} semillas x "
          f"{len(textos)} textos), techo {techo:.4f}")
    print(f"{'cfg':>5} {'ECAPA':>8} {'+-':>6} {'recorr':>8} {'+-':>6} {'WER':>7}")
    for cfg in args.cfg:
        v = [f for f in filas if f["cfg"] == cfg]
        ec = np.array([f["ecapa"] for f in v]); rc = np.array([f["recorrido"] for f in v])
        we = np.array([f["wer"] for f in v])
        print(f"{cfg:5.1f} {ec.mean():8.4f} {ec.std():6.4f} {rc.mean():8.1f} "
              f"{rc.std():6.1f} {np.nanmean(we):7.3f}")
    print("\nOJO: si la desviacion entre semillas es del tamano de la diferencia\n"
          "entre cfg, ese factor NO esta decidido. Sube --semillas antes de concluir.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(filas[0]))
            wcsv.writeheader(); wcsv.writerows(filas)
        print(f"\n{len(filas)} filas en {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
