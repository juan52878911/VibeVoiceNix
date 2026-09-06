#!/usr/bin/env python
"""Banco de comparacion entre motores TTS con la MISMA voz y las MISMAS frases.

    # 1. sintetizar, un motor por corrida (cada motor vive en su propio entorno)
    python scripts/banco_motores.py sintetizar --motor qwen3 \
        --referencia nota.opus --transcripcion "lo que dice la nota" \
        --salida banco/qwen3
    python scripts/banco_motores.py sintetizar --motor vibevoice \
        --voz-vibevoice juan --referencia nota.opus --salida banco/vibevoice
    python scripts/banco_motores.py sintetizar --motor http --url http://voz:8082 \
        --voz-http juan --referencia nota.opus --salida banco/vm

    # 2. puntuar (puede correr en OTRA maquina: solo necesita los wav y la referencia)
    python scripts/banco_motores.py puntuar --referencia nota.opus \
        --salida banco/qwen3 banco/vibevoice banco/vm

    # 3. consolidar en una tabla
    python scripts/banco_motores.py consolidar banco/*/puntuacion.csv -o docs/tabla.md

QUE RESPONDE
Si un motor candidato clona MEJOR que el actual, con la misma referencia, y en
que idiomas. No compara voces entre si (ver banco_clonado.py): compara MOTORES
sobre una voz, contra el techo de esa voz.

QUE MIDE
  identidad   coseno ECAPA contra la referencia (mismo juez que oido.py)
  techo       la referencia partida en dos mitades, una contra otra
  WER         faster-whisper en el idioma del texto
  sesgo_st    12*log2(f0_clon / f0_ref): el +2,5 st de las voces graves
  car/s       caracteres por segundo de habla (sin silencios): la deriva de ritmo
  RTF, RSS    coste; en el Mac solo orientativo, la cifra que vale es la de la VM

POR QUE TRES SUBCOMANDOS
Los motores no caben en el mismo proceso ni en el mismo entorno de Python
(vibevoice y qwen-tts fijan torchs distintos). Y puntuar es lo caro en CPU
lenta: separado, el M920q sintetiza mientras el i3 puntua.
"""
import argparse
import csv
import json
import math
import os
import resource
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RITMO = 24000

# es: los dos de banco_clonado.py (sin "¿ ¡": medido que disparan el WER en
# VibeVoice) mas uno largo, que es donde Qwen3 acelera (issue #239) y
# VibeVoice se rompe (550 chars). Los demas idiomas: un neutro y un largo.
TEXTOS = {
    "es": {
        "neutro": "El backup de anoche termino sin errores y los tres servicios responden con normalidad.",
        "expresivo": "En serio? No me lo puedo creer! Eso si que no me lo esperaba para nada, de verdad.",
        "largo": ("El problema no es el precio, es que nadie te explica lo que estas comprando. "
                  "Te ensenan una tabla con veinte filas, te sonrien, y cuando preguntas por la "
                  "letra pequena te dicen que eso ya lo veremos mas adelante, cuando firmes."),
    },
    "en": {
        "neutro": "Last night's backup finished without errors and all three services are responding normally.",
        "largo": ("The problem is not the price, it is that nobody explains what you are buying. "
                  "They show you a table with twenty rows, they smile, and when you ask about the "
                  "fine print they tell you that you will see that later, once you have signed."),
    },
    "pt": {
        "neutro": "O backup de ontem terminou sem erros e os tres servicos respondem normalmente.",
        "largo": ("O problema nao e o preco, e que ninguem explica o que voce esta comprando. "
                  "Mostram uma tabela com vinte linhas, sorriem, e quando voce pergunta pela letra "
                  "miuda dizem que isso a gente ve depois, quando voce assinar."),
    },
    "fr": {
        "neutro": "La sauvegarde de cette nuit s'est terminee sans erreur et les trois services repondent normalement.",
        "largo": ("Le probleme n'est pas le prix, c'est que personne ne vous explique ce que vous achetez. "
                  "On vous montre un tableau de vingt lignes, on vous sourit, et quand vous demandez les "
                  "petites lignes on vous dit qu'on verra ca plus tard, une fois signe."),
    },
    "it": {
        "neutro": "Il backup di stanotte e terminato senza errori e i tre servizi rispondono normalmente.",
        "largo": ("Il problema non e il prezzo, e che nessuno ti spiega cosa stai comprando. "
                  "Ti mostrano una tabella con venti righe, ti sorridono, e quando chiedi delle "
                  "clausole in piccolo ti dicono che lo vedremo piu avanti, quando avrai firmato."),
    },
    "de": {
        "neutro": "Das Backup von letzter Nacht ist ohne Fehler durchgelaufen und alle drei Dienste antworten normal.",
        "largo": ("Das Problem ist nicht der Preis, sondern dass dir niemand erklaert, was du kaufst. "
                  "Man zeigt dir eine Tabelle mit zwanzig Zeilen, laechelt, und wenn du nach dem "
                  "Kleingedruckten fragst, heisst es, das sehen wir spaeter, wenn du unterschrieben hast."),
    },
}

IDIOMA_QWEN = {"es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French",
               "it": "Italian", "de": "German"}


# ---------------------------------------------------------------- utilidades

def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        hz = w.getframerate()
    if hz != RITMO:
        x = remuestrear(x, hz, RITMO)
    return x


def escribir_wav(ruta, x, hz=RITMO):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(hz)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def remuestrear(x, de, a):
    if de == a:
        return x
    n = int(round(len(x) * a / de))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def leer_referencia(ruta):
    """Cualquier formato que entienda ffmpeg -> 24 kHz mono (clonar_voz.leer_audio)."""
    from clonar_voz import leer_audio
    return leer_audio(ruta)


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def duracion_habla(x, hz=RITMO, db_bajo_pico=25.0, marco=240):
    """Segundos con voz: quita silencios de cabeza, cola y pausas largas.
    Es lo que hace comparable el ritmo (car/s) entre motores que rellenan
    distinto al principio y al final.

    El umbral es RELATIVO al nivel del clip (p95 del RMS por marco menos 25
    dB): un umbral absoluto (0,01) contaba como silencio media frase de los
    motores que salen bajos y daba 30-50 car/s, que no dice nadie."""
    k = (len(x) // marco) * marco
    if not k:
        return 0.0
    rms = np.sqrt((x[:k].reshape(-1, marco) ** 2).mean(1))
    pico = np.percentile(rms, 95)
    if pico <= 0:
        return 0.0
    umbral = pico * 10 ** (-db_bajo_pico / 20)
    return float((rms >= umbral).sum() * marco / hz)


# ------------------------------------------------------------------ motores

class MotorQwen3:
    """PyTorch de referencia (qwen-tts). Calidad, no velocidad."""

    def __init__(self, modelo, referencia_wav, transcripcion, solo_xvector, hilos):
        import torch
        from qwen_tts import Qwen3TTSModel
        if hilos:
            torch.set_num_threads(hilos)
        self.torch = torch
        disp = "mps" if (torch.backends.mps.is_available() and
                        os.environ.get("BANCO_DISPOSITIVO", "auto") != "cpu") else "cpu"
        self.m = Qwen3TTSModel.from_pretrained(
            modelo, device_map=disp, dtype=torch.float32, attn_implementation="sdpa")
        kw = {"ref_audio": str(referencia_wav)}
        if transcripcion and not solo_xvector:
            kw["ref_text"] = transcripcion
        else:
            kw["x_vector_only_mode"] = True
        self.prompt = self.m.create_voice_clone_prompt(**kw)
        self.nombre = f"qwen3:{Path(modelo).name}" + (":xvec" if solo_xvector else ":icl")

    def hablar(self, texto, idioma, semilla):
        self.torch.manual_seed(semilla)
        wavs, hz = self.m.generate_voice_clone(
            text=texto, language=IDIOMA_QWEN[idioma], voice_clone_prompt=self.prompt)
        x = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return remuestrear(x, hz, RITMO)


class MotorC:
    """El motor C de Qwen3 (gabriele-mastrapasqua/qwen3-tts): el que iria a la
    VM. Un proceso por clip, con la referencia y la semilla en la orden; la
    carga del modelo (~2-3 s en el Mac) queda fuera del RTF, que se lee de lo
    que imprime el propio motor."""

    def __init__(self, binario, modelo, referencia_wav, cuant, hilos, voz_bin=None):
        import shutil
        self.bin = binario if os.path.exists(binario) else shutil.which(binario)
        if not self.bin:
            raise SystemExit(f"no encuentro el motor C '{binario}' (QWEN3TTS_BIN)")
        self.modelo = modelo
        # Dos formas de dar la voz: la referencia cruda (--ref-audio, injerto
        # ICL) o el x-vector .bin que fabrica clonar_voz_qwen.py, que es lo
        # que sirve el shim y lo que MEJOR clona (0,642 frente a 0,569).
        if voz_bin:
            self.voz = ["--load-voice", str(voz_bin), "--xvector-only"]
            forma = "xvec"
        else:
            self.voz = ["--ref-audio", str(referencia_wav)]
            forma = "icl"
        self.cuant, self.hilos = cuant, hilos or 4
        self.nombre = f"qwen3-c:{Path(modelo).name}:{cuant}:{forma}"
        self.ultimo_rtf = None

    def hablar(self, texto, idioma, semilla):
        import re
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp) / "clip.wav"
            orden = [self.bin, "-d", self.modelo, *self.voz,
                     "-l", IDIOMA_QWEN[idioma], f"--{self.cuant}", "-j", str(self.hilos),
                     "--seed", str(semilla), "--text", texto, "-o", str(salida)]
            r = subprocess.run(orden, capture_output=True, text=True)
            if r.returncode != 0 or not salida.exists():
                raise RuntimeError((r.stderr or r.stdout)[-300:])
            m = re.search(r"RTF ([0-9.]+)", r.stdout + r.stderr)
            self.ultimo_rtf = float(m.group(1)) if m else None
            return leer_wav(salida)


class MotorVibeVoice:
    """El motor actual, desnudo (sin los parches de voz_stream.py), como en
    banco_duracion.py. La voz es un prefijo .pt ya fabricado con clonar_voz.py
    a partir de la MISMA referencia."""

    def __init__(self, voz, modelo_dir, cache, pasos, cfg, hilos):
        import torch
        from banco_duracion import cargar_modelo, hablar
        if hilos:
            torch.set_num_threads(hilos)
        self._hablar = hablar
        self.disp = "cpu"
        self.proc, self.modelo = cargar_modelo(modelo_dir, cache, self.disp, pasos,
                                               con_encoder=False)
        ruta = Path(voz) if Path(voz).exists() else Path(cache) / "voces" / f"{voz}.pt"
        self.prefijo = torch.load(ruta, map_location="cpu", weights_only=False)
        self.cfg = cfg
        self.nombre = "vibevoice:0.5B"

    def hablar(self, texto, idioma, semilla):
        # VibeVoice no tiene parametro de idioma: el prefijo manda y el texto
        # va tal cual. Eso es justo lo que se quiere medir en el bloque cruzado.
        return self._hablar(self.proc, self.modelo, self.prefijo, texto, self.cfg,
                            semilla, self.disp)


class MotorHTTP:
    """Cualquier servidor que hable POST /tts/stream (voz_stream.py o el shim).
    Mide el motor de PRODUCCION tal cual sirve, parches incluidos."""

    def __init__(self, url, token, voz):
        import urllib.request
        self.urllib = urllib.request
        self.url, self.token, self.voz = url.rstrip("/"), token, voz
        salud = json.load(urllib.request.urlopen(f"{self.url}/health", timeout=30))
        self.nombre = "http:" + str(salud.get("motor", "?"))

    def hablar(self, texto, idioma, semilla):
        cuerpo = {"texto": texto, "voz": self.voz, "semilla": semilla, "idioma": idioma}
        pet = self.urllib.Request(
            f"{self.url}/tts/stream", method="POST", data=json.dumps(cuerpo).encode(),
            headers={"content-type": "application/json",
                     **({"authorization": f"Bearer {self.token}"} if self.token else {})})
        datos = self.urllib.urlopen(pet, timeout=900).read()
        return np.frombuffer(datos[44:], "<i2").astype(np.float32) / 32768


# ------------------------------------------------------------- subcomandos

def sintetizar(args):
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    ref = leer_referencia(args.referencia)
    ref_wav = salida / "referencia.wav"
    escribir_wav(ref_wav, ref)

    if args.motor == "qwen3":
        motor = MotorQwen3(args.qwen_modelo, ref_wav, args.transcripcion,
                           args.solo_xvector, args.hilos)
    elif args.motor == "c":
        motor = MotorC(args.bin, args.qwen_modelo, ref_wav, args.cuant, args.hilos,
                       args.voz_bin)
    elif args.motor == "vibevoice":
        if not args.voz_vibevoice:
            raise SystemExit("--voz-vibevoice es obligatorio con --motor vibevoice")
        motor = MotorVibeVoice(args.voz_vibevoice, args.vibevoice_modelo, args.cache,
                               args.pasos, args.cfg, args.hilos)
    else:
        if not args.url or not args.voz_http:
            raise SystemExit("--url y --voz-http son obligatorios con --motor http")
        motor = MotorHTTP(args.url, args.token, args.voz_http)

    rss_carga = rss_mb()
    filas = []
    for idioma in args.idiomas:
        for clave, texto in TEXTOS[idioma].items():
            for semilla in args.semillas:
                nombre = f"{idioma}_{clave}_{semilla}.wav"
                t0 = time.perf_counter()
                try:
                    x = motor.hablar(texto, idioma, semilla)
                except Exception as e:            # un fallo no tira el banco entero
                    print(f"  [fallo] {nombre}: {e}")
                    filas.append({"fichero": nombre, "idioma": idioma, "texto": clave,
                                  "semilla": semilla, "error": str(e)[:120]})
                    continue
                dt = time.perf_counter() - t0
                escribir_wav(salida / nombre, x)
                dur = len(x) / RITMO
                # El motor C cuenta su RTF sin la carga del modelo (que en el
                # servidor se paga una vez); los demas motores ya estan cargados.
                rtf = getattr(motor, "ultimo_rtf", None) or (dt / dur if dur else None)
                filas.append({"fichero": nombre, "idioma": idioma, "texto": clave,
                              "semilla": semilla, "dur_s": round(dur, 2),
                              "rtf": round(rtf, 3) if rtf else None,
                              "segundos": round(dt, 1)})
                print(f"  {nombre:28s} {dur:5.1f} s  RTF {dt / dur if dur else 0:.2f}")
    meta = {"motor": motor.nombre, "argumentos": {k: v for k, v in vars(args).items()
                                                  if k not in ("func",)},
            "rss_carga_mb": round(rss_carga), "rss_pico_mb": round(rss_mb()),
            "plataforma": sys.platform, "filas": filas}
    (salida / "sintesis.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"\n{motor.nombre}: {len(filas)} clips en {salida}  "
          f"(RSS carga {rss_carga:.0f} MB, pico {rss_mb():.0f} MB)")


def puntuar(args):
    from banco_clonado import Huella, wer
    from prosodia import descripcion
    from faster_whisper import WhisperModel

    huella = Huella()
    whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8")

    ref = leer_referencia(args.referencia)
    h_ref = huella(ref)
    mitad = len(ref) // 2
    techo = float(huella(ref[:mitad]) @ huella(ref[mitad:]))
    f0_ref = descripcion(ref)["hz"]
    print(f"referencia: {len(ref) / RITMO:.1f} s, techo ECAPA {techo:.3f}, f0 {f0_ref:.0f} Hz")

    for carpeta in args.salida:
        carpeta = Path(carpeta)
        meta = json.loads((carpeta / "sintesis.json").read_text())
        filas = []
        for f in meta["filas"]:
            if "error" in f:
                filas.append({**f, "motor": meta["motor"]})
                continue
            x = leer_wav(carpeta / f["fichero"])
            texto = TEXTOS[f["idioma"]][f["texto"]]
            segs, _ = whisper.transcribe(str(carpeta / f["fichero"]), language=f["idioma"],
                                         beam_size=5)
            hip = " ".join(s.text for s in segs)
            f0 = descripcion(x)["hz"]
            habla = duracion_habla(x)
            filas.append({
                **f, "motor": meta["motor"],
                "ecapa": round(float(huella(x) @ h_ref), 3),
                "wer": round(wer(texto, hip), 3),
                "sesgo_st": round(12 * math.log2(f0 / f0_ref), 2) if f0 and f0_ref else None,
                "car_s": round(len(texto) / habla, 1) if habla else None,
                "habla_s": round(habla, 2),
            })
            print(f"  {f['fichero']:28s} ECAPA {filas[-1]['ecapa']:.3f}  WER {filas[-1]['wer']:.3f}"
                  f"  st {filas[-1]['sesgo_st']:+.1f}  {filas[-1]['car_s']} car/s")
        campos = ["motor", "fichero", "idioma", "texto", "semilla", "dur_s", "habla_s", "rtf",
                  "ecapa", "wer", "sesgo_st", "car_s", "error"]
        with open(carpeta / "puntuacion.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
            w.writeheader(); w.writerows(filas)
        (carpeta / "techo.json").write_text(json.dumps(
            {"techo_ecapa": techo, "f0_ref": f0_ref, "rss_carga_mb": meta["rss_carga_mb"],
             "rss_pico_mb": meta["rss_pico_mb"], "plataforma": meta["plataforma"]}))
        print(f"-> {carpeta / 'puntuacion.csv'}")


def consolidar(args):
    """Una fila por (motor, idioma): medias y peor caso. Lo que va al doc."""
    grupos = {}
    techos = {}
    for ruta in args.csv:
        ruta = Path(ruta)
        try:
            techos[ruta.parent] = json.loads((ruta.parent / "techo.json").read_text())
        except FileNotFoundError:
            pass
        with open(ruta) as fh:
            for f in csv.DictReader(fh):
                grupos.setdefault((f["motor"], f["idioma"]), []).append(f)

    def num(f, k):
        try:
            return float(f[k])
        except (KeyError, ValueError, TypeError):
            return None

    lineas = ["| motor | idioma | n | fallos | ECAPA media | ECAPA min | WER media | WER peor | sesgo st | car/s | RTF |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (motor, idioma), filas in sorted(grupos.items()):
        ok = [f for f in filas if not f.get("error")]
        col = lambda k: [v for v in (num(f, k) for f in ok) if v is not None]
        e, w, st, cs, rtf = col("ecapa"), col("wer"), col("sesgo_st"), col("car_s"), col("rtf")
        lineas.append(
            f"| {motor} | {idioma} | {len(filas)} | {len(filas) - len(ok)} | "
            f"{np.mean(e):.3f} | {np.min(e):.3f} | {np.mean(w):.1%} | {np.max(w):.1%} | "
            f"{np.mean(st):+.1f} | {np.mean(cs):.1f} | {np.mean(rtf):.2f} |"
            if ok else f"| {motor} | {idioma} | {len(filas)} | {len(filas)} | - | - | - | - | - | - | - |")
    cabecera = []
    for carpeta, t in techos.items():
        cabecera.append(f"- `{carpeta}`: techo ECAPA de la referencia {t['techo_ecapa']:.3f}, "
                        f"f0 {t['f0_ref']:.0f} Hz, RSS carga {t['rss_carga_mb']} MB / "
                        f"pico {t['rss_pico_mb']} MB ({t['plataforma']})")
    md = "\n".join(cabecera + [""] + lineas) + "\n"
    if args.o:
        Path(args.o).write_text(md)
        print(f"-> {args.o}")
    print(md)


def referencia(args):
    """Saca de un taller de dobla la referencia de UN hablante: los segmentos
    anotados por el humano (anotaciones/<video>.json), en orden, hasta --max
    segundos, y el texto literal al lado. Es la misma voz que se dobla, asi
    que el banco mide lo que el doblaje va a sufrir."""
    import wave as _w
    taller = Path(args.taller)
    fuente = taller / ("voces24k.wav" if (taller / "voces24k.wav").exists() else "audio24k.wav")
    x = leer_wav(fuente)
    anot = json.loads(Path(args.anotacion).read_text())
    segs = [s for s in anot["segmentos"] if str(s.get("hablante")) == str(args.hablante)
            and s.get("texto", "").strip()]
    segs.sort(key=lambda s: float(s["ini"]))
    trozos, textos, total = [], [], 0.0
    hueco = np.zeros(int(0.2 * RITMO), np.float32)
    for s in segs:
        ini, fin = float(s["ini"]), float(s["fin"])
        if fin - ini < 1.0:
            continue
        if total + (fin - ini) > args.max:
            break
        trozos += [x[int(ini * RITMO):int(fin * RITMO)], hueco]
        textos.append(s["texto"].strip())
        total += fin - ini
    if not trozos:
        raise SystemExit("ningun segmento valido para ese hablante")
    salida = Path(args.o)
    escribir_wav(salida, np.concatenate(trozos))
    salida.with_suffix(".txt").write_text(" ".join(textos) + "\n")
    print(f"{salida}: {total:.1f} s de {len(textos)} segmentos; texto en {salida.with_suffix('.txt')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sintetizar")
    s.add_argument("--motor", choices=["qwen3", "c", "vibevoice", "http"], required=True)
    s.add_argument("--referencia", required=True, help="audio de la voz (opus, wav, m4a...)")
    s.add_argument("--transcripcion", default=None,
                   help="texto literal de la referencia (modo en contexto de Qwen3)")
    s.add_argument("--salida", required=True)
    s.add_argument("--idiomas", nargs="+", default=list(TEXTOS), choices=list(TEXTOS))
    s.add_argument("--semillas", type=int, nargs="+", default=[11, 42, 101])
    s.add_argument("--hilos", type=int, default=0, help="0 = lo que decida torch")
    # qwen3
    s.add_argument("--qwen-modelo", default=os.environ.get(
        "QWEN3TTS_MODELO", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"))
    s.add_argument("--solo-xvector", action="store_true",
                   help="clonar solo por x-vector, sin transcripcion (peor, pero sin texto)")
    # motor c
    s.add_argument("--bin", default=os.environ.get("QWEN3TTS_BIN", "qwen_tts"))
    s.add_argument("--cuant", choices=["int8", "int4"], default="int8")
    s.add_argument("--voz-bin", default=None,
                   help="x-vector .bin de clonar_voz_qwen.py; sin el, --ref-audio (injerto ICL)")
    # vibevoice
    s.add_argument("--voz-vibevoice", help="nombre del .pt (en --cache/voces) o ruta")
    s.add_argument("--vibevoice-modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    s.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    s.add_argument("--pasos", type=int, default=6)
    s.add_argument("--cfg", type=float, default=3.0)
    # http
    s.add_argument("--url"); s.add_argument("--token", default=os.environ.get("VOZ_TOKEN"))
    s.add_argument("--voz-http")
    s.set_defaults(func=sintetizar)

    p = sub.add_parser("puntuar")
    p.add_argument("--referencia", required=True)
    p.add_argument("--salida", nargs="+", required=True, help="carpetas de sintetizar")
    p.add_argument("--whisper", default="small")
    p.set_defaults(func=puntuar)

    r = sub.add_parser("referencia")
    r.add_argument("--taller", required=True, help="carpeta talleres/<video> de dobla")
    r.add_argument("--anotacion", required=True, help="anotaciones/<video>.json de dobla")
    r.add_argument("--hablante", required=True)
    r.add_argument("--max", type=float, default=30.0, help="segundos de referencia")
    r.add_argument("-o", required=True, help="wav de salida (el .txt va al lado)")
    r.set_defaults(func=referencia)

    c = sub.add_parser("consolidar")
    c.add_argument("csv", nargs="+")
    c.add_argument("-o", default=None)
    c.set_defaults(func=consolidar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
