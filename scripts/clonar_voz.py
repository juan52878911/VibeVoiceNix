#!/usr/bin/env python
"""Fabrica un prefijo de voz `.pt` para el Realtime-0.5B a partir de un audio.

    python scripts/clonar_voz.py --audio mi_voz.wav \
        --transcripcion "la transcripcion literal de ese audio" \
        --salida ~/.cache/vibevoice-nix/voces/mi_voz.pt

Es la pieza que faltaba. El formato del prefijo ya estaba resuelto en
docs/clonado-de-voz.md; lo que no habia era manera de sacar los latentes `z` de
un audio nuevo, porque el 0.5B se publico SIN codificador acustico. Este script
usa el codificador de la comunidad
(mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder, MIT), auditado en
scripts/auditar_encoder.py.

RESULTADO MEDIDO sobre las seis voces espanolas (huella ECAPA, el mismo modelo
y los mismos umbrales que scripts/oido.py: mismo locutor >= 0,626):

    clon contra SU voz oficial     media 0,850   minimo 0,803
    clon contra OTRA voz           media 0,176   maximo 0,404

No hay solape: el hueco entre 0,803 y 0,404 es mas ancho que el que separa a
dos locutores distintos. WER 0,000 en la frase de prueba.

DOS DETALLES QUE DECIDEN SI ESTO FUNCIONA O NO

1. La rama `tts_lm` va con N+M posiciones y SIN marcadores -- los N latentes
   primero, los M tokens de texto despues, todo con tipo 0. Es la receta que
   docs/clonado-de-voz.md §3 recupero invirtiendo la cache KV de las voces
   oficiales, con coseno >= 0,9993.

   El `make_voice_prompt.py` del repo comunitario mete ademas un
   `<|vision_start|>` delante y un `<|vision_end|>` detras del bloque de
   latentes (N+2+M). MEDIDO: con esos dos marcadores el clon sigue siendo
   inteligible (WER 0,000) y conserva el tono, pero PIERDE la identidad --
   ECAPA 0,410 frente a 0,864 con la receta de aqui, sobre exactamente el mismo
   audio, la misma frase y la misma semilla. Dos posiciones de mas y deja de
   ser la misma persona.

2. La transcripcion tiene que ser LITERAL. La rama `lm` es el prefill del texto
   que se oye en la referencia; si no coincide, el modelo arrastra un
   condicionamiento que no corresponde al audio.

VARIAS MUESTRAS DE LA MISMA VOZ
`--audio` y `--transcripcion` se pueden repetir, emparejados y en el mismo
orden. Cada clip se codifica POR SEPARADO y luego se concatenan los latentes;
las transcripciones se concatenan en el mismo orden en la rama de texto. Se
codifica por separado a proposito: pegar las ondas primero mete un salto
artificial en cada empalme, y el encoder es convolucional, asi que ese salto se
le cuela en la ventana receptiva y ensucia los latentes de alrededor.

    python scripts/clonar_voz.py \
        --audio a.wav --transcripcion "lo que dice a" \
        --audio b.wav --transcripcion "lo que dice b" \
        --audio c.wav --transcripcion "lo que dice c" \
        --salida mi_voz.pt

El prefijo entero viaja en CADA generacion, asi que mas audio cuesta memoria y
latencia en todas las frases, no solo al fabricarlo. A 7,5 latentes por segundo,
60 s de referencia son 450 posiciones de las 8192 de contexto.

OJO: MAS AUDIO NO ES MEJOR SI ES DISPAR. Medido sobre una voz con tres notas de
voz de distinta sala y registro: solo el clip largo de 31,4 s da 0,684 de ECAPA;
los tres juntos, 41,1 s, dan 0,592. El script avisa cuando detecta clips que no
suenan entre si a la misma grabacion.

CALIDAD DE LA REFERENCIA
Los prefijos oficiales llevan 22-33 s de UNA grabacion continua. Con 15,5 s ya
salen las cifras de arriba. La curva de "cuanto audio hace falta" esta medida en
docs/clonado-de-voz.md §7.8; el script avisa cuando te quedas corto.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auditar_encoder import encoder_comunitario  # noqa: E402

RITMO = 24000
MUESTRAS_LATENTE = 3200  # 1 latente = 1/7,5 s


def leer_audio(ruta):
    """Lee cualquier cosa que entienda ffmpeg y la deja en 24 kHz mono.

    Acepta .opus, .m4a, .mp3, WAV a otro ritmo, estereo... Antes exigia un WAV
    de 24 kHz mono y obligaba a convertir a mano; el paso extra no aportaba nada
    y era una fuente de errores. Las notas de voz de WhatsApp son .opus a 48 kHz,
    que es el caso mas comun aqui.
    """
    ruta = Path(ruta).expanduser()
    if not ruta.exists():
        raise SystemExit(f"no existe el fichero '{ruta}'")

    if ruta.suffix.lower() == ".wav":
        try:
            with wave.open(str(ruta)) as w:
                if w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == RITMO:
                    return (np.frombuffer(w.readframes(w.getnframes()), "<i2")
                            .astype(np.float32) / 32768)
        except wave.Error:
            pass                       # WAV raro: que lo arregle ffmpeg

    ffmpeg = os.environ.get("VOZ_FFMPEG", "ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "convertido.wav"
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(ruta),
             "-ar", str(RITMO), "-ac", "1", "-c:a", "pcm_s16le", str(destino)],
            capture_output=True)
        if r.returncode != 0 or not destino.exists():
            raise SystemExit(f"ffmpeg no pudo leer '{ruta}':\n"
                             f"{r.stderr.decode()[:300]}")
        print(f"    (convertido a 24 kHz mono con ffmpeg)")
        with wave.open(str(destino)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def avisar_calidad(x, sangria=""):
    avisos = []
    if float(np.abs(x).max()) >= 0.999:
        avisos.append("RECORTADO (pico en 1,0): la grabacion venia saturada")
    k = (len(x) // 240) * 240
    if k:
        sil = float(np.mean(np.sqrt((x[:k].reshape(-1, 240) ** 2).mean(1)) < 0.01))
        if sil > 0.35:
            avisos.append(f"{sil:.0%} de silencio: solo hay {len(x)/RITMO*(1-sil):.1f} s de voz")
    # Una grabacion de 24 kHz de verdad tiene energia por encima de 8 kHz. Si no
    # la tiene suele ser un 16 kHz reescalado, y el clonado se resiente. OJO:
    # tambien salta con audio generado por el PROPIO VibeVoice, que apenas
    # tiene brillo ahi arriba y sin embargo clona perfectamente (ECAPA 0,850).
    # Es un aviso, no un error.
    trozo = x[:RITMO * 10]
    e = np.abs(np.fft.rfft(trozo))
    f = np.fft.rfftfreq(len(trozo), 1 / RITMO)
    alto, bajo = e[f > 8000].sum(), e[(f > 100) & (f <= 8000)].sum()
    if bajo > 0 and alto / bajo < 1e-3:
        avisos.append("sin energia por encima de 8 kHz: o es un 16 kHz reescalado, "
                      "o lo genero VibeVoice (entonces es normal)")
    print()
    for a in avisos:
        print(f"{sangria}[aviso] {a}")


def avisar_heterogeneos(clips, rutas, umbral=0.626):
    """Avisa si los clips no suenan a la MISMA grabacion de la misma persona.

    MEDIDO, y es contraintuitivo: juntar clips heterogeneos del mismo hablante
    EMPEORA el clon. Sobre una voz con tres notas de voz de distinta sala,
    distancia y registro (ECAPA entre clips 0,51-0,59, o sea zona gris):

        solo el clip largo (31,4 s)      ECAPA del clon 0,684
        los tres juntos    (41,1 s)      ECAPA del clon 0,592
        solo los dos cortos ( 9,6 s)     ECAPA del clon 0,460

    Diez segundos MAS de material y el clon empeora 0,09. La curva de la §7.8
    del doc -- mas audio es mejor -- solo vale si los clips son homogeneos.
    Con material dispar, el prefijo promedia condiciones de grabacion en vez de
    acumular informacion de la voz.

    El aviso es informativo: se usa ECAPA si esta disponible, y si no, se calla.
    """
    if len(clips) < 2:
        return
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        import torchaudio
    except ImportError:
        return
    modelo = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
        run_opts={"device": "cpu"})

    def huella(x):
        t = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(x))[None], RITMO, 16000)
        with torch.no_grad():
            e = modelo.encode_batch(t).squeeze()
        return (e / e.norm()).numpy()

    hs = [huella(c) for c in clips]
    flojos = []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            c = float(hs[i] @ hs[j])
            if c < umbral:
                flojos.append((i, j, c))
    if flojos:
        print(f"[aviso] los clips no suenan igual entre si (umbral {umbral:.3f}):")
        for i, j, c in flojos:
            print(f"          {Path(rutas[i]).name}  vs  {Path(rutas[j]).name}:  {c:.3f}")
        print("        MEDIDO que juntar clips dispares EMPEORA el clon. Si uno de "
              "ellos\n        es claramente el mejor (mas largo y mas limpio), usa "
              "solo ese.")


def igualar_volumen(clips, objetivo=0.07):
    """Lleva cada clip al mismo RMS antes de codificar.

    Sin esto, un clip grabado mas alto pesa mas en los latentes que los demas y
    la voz resultante se parece mas a ESE clip que a la persona. 0,07 es el RMS
    medio de las voces oficiales. No toca el pico: si un clip venia recortado,
    lo sigue estando y eso se avisa aparte.
    """
    fuera = []
    for x in clips:
        rms = float(np.sqrt(np.mean(x ** 2)))
        fuera.append(x if rms < 1e-6 else np.clip(x * (objetivo / rms), -1, 1))
    return fuera


def a_cpu(salida):
    """Deja la cache KV en CPU para que el `.pt` cargue en cualquier maquina."""
    pkv = salida.past_key_values
    if hasattr(pkv, "layers"):                      # DynamicCache moderno
        # Hay capas sin rellenar: DynamicCache reserva una ranura por capa del
        # modelo entero, y el `lm` solo usa 4 de ellas. Saltarlas, no tocarlas.
        for capa in pkv.layers:
            if getattr(capa, "keys", None) is None:
                continue
            capa.keys, capa.values = capa.keys.cpu(), capa.values.cpu()
    elif hasattr(pkv, "key_cache"):                 # DynamicCache antiguo
        pkv.key_cache = [k.cpu() for k in pkv.key_cache]
        pkv.value_cache = [v.cpu() for v in pkv.value_cache]
    if salida.last_hidden_state is not None:
        salida.last_hidden_state = salida.last_hidden_state.cpu()
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", action="append",
                    help="audio con la voz (opus, m4a, mp3, wav...; se convierte solo). "
                         "Se puede repetir para varias muestras")
    ap.add_argument("--transcripcion", action="append",
                    help="lo que se dice en ese audio, LITERAL; uno por --audio y en el mismo orden")
    ap.add_argument("--salida", help="fichero .pt a escribir")
    ap.add_argument("--lote", help=
                    "JSON con VARIAS voces para fabricar de una sentada, "
                    "cargando el modelo UNA sola vez: una lista de "
                    "{\"salida\": ruta.pt, \"refs\": [{\"audio\": ..., "
                    "\"transcripcion\": ...}, ...]}. Excluye --audio/"
                    "--transcripcion/--salida")
    ap.add_argument("--sin-igualar-volumen", action="store_true",
                    help="no lleva todos los clips al mismo RMS antes de codificar")
    ap.add_argument("--techo-minimo", type=float, default=0.0,
                    help="no fabricar la voz si su techo ECAPA (una mitad de la referencia "
                         "contra la otra) queda por debajo; 0 = solo avisar. Medido: por "
                         "debajo de 0,70 el clon lo limita la grabacion, no el motor")
    ap.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    ap.add_argument("--dispositivo", default="auto")
    args = ap.parse_args()

    disp = args.dispositivo
    if disp == "auto":
        disp = "mps" if torch.backends.mps.is_available() else "cpu"
    tipo = torch.float32

    # ---------------------------------------------------- que hay que hacer --
    # Un "grupo" es una voz: sus clips de referencia y el .pt que sale. Con
    # --lote vienen varios en un JSON y el modelo se carga UNA vez para todos
    # (cargarlo son ~40 s y el doblaje de un panel fabricaba un clon por
    # hablante, cada uno en su proceso: cuatro voces eran cuatro cargas).
    if args.lote:
        if args.audio or args.transcripcion or args.salida:
            raise SystemExit("--lote ya trae las voces: no se mezcla con "
                             "--audio/--transcripcion/--salida")
        try:
            crudo = json.loads(Path(args.lote).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"no se pudo leer --lote {args.lote}: {e}")
        if not isinstance(crudo, list) or not crudo:
            raise SystemExit("--lote tiene que ser una lista de voces no vacia")
        grupos = []
        for k, g in enumerate(crudo):
            refs = g.get("refs") or []
            if not g.get("salida") or not refs:
                raise SystemExit(f"la voz {k} del lote necesita 'salida' y 'refs'")
            grupos.append({"salida": g["salida"],
                           "audios": [r["audio"] for r in refs],
                           "textos": [r["transcripcion"] for r in refs]})
    else:
        if not args.audio or not args.transcripcion or not args.salida:
            raise SystemExit("hacen falta --audio, --transcripcion y --salida "
                             "(o un --lote con todo)")
        if len(args.audio) != len(args.transcripcion):
            raise SystemExit(f"hay {len(args.audio)} --audio y {len(args.transcripcion)} "
                             "--transcripcion: tiene que haber uno por cada uno, en el mismo orden")
        grupos = [{"salida": args.salida, "audios": args.audio,
                   "textos": args.transcripcion}]

    # los audios se leen ANTES de cargar el modelo: un fichero que falta o un
    # numero de transcripciones que no cuadra tiene que fallar en el primer
    # segundo, no despues de 40 s de carga (y con --lote, no a la tercera voz)
    for g in grupos:
        if len(g["audios"]) != len(g["textos"]):
            raise SystemExit(f"{g['salida']}: {len(g['audios'])} audio(s) y "
                             f"{len(g['textos'])} transcripcion(es)")
        print(f"voz -> {g['salida']}")
        clips = []
        for ruta in g["audios"]:
            c = leer_audio(ruta)
            print(f"  {ruta}: {len(c)/RITMO:.1f} s", end="")
            avisar_calidad(c, sangria="    ")
            clips.append(c)
        g["total"] = sum(len(c) for c in clips) / RITMO
        if len(clips) > 1:
            print(f"  {len(clips)} muestras, {g['total']:.1f} s en total")
        if g["total"] < 15:
            print(f"[aviso] {g['total']:.1f} s en total. Las cifras de esta herramienta estan "
                  f"medidas con 15,5 s y las voces oficiales llevan 22-33 s.")
        avisar_heterogeneos(clips, g["audios"])
        if not args.sin_igualar_volumen and len(clips) > 1:
            clips = igualar_volumen(clips)
        # El techo va ANTES de cargar el modelo: si la grabacion no se
        # reconoce a si misma, 40 s de carga y un .pt no arreglan nada.
        try:
            from techo import techo_de_clips, informar as informar_techo
            g["techo"] = techo_de_clips(clips)
            informar_techo(g["techo"], sangria="  ", segundos=g["total"])
        except ImportError:
            g["techo"] = None          # sin speechbrain no hay techo; no es un error
        if args.techo_minimo and (g["techo"] or 0) < args.techo_minimo:
            raise SystemExit(f"{g['salida']}: techo {g['techo']} por debajo de "
                             f"--techo-minimo {args.techo_minimo}: hace falta otra grabacion")
        g["clips"] = clips
        g["texto"] = "".join(t if t.endswith("\n") else t + "\n"
                             for t in g["textos"])

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    print(f"cargando el modelo en {disp}...")
    proc = VibeVoiceStreamingProcessor.from_pretrained(args.modelo)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.modelo, dtype=tipo, device_map="cpu", attn_implementation="sdpa").eval()
    m = modelo.model
    # La tabla de embeddings del tts_lm no se usa nunca y el modelo preparado la
    # trae podada: se apunta a la del otro LM, como hace voz_stream.py.
    m.tts_language_model.embed_tokens = m.language_model.embed_tokens
    # El encoder que el 0.5B no trae. from_pretrained lo instancio con pesos
    # aleatorios y aqui se sustituye por el de verdad.
    m.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.to(tipo) for k, v in encoder_comunitario(Path(args.cache)).items()}, strict=True)
    modelo.to(disp)

    tok = proc.tokenizer

    # 4. ramas negativas del CFG: un solo <|image_pad|>, y la del tts con
    #    tipo 1 sobre la SALIDA del lm negativo (no sobre su embedding). NO
    #    dependen de la voz, asi que se calculan una vez para todo el lote.
    with torch.no_grad():
        neg = torch.tensor([[tok.convert_tokens_to_ids("<|image_pad|>")]], device=disp)
        neg_lm = modelo.forward_lm(input_ids=neg, attention_mask=torch.ones_like(neg),
                                   use_cache=True, return_dict=True)
        neg_tts = modelo.forward_tts_lm(
            input_ids=neg, attention_mask=torch.ones_like(neg),
            lm_last_hidden_state=neg_lm.last_hidden_state,
            tts_text_masks=torch.ones_like(neg), use_cache=True, return_dict=True)
    neg_cpu = {"neg_lm": a_cpu(neg_lm), "neg_tts_lm": a_cpu(neg_tts)}

    for g in grupos:
        clips, texto = g["clips"], g["texto"]
        ids = torch.tensor([tok.encode(texto, add_special_tokens=False)], device=disp)
        with torch.no_grad():
            # 1. cada clip por separado -> latentes -> escalado -> conector, y se
            #    concatenan los latentes en el orden en que llegaron. Codificar por
            #    separado evita que el salto del empalme entre en la ventana
            #    receptiva del encoder convolucional.
            trozos = []
            for c in clips:
                n_c = math.ceil(len(c) / MUESTRAS_LATENTE)
                e = m.acoustic_tokenizer.encode(torch.from_numpy(c)[None, None].to(disp, tipo))
                z, _ = e.sample(dist_type=m.acoustic_tokenizer.std_dist_type)
                rasgos = (z + m.speech_bias_factor) * m.speech_scaling_factor
                trozos.append(m.acoustic_connector(rasgos[:, :n_c].to(tipo)))
            conectado = torch.cat(trozos, dim=1)
            n = conectado.shape[1]

            # 2. rama de texto: prefill directo, sin plantilla ni tokens especiales
            lm = modelo.forward_lm(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   use_cache=True, return_dict=True)
            M = lm.last_hidden_state.shape[1]

            # 3. rama TTS: N latentes + M texto, TODO con tipo 0 y sin marcadores.
            #    forward_tts_lm sobrescribe las M ultimas posiciones con el hidden
            #    del lm, asi que ahi va relleno; y suma tts_input_types(mascara).
            embeds = torch.cat(
                [conectado, torch.zeros(1, M, conectado.shape[-1], device=disp, dtype=tipo)], 1)
            tts = modelo.forward_tts_lm(
                attention_mask=torch.ones(1, n + M, dtype=torch.long, device=disp),
                inputs_embeds=embeds, lm_last_hidden_state=lm.last_hidden_state,
                tts_text_masks=torch.zeros(1, n + M, dtype=torch.long, device=disp),
                use_cache=True, return_dict=True)

        prefijo = {"lm": a_cpu(lm), "tts_lm": a_cpu(tts), **neg_cpu}
        salida = Path(g["salida"])
        salida.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prefijo, salida)
        # La ficha al lado del .pt: el techo y de donde salio la voz. El .pt
        # no se toca (voz_stream.py lo carga tal cual); quien quiera saber
        # cuanto puede dar esta voz lo lee aqui, no lo vuelve a medir.
        salida.with_suffix(".json").write_text(json.dumps({
            "techo": g["techo"], "segundos": round(g["total"], 1),
            "muestras": len(clips), "fuentes": [str(a) for a in g["audios"]],
            "posiciones": n + M}, ensure_ascii=False, indent=1))
        print(f"\n{g['total']:.1f} s de referencia en {len(clips)} muestra(s) -> "
              f"{n} latentes + {M} tokens de texto = {n+M} posiciones")
        print(f"prefijo escrito en {salida} ({salida.stat().st_size/2**20:.1f} MB)")
        if not args.lote:
            print("\nProbarlo:")
            print(f"  VIBEVOICE_VOZ={salida.stem} ./scripts/voz-stream-mac.sh")
    if args.lote:
        print(f"\n{len(grupos)} voces con UNA sola carga del modelo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
