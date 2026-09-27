#!/usr/bin/env python3
"""Emociones graduables por niveles con direcciones sacadas de PARES REALES (plan de emocion, E2).

Lo que cambia frente a los barridos de la red (2026-09-27): alli la direccion era el vector de una regresion
(ridge, normalizado) y la fuerza una fraccion de la norma (lambda 0,2 = 0,7 % de la condicion): no movio el tono.
Aqui la direccion es la DIFERENCIA DE MEDIAS entre el mismo actor diciendo la misma frase con una emocion y en
neutro (CREMA-D), y se suma SIN normalizar: nivel 1 = el cambio medio de un actor real; nivel 2 = el doble.

  extraer      CREMA-D (wav 16 kHz, <actor>_<frase>_<EMO>_<intensidad>.wav) -> por clip, la media por fotograma de
               la condicion, de la negativa y de los residuales de las 20 capas en las dos ramas, en la pasada
               forzada (scripts/lora/forzado.py) con una referencia NEUTRA del mismo actor como prefijo.
  direcciones  delta[sitio/EMO] = media sobre (actor, frase) de x(EMO) - x(NEU); y la prueba de graduacion:
               la proyeccion de los clips de IEO en intensidad LO/MD/HI sobre su delta (tiene que crecer).
  generar      voces x frases x emociones x niveles, misma semilla por nivel (pareado), sumando nivel * delta
               en la condicion (rama pos, ambas o natural = delta de la negativa en la negativa) o en el
               residual de una capa.
  juzgar       emotion2vec+ (9 clases) o, si no esta, speechbrain IEMOCAP (4); F0, recorrido, energia y ritmo
               (perfil_vocal); WER (whisper large-v3); ECAPA y UTMOS; todo pareado contra el nivel 0.

    python3 scripts/red/emociones.py extraer --crema <dir wav> --modelo <ruta> --salida emo/extra
    python3 scripts/red/emociones.py direcciones --extra emo/extra --salida emo/direcciones.npz
    python3 scripts/red/emociones.py generar --modelo <ruta> --voces <dir .pt> --voz sp-Spk1_man \
        --direcciones emo/direcciones.npz --emociones ANG,HAP,SAD --niveles 0,1,2,3 --sitio condicion \
        --rama natural --salida emo/gen/cond_natural
    python3 scripts/red/emociones.py juzgar --carpeta emo/gen/cond_natural [--dispositivo cuda]

Con RED_DISPOSITIVO=cuda la sintesis va en la GPU en fp32 (modelo.py). CREMA-D es ODbL: aqui se usa para sacar
direcciones y medir; si algun dia entra en pesos, revisar el share-alike (plan de emocion §3.2).
"""
import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI.parent / "lora"))
import modelo as MO  # noqa: E402

EMOCIONES = {"ANG": "enojo", "DIS": "asco", "FEA": "miedo", "HAP": "alegria", "SAD": "tristeza", "NEU": "neutral"}
FRASES_CREMA = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.", "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
CAPAS = 20


def leer24(ruta):
    import librosa
    import soundfile as sf
    x, hz = sf.read(str(ruta), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if hz != 24000:
        x = librosa.resample(x, orig_sr=hz, target_sr=24000)
    return x


def cargar_con_codificador(ruta, cache):
    from auditar_encoder import encoder_comunitario
    m, tok = MO.cargar(ruta)
    m.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(cache).expanduser()).items()}, strict=True)
    return m, tok


# ---------------------------------------------------------------------------------------------- extraer
@torch.no_grad()
def estados_con_residuales(modelo, tok, ref_txt, ref_lat, txt, lat):
    """forzado.estados() mas los residuales de las 20 capas en las posiciones que condicionan, en las dos ramas."""
    from forzado import disposicion
    m = modelo.model
    d = next(modelo.parameters()).device
    ref_ids = tok.encode(ref_txt, add_special_tokens=False)
    txt_ids = tok.encode(txt.strip() + "\n", add_special_tokens=False)
    orden = disposicion(len(txt_ids), lat.shape[0])
    if orden is None:
        return None
    capturas = []
    ganchos = [c.register_forward_hook(lambda mod, e, s: capturas.append((s[0] if isinstance(s, tuple) else s)[0]))
               for c in m.tts_language_model.layers]
    try:
        ids = torch.tensor([ref_ids + txt_ids], device=d)
        h_lm = modelo.forward_lm(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True).last_hidden_state[0]
        M = len(ref_ids)
        con_ref = m.acoustic_connector(lat_ref_a(ref_lat, d)[None])[0]
        con = m.acoustic_connector(lat.to(d)[None])[0]
        filas, tipos, pos_voz = [con_ref, h_lm[:M]], [0] * (ref_lat.shape[0] + M), {}
        for tipo, i in orden:
            if tipo == "t":
                filas.append(h_lm[M + i][None]); tipos.append(1)
            else:
                pos_voz[i] = len(tipos); filas.append(con[i][None]); tipos.append(0)
        emb = torch.cat(filas, 0)[None] + m.tts_input_types(torch.tensor([tipos], device=d))
        capturas.clear()
        h = m.tts_language_model(inputs_embeds=emb, attention_mask=torch.ones(1, emb.shape[1], device=d),
                                 return_dict=True).last_hidden_state[0]
        res_pos = torch.stack(capturas)                          # [20, L, 896]
        T = lat.shape[0]
        idx = torch.tensor([pos_voz[j] for j in range(T)], device=d) - 1
        cond, rp = h[idx], res_pos[:, idx]                       # [T, 896], [20, T, 896]
        neg_id = torch.tensor([[tok.convert_tokens_to_ids("<|image_pad|>")]], device=d)
        h_neg_lm = modelo.forward_lm(input_ids=neg_id, attention_mask=torch.ones_like(neg_id), return_dict=True).last_hidden_state
        emb_n = torch.cat([h_neg_lm[0], con], 0)[None] + m.tts_input_types(torch.tensor([[1] + [0] * T], device=d))
        capturas.clear()
        h_n = m.tts_language_model(inputs_embeds=emb_n, attention_mask=torch.ones(1, T + 1, device=d),
                                   return_dict=True).last_hidden_state[0]
        rn = torch.stack(capturas)[:, :T]
        return cond, h_n[:T], rp, rn
    finally:
        for g in ganchos:
            g.remove()


def lat_ref_a(x, d):
    return x.to(d)


def extraer(a):
    from forzado import latentes
    m, tok = cargar_con_codificador(a.modelo, a.cache)
    d = next(m.parameters()).device
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    clips = defaultdict(list)
    for f in sorted(Path(a.crema).glob("*.wav")):
        p = f.stem.split("_")
        if len(p) == 4 and p[1] in FRASES_CREMA and p[2] in EMOCIONES:
            clips[p[0]].append((f, p[1], p[2], p[3]))
    actores = sorted(clips)[: a.max_actores] if a.max_actores else sorted(clips)
    t0 = time.time()
    for n, actor in enumerate(actores, 1):
        dest = sal / f"{actor}.npz"
        if dest.exists():
            continue
        # referencia: el primer neutro del actor en este orden de frases; esa frase no se usa como objetivo
        neutros = {fr: f for f, fr, e, i in clips[actor] if e == "NEU"}
        ref_fr = next((fr for fr in ("TSI", "DFA", "WSI", "TAI", "ITS") if fr in neutros), None)
        if ref_fr is None:
            continue
        ref_lat = latentes(m, leer24(neutros[ref_fr]), d, muestrear=False)
        filas = dict(cond=[], neg=[], res=[], res_neg=[], frase=[], emocion=[], intensidad=[], T=[])
        for f, fr, emo, inten in clips[actor]:
            if fr == ref_fr:
                continue
            lat = latentes(m, leer24(f), d, muestrear=False)
            r = estados_con_residuales(m, tok, FRASES_CREMA[ref_fr], ref_lat, FRASES_CREMA[fr], lat)
            if r is None:
                continue
            cond, neg, rp, rn = r
            filas["cond"].append(cond.mean(0).float().cpu().numpy())
            filas["neg"].append(neg.mean(0).float().cpu().numpy())
            filas["res"].append(rp.mean(1).half().cpu().numpy())
            filas["res_neg"].append(rn.mean(1).half().cpu().numpy())
            filas["frase"].append(fr); filas["emocion"].append(emo); filas["intensidad"].append(inten)
            filas["T"].append(lat.shape[0])
        np.savez_compressed(dest, **{k: np.array(v) for k, v in filas.items()}, referencia=ref_fr)
        print(f"[extraer] {actor}: {len(filas['T'])} clips ({n}/{len(actores)}, {time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------------------- direcciones
def direcciones(a):
    datos = [np.load(f, allow_pickle=True) for f in sorted(Path(a.extra).glob("*.npz"))]
    sitios = {"condicion": "cond", "negativa": "neg"}
    difs = defaultdict(list)              # (sitio, EMO) -> [vector]
    graduacion = defaultdict(lambda: defaultdict(list))
    for d in datos:
        fr, emo, inten = d["frase"], d["emocion"], d["intensidad"]
        X = {"condicion": d["cond"], "negativa": d["neg"]}
        for c in range(CAPAS):
            X[f"capa{c:02d}"] = d["res"][:, c].astype(np.float32)
            X[f"capa{c:02d}_neg"] = d["res_neg"][:, c].astype(np.float32)
        for frase in set(fr):
            neu = np.where((fr == frase) & (emo == "NEU"))[0]
            if not len(neu):
                continue
            for e in EMOCIONES:
                if e == "NEU":
                    continue
                # para la direccion, las intensidades sin marca (XX) y la alta (HI) de IEO
                sel = np.where((fr == frase) & (emo == e) & np.isin(inten, ["XX", "HI"]))[0]
                for s in X:
                    for i in sel:
                        difs[(s, e)].append(X[s][i] - X[s][neu].mean(0))
    delta = {f"{s}/{e}": np.mean(v, 0).astype(np.float32) for (s, e), v in difs.items()}
    # prueba de graduacion: IEO en LO / MD / HI proyectado sobre su delta (en unidades de delta: 1 = cambio medio)
    for d in datos:
        fr, emo, inten = d["frase"], d["emocion"], d["intensidad"]
        neu = np.where((fr == "IEO") & (emo == "NEU"))[0]
        if not len(neu):
            continue
        for e in EMOCIONES:
            if e == "NEU":
                continue
            u = delta[f"condicion/{e}"]
            for nivel in ("LO", "MD", "HI"):
                for i in np.where((fr == "IEO") & (emo == e) & (inten == nivel))[0]:
                    graduacion[e][nivel].append(float((d["cond"][i] - d["cond"][neu].mean(0)) @ u / (u @ u)))
    rms = float(np.sqrt(np.mean(np.concatenate([d["cond"] for d in datos]) ** 2)))
    np.savez(a.salida, **delta)
    info = dict(n_actores=len(datos), rms_condicion=rms,
                norma={k: round(float(np.linalg.norm(v)), 3) for k, v in delta.items()
                       if k.split("/")[0] in ("condicion", "negativa", "capa11", "capa12", "capa14", "capa16")},
                pares={f"{s}/{e}": len(v) for (s, e), v in difs.items() if s == "condicion"},
                coseno_entre_emociones={f"{e1}-{e2}": round(float(delta[f'condicion/{e1}'] @ delta[f'condicion/{e2}'] /
                                                                np.linalg.norm(delta[f'condicion/{e1}']) / np.linalg.norm(delta[f'condicion/{e2}'])), 3)
                                        for i, e1 in enumerate([e for e in EMOCIONES if e != "NEU"])
                                        for e2 in [e for e in EMOCIONES if e != "NEU"][i + 1:]},
                graduacion_IEO={e: {n: [round(float(np.mean(v)), 3), len(v)] for n, v in g.items()} for e, g in graduacion.items()})
    Path(a.salida).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("n_actores", "rms_condicion", "pares", "graduacion_IEO", "coseno_entre_emociones")}, indent=1))
    print("normas de la condicion:", {k: v for k, v in info["norma"].items() if k.startswith("condicion")})


def contraste(a):
    """Direcciones que parten de como lee el modelo, no de un actor neutro. El juez de emocion califica la voz de
    base del 0.5B como alegre (0,82 medido): sumar tristeza - neutro a una voz que ya suena alegre apenas se nota.
    <E>_rel = delta(E) - delta(alegria) lleva de alegre a E; CALM = -delta(alegria) quita la alegria."""
    D = dict(np.load(a.direcciones))
    nuevos = {}
    for k in list(D):
        sitio, emo = k.split("/")
        if emo != "HAP":
            continue
        for e in ("ANG", "SAD", "FEA", "DIS"):
            if f"{sitio}/{e}" in D:
                nuevos[f"{sitio}/{e}_rel"] = (D[f"{sitio}/{e}"] - D[k]).astype(np.float32)
        nuevos[f"{sitio}/CALM"] = (-D[k]).astype(np.float32)
    D.update(nuevos)
    np.savez(a.direcciones, **D)
    print({k: round(float(np.linalg.norm(v)), 2) for k, v in nuevos.items() if k.startswith("condicion/")})


# --------------------------------------------------------------------------------------------- generar
class Mando:
    """Suma nivel * delta en la condicion (enganche de bucle.Generador) o en el residual de una capa."""

    def __init__(self, gen, deltas, sitio, rama, nivel):
        self.gen, self.nivel, self.rama, self.sitio = gen, nivel, rama, sitio
        dv = next(gen.m.parameters()).device
        self.dp = torch.tensor(deltas[0], device=dv)
        self.dn = torch.tensor(deltas[1], device=dv) if deltas[1] is not None else self.dp
        self.h = None
        if sitio.startswith("capa") and nivel != 0:
            self.h = gen.m.model.tts_language_model.layers[int(sitio[4:])].register_forward_hook(self.capa)

    def condicion(self, c, n, j):
        if self.nivel == 0:
            return c, n
        c = c + self.nivel * self.dp
        if self.rama == "ambas":
            n = n + self.nivel * self.dp
        elif self.rama == "natural":
            n = n + self.nivel * self.dn
        return c, n

    def capa(self, mod, ent, sal):
        neg = self.gen._en_negativo
        if neg and self.rama == "pos":
            return None
        d = (self.dn if self.rama == "natural" else self.dp) if neg else self.dp
        h = sal[0] if isinstance(sal, tuple) else sal
        h = h + self.nivel * d
        return (h,) + tuple(sal[1:]) if isinstance(sal, tuple) else h

    def quitar(self):
        if self.h:
            self.h.remove()


def generar(a):
    import soundfile as sf
    from bucle import Generador
    m, tok = MO.cargar(a.modelo)
    D = np.load(a.direcciones)
    frases = json.loads(Path(a.frases).read_text()) if a.frases else FRASES_DEMO
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    niveles = [float(x) for x in a.niveles.split(",")]
    for voz in a.voz:
        base = MO.prefijo(Path(a.voces) / f"{voz}.pt")
        for k, (idioma, texto) in enumerate(frases):
            for s in (int(x) for x in a.semillas.split(",")):
                for emo in a.emociones.split(","):
                    for nivel in niveles:
                        nombre = f"{voz}__f{k}{idioma}__s{s}__{emo}__n{nivel:g}"
                        if nivel == 0:
                            nombre = f"{voz}__f{k}{idioma}__s{s}__NEU__n0"
                        if (sal / f"{nombre}.wav").exists():
                            continue
                        clave = f"{a.sitio}/{emo}"
                        dn = D[f"{'negativa' if a.sitio == 'condicion' else a.sitio + '_neg'}/{emo}"] if a.rama == "natural" else None
                        torch.manual_seed(s)
                        gen = Generador(m, base, MO.fichas(texto, tok), cfg_scale=a.cfg)
                        mando = Mando(gen, (D[clave], dn), a.sitio, a.rama, nivel)
                        if a.sitio == "condicion":
                            gen.al_condicion = mando.condicion
                        t0 = time.time()
                        onda = gen.correr(max_fotogramas=a.max_fotogramas)
                        mando.quitar()
                        sf.write(str(sal / f"{nombre}.wav"), onda.clamp(-1, 1).numpy(), 24000)
                        print(f"{nombre}: {onda.shape[0] / 24000:.1f} s en {time.time() - t0:.0f} s", flush=True)
    (sal / "frases.json").write_text(json.dumps(frases, ensure_ascii=False))


FRASES_DEMO = [
    ["es", "Acabo de recibir la noticia y quería contártela antes que a nadie."],
    ["es", "Mañana vamos a revisar otra vez todo el proyecto desde el principio."],
    ["es", "No puedo creer que ya sea viernes y todavía quede tanto por hacer."],
    ["en", "I just got the news and I wanted you to hear it first."],
]


# ---------------------------------------------------------------------------------------------- juzgar
class JuezEmocion:
    """emotion2vec+ large (9 clases, multilingue) si funasr esta instalado; si no, speechbrain IEMOCAP (4)."""

    def __init__(self, dispositivo):
        self.tipo = None
        try:
            from funasr import AutoModel
            self.m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", device=dispositivo, disable_update=True)
            self.tipo = "emotion2vec"
        except Exception as e:
            print(f"[juez] emotion2vec no disponible ({type(e).__name__}); speechbrain IEMOCAP", flush=True)
            from speechbrain.inference.interfaces import foreign_class
            self.m = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier", run_opts={"device": dispositivo},
                                   savedir=str(Path.home() / ".cache/juez-emocion-iemocap"))
            self.tipo = "iemocap"

    def __call__(self, ruta16):
        if self.tipo == "emotion2vec":
            r = self.m.generate(str(ruta16), granularity="utterance", extract_embedding=False)[0]
            return {lab.split("/")[-1]: round(float(p), 4) for lab, p in zip(r["labels"], r["scores"])}
        out_prob, score, index, lab = self.m.classify_file(str(ruta16))
        labs = ["neu", "ang", "hap", "sad"]
        p = torch.softmax(out_prob[0], -1).tolist() if out_prob.dim() > 1 else out_prob.tolist()
        mapa = {"neu": "neutral", "ang": "angry", "hap": "happy", "sad": "sad"}
        return {mapa[l]: round(float(v), 4) for l, v in zip(labs, p)}


CLASE = {"ANG": "angry", "HAP": "happy", "SAD": "sad", "FEA": "fearful", "DIS": "disgusted", "NEU": "neutral", "CALM": "neutral"}


class JuezDimensional:
    """Activacion, dominancia y valencia (0-1) del modelo de audEERING (MSP-Podcast). CC BY-NC-SA: SOLO juez."""

    def __init__(self, dispositivo):
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class Cabeza(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.dense, self.dropout = nn.Linear(c.hidden_size, c.hidden_size), nn.Dropout(c.final_dropout)
                self.out_proj = nn.Linear(c.hidden_size, c.num_labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class Modelo(Wav2Vec2PreTrainedModel):
            def __init__(self, c):
                super().__init__(c)
                self.wav2vec2, self.classifier = Wav2Vec2Model(c), Cabeza(c)
                self.init_weights()

            def forward(self, x):
                return self.classifier(self.wav2vec2(x)[0].mean(1))

        nombre = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.proc = Wav2Vec2Processor.from_pretrained(nombre)
        self.m = Modelo.from_pretrained(nombre).to(dispositivo).eval()
        self.d = dispositivo

    def __call__(self, x16):
        v = self.proc(x16, sampling_rate=16000, return_tensors="pt").input_values.to(self.d)
        with torch.inference_mode():
            a, d, val = self.m(v)[0].tolist()
        return dict(activacion=round(a, 4), dominancia=round(d, 4), valencia=round(val, 4))


def juzgar(a):
    import tempfile
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    import juez_lote as JL
    import perfil_vocal as PV
    from normalizar_texto import normalizar
    carpeta = Path(a.carpeta)
    frases = json.loads((carpeta / "frases.json").read_text())
    sal = carpeta / "juez.json"
    med = json.loads(sal.read_text()) if sal.exists() else {}
    wavs = [w for w in sorted(carpeta.glob("*.wav")) if w.stem not in med]
    print(f"[juez] {len(wavs)} clips por juzgar en {carpeta}", flush=True)
    if wavs:
        gpu = a.dispositivo.startswith("cuda")
        wh = WhisperModel("large-v3", device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8", cpu_threads=8)
        ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": a.dispositivo},
                                               savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"))
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(a.dispositivo).eval()
        emo = JuezEmocion(a.dispositivo)
        dim = JuezDimensional(a.dispositivo)
        tmp = Path(tempfile.mkdtemp())
        for n, w in enumerate(wavs, 1):
            voz, fr, s, e, nv = w.stem.split("__")
            k, idioma = int(re.match(r"f(\d+)", fr).group(1)), fr[-2:]
            texto = frases[k][1]
            x, hz = sf.read(str(w), dtype="float32")
            x16 = librosa.resample(x, orig_sr=hz, target_sr=16000)
            r16 = tmp / f"{w.stem}.wav"
            sf.write(str(r16), x16, 16000)
            segs, _ = wh.transcribe(x16, language=idioma, beam_size=5, condition_on_previous_text=False, temperature=0.0)
            oido = " ".join(z.text for z in segs).strip()
            with torch.inference_mode():
                hue = ecapa.encode_batch(torch.from_numpy(x16[: 16000 * 30])[None].to(a.dispositivo)).squeeze().cpu().numpy()
                u = float(utmos(torch.from_numpy(x16)[None].to(a.dispositivo), 16000).item())
            p = PV.perfil(x, hz, texto)
            med[w.stem] = dict(voz=voz, frase=k, idioma=idioma, semilla=int(s[1:]), emocion=e, nivel=float(nv[1:]),
                               dur=round(len(x) / hz, 2), oido=oido,
                               wer=round(JL.wer(normalizar(texto, idioma), normalizar(oido, idioma)), 4),
                               utmos=round(u, 3), huella=[round(float(v), 5) for v in hue / np.linalg.norm(hue)],
                               hz=p.get("hz"), recorrido=p.get("recorrido"), rango_db=p.get("rango_db"),
                               silabas_s=p.get("silabas_s"), pausas_min=p.get("pausas_min"), emocion_juez=emo(r16), **dim(x16))
            if n % 10 == 0 or n == len(wavs):
                sal.write_text(json.dumps(med, ensure_ascii=False))
                print(f"[juez] {n}/{len(wavs)}", flush=True)
    resumen(carpeta, med)


def resumen(carpeta, med):
    """Por emocion y nivel, pareado contra el nivel 0 de la misma (voz, frase, semilla)."""
    base = {(m["voz"], m["frase"], m["semilla"]): m for m in med.values() if m["nivel"] == 0}
    filas = defaultdict(lambda: defaultdict(list))
    for m in med.values():
        b = base.get((m["voz"], m["frase"], m["semilla"]))
        if not b or m["nivel"] == 0:
            continue
        f = filas[(m["emocion"], m["nivel"])]
        clase = CLASE[m["emocion"].split("_")[0]]
        f["p_emocion"].append(m["emocion_juez"].get(clase, 0.0) - b["emocion_juez"].get(clase, 0.0))
        f["gana"].append(float(max(m["emocion_juez"], key=m["emocion_juez"].get) == clase))
        if m["hz"] and b["hz"]:
            f["tono_st"].append(12 * math.log2(m["hz"] / b["hz"]))
        for k in ("recorrido", "rango_db", "silabas_s", "activacion", "valencia", "dominancia"):
            if m.get(k) is not None and b.get(k) is not None:
                f[k].append(m[k] - b[k])
        f["wer"].append(100 * (m["wer"] - b["wer"]))
        f["utmos"].append(m["utmos"] - b["utmos"])
        f["ecapa_base"].append(float(np.dot(m["huella"], b["huella"])))
        f["rotos"].append(float(m["wer"] > 0.25 and b["wer"] <= 0.10))
    out = {}
    print(f"{'emocion':8} {'nivel':>5} {'n':>3} {'p(emo) dif':>10} {'gana':>5} {'tono st':>8} {'recorr':>7} {'rango dB':>8} "
          f"{'sil/s':>6} {'activ':>6} {'valen':>6} {'WER pts':>8} {'UTMOS':>7} {'ECAPA':>6} {'rotos':>5}")
    for (e, nv), f in sorted(filas.items()):
        r = {k: round(float(np.mean(v)), 3) for k, v in f.items() if v}
        r["n"] = len(f["wer"])
        out[f"{e}/{nv:g}"] = r
        print(f"{e:8} {nv:5g} {r['n']:3d} {r.get('p_emocion', 0):+10.3f} {r.get('gana', 0):5.2f} {r.get('tono_st', 0):+8.2f} "
              f"{r.get('recorrido', 0):+7.2f} {r.get('rango_db', 0):+8.2f} {r.get('silabas_s', 0):+6.2f} "
              f"{r.get('activacion', 0):+6.3f} {r.get('valencia', 0):+6.3f} {r.get('wer', 0):+8.2f} "
              f"{r.get('utmos', 0):+7.3f} {r.get('ecapa_base', 0):6.3f} {r.get('rotos', 0):5.2f}")
    (carpeta / "resumen.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="orden", required=True)
    e = sub.add_parser("extraer")
    e.add_argument("--crema", required=True)
    e.add_argument("--modelo", required=True)
    e.add_argument("--cache", default="~/.cache/vibevoice-nix")
    e.add_argument("--salida", required=True)
    e.add_argument("--max-actores", type=int, default=0)
    d = sub.add_parser("direcciones")
    d.add_argument("--extra", required=True)
    d.add_argument("--salida", required=True)
    g = sub.add_parser("generar")
    g.add_argument("--modelo", required=True)
    g.add_argument("--voces", required=True)
    g.add_argument("--voz", action="append", required=True)
    g.add_argument("--direcciones", required=True)
    g.add_argument("--emociones", default="ANG,HAP,SAD,FEA,DIS")
    g.add_argument("--niveles", default="0,1,2,3")
    g.add_argument("--sitio", default="condicion")
    g.add_argument("--rama", choices=["pos", "ambas", "natural"], default="natural")
    g.add_argument("--semillas", default="11")
    g.add_argument("--frases", default=None, help="json [[idioma, texto], ...]; por defecto FRASES_DEMO")
    g.add_argument("--cfg", type=float, default=3.0)
    g.add_argument("--max-fotogramas", type=int, default=300)
    g.add_argument("--salida", required=True)
    j = sub.add_parser("juzgar")
    j.add_argument("--carpeta", required=True)
    j.add_argument("--dispositivo", default="cpu")
    c = sub.add_parser("contraste")
    c.add_argument("--direcciones", required=True)
    r = sub.add_parser("resumen")
    r.add_argument("--carpeta", required=True)
    a = ap.parse_args()
    if a.orden == "resumen":
        resumen(Path(a.carpeta), json.loads((Path(a.carpeta) / "juez.json").read_text()))
    else:
        {"extraer": extraer, "direcciones": direcciones, "generar": generar, "juzgar": juzgar, "contraste": contraste}[a.orden](a)


if __name__ == "__main__":
    main()
