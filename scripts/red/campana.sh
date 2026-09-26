#!/usr/bin/env bash
# La campana de la red por dentro en una maquina de CPU en spot: instrumentar -> sondas -> dirigir -> jueces.
# Corre DENTRO de la maquina, despues de spot_entorno.sh (que deja ~/red.env y ~/entorno.ok):
#
#   RED_S3=s3://<bucket>/<prefijo> bash ~/mejora/scripts/red/campana.sh
#
# REANUDABLE: al arrancar baja $RED_S3/trabajo/ a ~/campana y quita lo que una interrupcion pudo dejar a medias
# (un .npz o .wav truncado); durante, sube ~/campana a S3 cada $SYNC_CADA s y al acabar cada paso; al morir
# (fin, error o el SIGTERM del aviso de spot) sube una ultima vez. Cada paso se salta lo que ya existe:
# instrumentar y dirigir por fichero, sondas si ya estan sus direcciones, jueces por clip (juez.json).
# Una interrupcion de spot cuesta como mucho el clip en curso: se vuelve a `spot_ec2.sh lanzar`, se monta el
# entorno y se relanza esto mismo.
#
# Parametros (entorno), por defecto los de la campana (docs/bancos/2026-09-26-red-interna-entorno.md §4):
#   VOCES      "sp-Spk1_man sp-Spk0_woman carlos liliana"   prefijos .pt en ~/voces: los de serie los baja
#              spot_entorno.sh; los clones (clonar_voz.py --lote, lote_clones.json) se suben a $RED_S3/voces_pt/
#   GRUPOS     es,en,es_numeros                  grupos de scripts/corpus_mejora.json (12 frases)
#   FRASES     (vacio = todas)                   solo las N primeras de cada grupo
#   DURACIONES 1                                 instrumentar tambien largas,cortas de scripts/red/corpus_duraciones.json
#   SEMILLAS   11,101
#   DIR_VOCES  "sp-Spk1_man carlos liliana"      voces del barrido de dirigir.py
#   LAMBDAS    0,0.05,0.1,0.2
#   BARRIDOS   clave:rama:desde:grupos[:duraciones], separados por espacios. Por defecto:
#              condicion/f0_st en las ramas ambas y pos; la mejor capaNN/f0_st de las sondas (R2 por frase
#              apartada); condicion/sexo (de $W/prefijos_direcciones.npz, que deja analizar_prefijos.py); y
#              condicion/f0_st --desde 40 sobre las frases largas (con las de "es" el clip acaba hacia el 40)
#   DIR_GRUPOS es                                grupos del barrido cuando el item no dice otros
#   IDENTIDADES voz=<ruta en ~/voces_refs>       control ECAPA de un clon: SOLO la referencia que se aparto al
#              clonar (nunca las refs con que se clono); las voces de serie, sus clips base de CONTROL_GRUPOS
#   DESCRIPTOR hz   SIGNO 1                      lo que la direccion tiene que mover, y hacia donde
#   PASOS      "instrumentar sondas dirigir jueces"
#              otros: "jueces_rapidos" (criba sin UTMOS) y "paridad" (el clip del Mac en ~/patron)
#
# Puertas: las fijadas antes de medir en docs/bancos/2026-09-27-red-interna-mac.md (I1: sondas.py por frase
# apartada; I2: puerta_dirigir.py sobre juez_lote.py). juez_lote.py lleva whisper large-v3 int8 y UTMOS en CPU:
# es lo caro de la campana; jueces_rapidos (whisper small int8 y ECAPA, pareados) sirve de criba antes.
# Los barridos con desde > 0 pasan ademas por costura.py (la ventana de +-2 s alrededor de donde entra el mando).
set -euo pipefail
: "${RED_S3:?falta RED_S3=s3://<bucket>/<prefijo>}"
[[ -f "$HOME/entorno.ok" ]] || { echo "[campana] antes: spot_entorno.sh"; exit 1; }
source "$HOME/red.env"
RAIZ="$HOME/mejora"
MODELO="$HOME/.cache/vibevoice-nix/modelo"
W="$HOME/campana"
VOCES="${VOCES:-sp-Spk1_man sp-Spk0_woman carlos liliana}"
DIR_VOCES="${DIR_VOCES:-sp-Spk1_man carlos liliana}"
GRUPOS="${GRUPOS:-es,en,es_numeros}"
FRASES="${FRASES:-}"
SEMILLAS="${SEMILLAS:-11,101}"
LAMBDAS="${LAMBDAS:-0,0.05,0.1,0.2}"
BARRIDOS="${BARRIDOS:-condicion/f0_st:ambas:0:es condicion/f0_st:pos:0:es mejor_capa/f0_st:ambas:0:es condicion/sexo:ambas:0:es condicion/f0_st:ambas:40:largas:duraciones}"
DIR_GRUPOS="${DIR_GRUPOS:-es}"
IDENTIDADES="${IDENTIDADES:-carlos=carlos-segura/refs/videoplayback-ref-h0-3.wav liliana=liliana-morales/refs/videoplayback-ref-h1-4.wav}"
DURACIONES="${DURACIONES:-1}"
CONTROL_GRUPOS="${CONTROL_GRUPOS:-en,es_numeros}"   # clips base que hacen de control ECAPA de las voces de serie
DESCRIPTOR="${DESCRIPTOR:-hz}"
SIGNO="${SIGNO:-1}"
PASOS="${PASOS:-instrumentar sondas dirigir jueces}"
SYNC_CADA="${SYNC_CADA:-120}"
ID=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" |
     xargs -I{} curl -s -m 2 -H "X-aws-ec2-metadata-token: {}" http://169.254.169.254/latest/meta-data/instance-id || hostname)
mkdir -p "$W/logs"
cd "$RAIZ"
exec > >(tee -a "$W/logs/campana-$ID.log") 2>&1

subir() { aws s3 sync --only-show-errors --exclude "ids/*" "$W/" "$RED_S3/trabajo/"; }
paso() { echo "[campana] $(date -u +%T) $*"; }

# 1. reanudar: bajar lo hecho y limpiar lo que una interrupcion dejo truncado
paso "bajando $RED_S3/trabajo/ (reanudar)"
aws s3 sync --only-show-errors "$RED_S3/trabajo/" "$W/"
python - "$W" <<'FIN_PY'
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
w = Path(sys.argv[1])
fuera = []
def wav_ok(p):
    try:
        return sf.info(str(p)).frames > 0 and len(sf.read(str(p))[0]) == sf.info(str(p)).frames
    except Exception:
        return False
for npz in sorted((w / "inst").glob("*.npz")):
    ok = wav_ok(npz.with_suffix(".wav"))
    if ok:
        try:
            with np.load(npz, allow_pickle=True) as d:
                ok = d["cond"].shape[0] > 0 and d["res"] is not None
        except Exception:
            ok = False
    if not ok:     # instrumentar escribe .npz y luego .wav: sin los dos sanos, el clip se rehace
        for f in (npz, npz.with_suffix(".wav")):
            f.unlink(missing_ok=True)
        fuera.append(npz.stem)
for wav in sorted((w / "dirigir").glob("*/*.wav")):
    if not wav_ok(wav):
        wav.unlink()
        fuera.append(wav.stem)
hechos = len(list((w / "inst").glob("*.npz"))), len(list((w / "dirigir").glob("*/*.wav")))
print(f"[campana] ya hechos: {hechos[0]} clips instrumentados, {hechos[1]} dirigidos; a rehacer por truncados: {fuera or 'ninguno'}")
FIN_PY

# 2. subir cada SYNC_CADA s mientras se trabaja, y una ultima vez al salir (fin, error o aviso de spot)
( while sleep "$SYNC_CADA"; do subir || true; done ) &
SYNC_PID=$!
final() { kill "$SYNC_PID" 2>/dev/null || true; subir || true; paso "subido a $RED_S3/trabajo/"; }
trap final EXIT
trap 'paso "SIGTERM: ¿aviso de spot? subiendo"; exit 143' TERM

# corpus: el congelado, o sus N primeras frases por grupo (los nombres conservan el indice)
CORPUS="$RAIZ/scripts/corpus_mejora.json"
if [[ -n "$FRASES" ]]; then
  python -c "import json,sys; d=json.load(open('$CORPUS')); json.dump({k:(v[:$FRASES] if isinstance(v,list) else v) for k,v in d.items()}, open('$W/corpus.json','w'), ensure_ascii=False)"
  CORPUS="$W/corpus.json"
fi
CORPUS_TODOS="$RAIZ/scripts/corpus_mejora.json,$RAIZ/scripts/red/corpus_duraciones.json"
ARG_VOCES=(); for v in $VOCES; do ARG_VOCES+=(--voz "$v"); done
ARG_DIR_VOCES=(); for v in $DIR_VOCES; do ARG_DIR_VOCES+=(--voz "$v"); done

for P in $PASOS; do
  case "$P" in
    instrumentar)
      paso "instrumentar: $VOCES · $GRUPOS${FRASES:+ ($FRASES por grupo)} · semillas $SEMILLAS"
      python scripts/red/instrumentar.py --modelo "$MODELO" --voces "$HOME/voces" "${ARG_VOCES[@]}" \
        --corpus "$CORPUS" --grupos "$GRUPOS" --semillas "$SEMILLAS" --salida "$W/inst" 2>&1 | tee -a "$W/logs/instrumentar-$ID.log"
      subir
      if [[ "$DURACIONES" == 1 ]]; then   # duraciones variadas: que la sonda no pueda leer el reloj (docs/bancos/2026-09-27)
        paso "instrumentar: frases largas y cortas (corpus_duraciones.json)"
        python scripts/red/instrumentar.py --modelo "$MODELO" --voces "$HOME/voces" "${ARG_VOCES[@]}" \
          --corpus scripts/red/corpus_duraciones.json --grupos largas,cortas --semillas "$SEMILLAS" --salida "$W/inst" \
          2>&1 | tee -a "$W/logs/instrumentar-$ID.log"
        subir
      fi ;;
    sondas)
      if [[ -f "$W/sondas/direcciones.npz" ]]; then paso "sondas: ya estaban (borra $W/sondas para rehacerlas)"; continue; fi
      paso "sondas"
      python scripts/red/sondas.py --inst "$W/inst" --salida "$W/sondas" 2>&1 | tee "$W/logs/sondas-$ID.log"
      subir ;;
    dirigir)
      for B in $BARRIDOS; do
        IFS=: read -r CL RA DESDE GR CO <<<"$B"
        if [[ "$CL" == mejor_capa/* ]]; then   # la capa con mas R2 por frase apartada para esa etiqueta
          CL=$(python -c "
import json, sys
r = json.load(open('$W/sondas/r2_por_sitio.json'))['r2']['texto']
e = '${CL#*/}'
capas = {s: v[e] for s, v in r.items() if s.startswith('capa') and v.get(e) is not None}
print(f'{max(capas, key=capas.get)}/{e}')")
        fi
        NPZ=""
        for f in "$W/sondas/direcciones.npz" "$W/prefijos_direcciones.npz"; do
          [[ -f "$f" ]] && python -c "import numpy as np,sys; sys.exit(0 if '$CL' in np.load('$f').files else 1)" && { NPZ="$f"; break; }
        done
        if [[ -z "$NPZ" ]]; then paso "dirigir: SALTO $CL (ninguna direcciones.npz la tiene)"; continue; fi
        CORP="$CORPUS"; [[ "${CO:-}" == duraciones ]] && CORP="$RAIZ/scripts/red/corpus_duraciones.json"
        D="$W/dirigir/${CL//\//_}__$RA"; [[ "$DESDE" -gt 0 ]] && D="${D}__desde$DESDE"
        paso "dirigir: $CL, rama $RA, desde $DESDE, grupos $GR, lambdas $LAMBDAS -> $(basename "$D")"
        python scripts/red/dirigir.py --modelo "$MODELO" --voces "$HOME/voces" "${ARG_DIR_VOCES[@]}" \
          --direcciones "$NPZ" --clave "$CL" --lambdas "$LAMBDAS" --rama "$RA" --desde "$DESDE" \
          --corpus "$CORP" --grupos "${GR:-$DIR_GRUPOS}" --semillas "$SEMILLAS" --salida "$D" \
          2>&1 | tee -a "$W/logs/dirigir-$ID.log"
        subir
      done ;;
    jueces)
      # la puerta de I2 tal cual (puerta_dirigir.py): juez_lote.py (whisper large-v3 int8, ECAPA, UTMOS) y la puerta
      # pareada. ECAPA contra audio real: las refs de dobla si la voz esta en IDENTIDADES; si no (voces de serie),
      # el centroide de sus propios clips base de los grupos que no se dirigen (como en el Mac).
      python - "$W" "$IDENTIDADES" "$CONTROL_GRUPOS" "$DIR_VOCES" <<'FIN_PY'
import sys
from pathlib import Path
w, ident, control, voces = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), sys.argv[4].split()
ident = dict(x.split("=") for x in ident.split()) if ident else {}
for voz in voces:
    d = w / "ids" / voz
    d.mkdir(parents=True, exist_ok=True)
    if voz in ident:   # SOLO la referencia apartada al clonar
        fuentes = [Path.home() / "voces_refs" / ident[voz]]
        assert fuentes[0].exists(), f"falta la referencia apartada de {voz}: {fuentes[0]}"
    else:
        fuentes = [f for f in sorted((w / "inst").glob(f"{voz}__*.wav"))
                   if f.stem.split("__")[1].rstrip("0123456789") in control]
    for f in fuentes:
        (d / f.name).unlink(missing_ok=True)
        (d / f.name).symlink_to(f)
    print(f"[jueces] control ECAPA de {voz}: {len(fuentes)} audios ({'la referencia apartada' if voz in ident else 'clips base de otros grupos'})")
FIN_PY
      for D in "$W"/dirigir/*/; do
        D="${D%/}"
        paso "jueces: $(basename "$D")"
        python scripts/red/puerta_dirigir.py lote "$D" --corpus "$CORPUS_TODOS"
        python scripts/juez_lote.py "$D/lote.json" "$D/juez.json" --identidades "$W/ids" --hilos "$OMP_NUM_THREADS" \
          2>&1 | tee -a "$W/logs/jueces-$ID.log"
        subir
        python scripts/red/puerta_dirigir.py puerta "$D" --descriptor "$DESCRIPTOR" --signo "$SIGNO" --corpus "$CORPUS_TODOS" \
          2>&1 | tee -a "$W/logs/puerta-$ID.log"
        if [[ "$D" =~ __desde([0-9]+)$ ]]; then
          python scripts/red/costura.py "$D" --desde "${BASH_REMATCH[1]}" --corpus "$CORPUS_TODOS" --hilos "$OMP_NUM_THREADS" \
            2>&1 | tee -a "$W/logs/costura-$ID.log"
          subir
        fi
      done
      subir ;;
    jueces_rapidos)
      # criba barata antes de la puerta: whisper small int8 y ECAPA, pareados frente a lambda 0; sin UTMOS
      paso "jueces rapidos en CPU (whisper small int8, ECAPA)"
      python - "$W" "$CORPUS" "$IDENTIDADES" <<'FIN_PY' 2>&1 | tee -a "$W/logs/jueces-$ID.log"
import json, os, re, sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, "scripts")
import juez_lote as JL
import perfil_vocal as PV
from normalizar_texto import normalizar
w, corpus = Path(sys.argv[1]), json.load(open(sys.argv[2]))
ident = dict(x.split("=") for x in sys.argv[3].split()) if sys.argv[3] else {}
sal = w / "jueces_rapidos.json"
med = json.loads(sal.read_text()) if sal.exists() else {}
clips = sorted((w / "dirigir").glob("*/*.wav"))
pend = [c for c in clips if f"{c.parent.name}/{c.stem}" not in med]
print(f"[jueces] {len(clips)} clips, {len(pend)} pendientes")
if pend:
    import torch
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    hilos = int(os.environ.get("OMP_NUM_THREADS", "4"))
    wh = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=hilos)
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / ".cache/asistente-huellas/ecapa"))
    def huella(x16):
        with torch.inference_mode():
            e = ecapa.encode_batch(torch.from_numpy(np.ascontiguousarray(x16[:16000 * 30]))[None]).squeeze().numpy()
        return e / (np.linalg.norm(e) + 1e-12)
    centros = {}
    for voz, idn in ident.items():   # centroide de las refs REALES de la persona (como juez_lote.py)
        hs = []
        for f in [Path.home() / "voces_refs" / idn]:   # la referencia apartada al clonar
            x, hz = sf.read(str(f), dtype="float32")
            x16 = JL.a16(x, hz)
            hs += [huella(x16[k:k + 16000 * 8]) for k in range(0, max(1, len(x16) - 16000 * 3), 16000 * 8)
                   if len(x16[k:k + 16000 * 8]) > 16000 * 2]
        if hs:
            v = np.mean(hs, 0)
            centros[voz] = v / np.linalg.norm(v)
    print(f"[jueces] centroides reales: {sorted(centros) or 'ninguno (solo ECAPA frente a lambda 0)'}")
    for n, c in enumerate(pend, 1):
        voz, fr, sem, lam = re.match(r"(.+)__([a-z_]+\d+)__s(\d+)__l([\d.]+q?)$", c.stem).groups()
        g, k = re.match(r"([a-z_]+?)(\d+)$", fr).groups()
        texto, idioma = corpus[g][int(k)], g.split("_")[0]
        x, hz = sf.read(str(c), dtype="float32")
        x16 = JL.a16(x, hz)
        segs, _ = wh.transcribe(x16, language=idioma, beam_size=5, condition_on_previous_text=False, temperature=0.0)
        oido = " ".join(s.text for s in segs).strip()
        p = PV.perfil(x, hz, texto)
        h = huella(x16)
        med[f"{c.parent.name}/{c.stem}"] = dict(
            sitio=c.parent.name, voz=voz, frase=fr, semilla=int(sem), lam=lam, dur=round(len(x16) / 16000, 3), oido=oido,
            wer=round(JL.wer(normalizar(texto, idioma), normalizar(oido, idioma)), 4),
            ecapa=round(float(h @ centros[voz]), 4) if voz in centros else None,
            huella=[round(float(v), 5) for v in h], hz=p.get("hz"), recorrido=p.get("recorrido"))
        if n % 5 == 0 or n == len(pend):
            sal.write_text(json.dumps(med, ensure_ascii=False))
            print(f"[jueces] {n}/{len(pend)}", flush=True)
# resumen pareado frente a lambda 0 de la misma (sitio, voz, frase, semilla)
base = {(m["sitio"], m["voz"], m["frase"], m["semilla"]): m for m in med.values() if m["lam"] == "0"}
filas = {}
for m in med.values():
    b = base.get((m["sitio"], m["voz"], m["frase"], m["semilla"]))
    if b is None or m["lam"] == "0":
        continue
    f = filas.setdefault((m["sitio"], m["lam"]), dict(dwer=[], decapa=[], cos_base=[], dhz_st=[], rotos=0))
    f["dwer"].append(100 * (m["wer"] - b["wer"]))
    if m["ecapa"] is not None and b["ecapa"] is not None:
        f["decapa"].append(m["ecapa"] - b["ecapa"])
    f["cos_base"].append(float(np.dot(m["huella"], b["huella"])))
    if m["hz"] and b["hz"]:
        f["dhz_st"].append(12 * np.log2(m["hz"] / b["hz"]))
    f["rotos"] += int(m["wer"] > 0.25 and b["wer"] <= 0.10)
res = {}
print(f"{'sitio':32} {'lam':>5} {'n':>3} {'dWER pts':>9} {'dECAPA':>8} {'cos base':>8} {'dtono st':>8} {'rotos':>5}  puerta")
for (sitio, lam), f in sorted(filas.items()):
    mw = float(np.mean(f["dwer"]))
    me = float(np.mean(f["decapa"])) if f["decapa"] else None
    pasa = mw <= 0.5 and f["rotos"] == 0 and (me is None or me >= -0.005)
    res[f"{sitio}/{lam}"] = dict(n=len(f["dwer"]), dwer_pts=round(mw, 2), decapa=None if me is None else round(me, 4),
                               cos_base=round(float(np.mean(f["cos_base"])), 4),
                               dtono_st=round(float(np.mean(f["dhz_st"])), 3) if f["dhz_st"] else None,
                               rotos=f["rotos"], pasa=pasa)
    r = res[f"{sitio}/{lam}"]
    print(f"{sitio:32} {lam:>5} {r['n']:>3} {r['dwer_pts']:>9} {str(r['decapa']):>8} {r['cos_base']:>8} {str(r['dtono_st']):>8} "
          f"{r['rotos']:>5}  {'pasa' if pasa else 'NO'}{'' if me is not None else ' (sin centroide: ECAPA sin mirar)'}")
(w / "jueces_rapidos_resumen.json").write_text(json.dumps(res, indent=1))
FIN_PY
      subir ;;
    paridad)
      # el clip de sp-Spk1_man, es0, semilla 11 de esta maquina frente al del Mac (~/patron, de $RED_S3/patron/)
      paso "paridad frente al clip del Mac"
      python - "$W" "$ID" <<'FIN_PY' 2>&1 | tee -a "$W/logs/paridad-$ID.log"
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import soundfile as sf
w, id_ = Path(sys.argv[1]), sys.argv[2]
res = {}
for pat in sorted((Path.home() / "patron").glob("*.wav")):
    mio = w / "inst" / pat.name
    if not mio.exists():
        print(f"[paridad] {pat.name}: no lo he generado (¿falta el paso instrumentar?)")
        continue
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    a, hz = sf.read(str(pat), dtype="float64")
    b, _ = sf.read(str(mio), dtype="float64")
    r = dict(md5_mac=md5(pat), md5_aqui=md5(mio), muestras_mac=len(a), muestras_aqui=len(b))
    r["identico"] = r["md5_mac"] == r["md5_aqui"]
    if len(a) == len(b):
        ruido = np.sum((a - b) ** 2)
        r["snr_db"] = round(float(10 * np.log10(np.sum(a ** 2) / ruido)), 1) if ruido > 0 else float("inf")
        r["dif_max"] = float(np.abs(a - b).max())
    # el md5 cambia con la CPU y hasta con el numero de hilos (en el mismo Mac, 3 hilos: 1 paso de PCM16, 3e-5):
    # pasa si es identico, o SNR > 60 dB, o misma longitud y diferencia maxima <= 1e-3
    r["pasa"] = r["identico"] or r.get("snr_db", 0) > 60 or r.get("dif_max", 1) <= 1e-3
    npz = mio.with_suffix(".npz")
    if npz.exists():
        with np.load(npz, allow_pickle=True) as d:
            r["fotogramas_aqui"] = int(d["cond"].shape[0])
    # RTF de esta maquina: del log de instrumentar (segundos de reloj por segundo de audio del clip)
    for log in (w / "logs").glob("instrumentar-*.log"):
        for l in log.read_text().splitlines():
            m = re.match(rf"{re.escape(mio.stem)}: (\d+) fotogramas, ([\d.]+) s, (\d+) s", l)
            if m:
                r.setdefault("rtf", {})[log.stem.split("-", 1)[1]] = round(int(m.group(3)) / float(m.group(2)), 2)
    res[pat.name] = r
    print(f"[paridad] {pat.name}: {'IDENTICO (md5)' if r['identico'] else 'distinto'}; muestras Mac {r['muestras_mac']} / aqui "
          f"{r['muestras_aqui']}; SNR {r.get('snr_db', 'n/a (otra duracion)')} dB; dif max {r.get('dif_max', 'n/a')}; "
          f"{'PASA' if r['pasa'] else 'NO PASA'}; RTF {r.get('rtf')}")
(w / f"paridad-{id_}.json").write_text(json.dumps(res, indent=1))
FIN_PY
      ;;
    *) echo "[campana] paso desconocido: $P"; exit 1 ;;
  esac
done
paso "hecho: $PASOS"
