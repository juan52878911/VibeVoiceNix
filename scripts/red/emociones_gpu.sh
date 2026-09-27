#!/usr/bin/env bash
# Emociones graduables en la g4dn (scripts/red/emociones.py). Corre DENTRO, tras scripts/lora/gpu_entorno.sh:
#
#   RED_S3=s3://<bucket>/<prefijo> FASE=piloto nohup bash ~/mejora/scripts/red/emociones_gpu.sh > ~/emo.out 2>&1 &
#
#   FASE=piloto  CREMA-D completo -> extraer (91 actores) -> direcciones -> tres formas de sumar la direccion
#                (condicion rama natural, condicion rama pos, residual de la capa 14) con 2 voces x 2 frases x 5
#                emociones x varios niveles -> juez (emotion2vec+, whisper large-v3, ECAPA, UTMOS, F0/energia/ritmo)
#   FASE=final   con NIVELES='{"HAP": [..], "ANG_rel": [..], ...}' y SITIO/RAMA elegidos en los pilotos:
#                4 voces x 4 frases x cada emocion con sus niveles (las claves son las de direcciones.npz, con los
#                contrastes de `emociones.py contraste`)
#
# Dos sintesis a la vez (la GPU no pasa del 25 % con una) y el juez DESPUES: juez + dos sintesis no caben en la T4.
# Sube ~/emo a S3 cada 3 min y al salir; se reanuda relanzando (cada paso se salta lo hecho).
set -uo pipefail
: "${RED_S3:?falta RED_S3}"
FASE="${FASE:-piloto}"
source /opt/pytorch/bin/activate
export RED_DISPOSITIVO=cuda PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
E="$HOME/emo"; M="$HOME/.cache/vibevoice-nix/modelo"; V="$HOME/voces"
mkdir -p "$E/logs" "$V"
cd "$HOME/mejora"
paso() { echo "[emo] $(date -u +%T) $*"; }
subir() { aws s3 sync --only-show-errors --exclude "crema/*" "$E/" "$RED_S3/emo/"; }
aws s3 sync --only-show-errors --exclude "crema/*" "$RED_S3/emo/" "$E/" || true
( while sleep 180; do subir || true; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null; subir; paso subido' EXIT
PY="python scripts/red/emociones.py"

# voces: las de serie del repo de VibeVoice y los clones con consentimiento de la campana de la red
for v in sp-Spk1_man sp-Spk0_woman; do
  [[ -s "$V/$v.pt" ]] || curl -sSfL -o "$V/$v.pt" "https://github.com/microsoft/VibeVoice/raw/94da20d98b2fa7688e9cbfaf7692ddb4954f7600/demo/voices/streaming_model/$v.pt"
done
aws s3 cp --only-show-errors "$RED_S3/red-interna/voces/carlos.pt" "$V/" || true
aws s3 cp --only-show-errors "$RED_S3/red-interna/voces/liliana.pt" "$V/" || true

gen() {   # gen <carpeta> <args...>
  $PY generar --modelo "$M" --voces "$V" --direcciones "$E/direcciones.npz" --salida "$E/gen/$1" "${@:2}" >> "$E/logs/gen_$1.log" 2>&1
}

if [[ "$FASE" == piloto ]]; then
  if [[ ! -f "$E/direcciones.npz" ]]; then
    if [[ ! -d "$E/crema" ]]; then
      paso "bajando CREMA-D (MahiA/CREMA-D en Hugging Face)"
      python - "$E/crema" <<'FIN_PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("MahiA/CREMA-D", repo_type="dataset", local_dir=sys.argv[1], allow_patterns=["audios/*.wav"], max_workers=16)
FIN_PY
    fi
    paso "extraer: $(ls "$E/crema/audios" | wc -l) clips"
    $PY extraer --crema "$E/crema/audios" --modelo "$M" --salida "$E/extra" > "$E/logs/extraer.log" 2>&1 || { paso "EXTRAER FALLO"; exit 5; }
    $PY direcciones --extra "$E/extra" --salida "$E/direcciones.npz" > "$E/logs/direcciones.log" 2>&1 || { paso "DIRECCIONES FALLO"; exit 5; }
    subir
  fi
  paso "piloto: tres formas de sumar, 2 voces x 2 frases"
  printf '[["es", "Acabo de recibir la noticia y quería contártela antes que a nadie."], ["es", "Mañana vamos a revisar otra vez todo el proyecto desde el principio."]]' > "$E/frases_piloto.json"
  COMUN=(--voz sp-Spk1_man --voz sp-Spk0_woman --frases "$E/frases_piloto.json" --emociones ANG,HAP,SAD,FEA,DIS)
  gen cond_natural "${COMUN[@]}" --sitio condicion --rama natural --niveles 0,0.5,1,1.5,2,3 &
  P1=$!
  gen cond_pos "${COMUN[@]}" --sitio condicion --rama pos --niveles 0,0.25,0.5,1
  gen capa14_natural "${COMUN[@]}" --sitio capa14 --rama natural --niveles 0,0.5,1,2
  wait $P1
  subir
  paso "juez del piloto"
  for c in cond_natural cond_pos capa14_natural; do
    $PY juzgar --carpeta "$E/gen/$c" --dispositivo cuda > "$E/logs/juez_$c.log" 2>&1
    tail -40 "$E/logs/juez_$c.log" | grep -a "^[A-Z][A-Z][A-Z] \|^emocion" > "$E/gen/$c/resumen.txt"
  done
  paso "piloto hecho"
fi

if [[ "$FASE" == final ]]; then
  : "${SITIO:?}" "${RAMA:?}" "${NIVELES:?json {EMO: [n1..n4]}}"
  paso "final: $SITIO rama $RAMA, niveles $NIVELES"
  printf '%s' "$NIVELES" > "$E/niveles_final.json"
  mitad() {   # mitad <voces...>: cada emocion con sus niveles, en una carpeta comun
    for emo in $(python -c "import json;print(' '.join(json.load(open('$E/niveles_final.json'))))"); do
      NV=$(python -c "import json;print(','.join(['0'] + [str(x) for x in json.load(open('$E/niveles_final.json'))['$emo']]))")
      local args=(); for v in "$@"; do args+=(--voz "$v"); done
      gen final "${args[@]}" --emociones "$emo" --niveles "$NV" --sitio "$SITIO" --rama "$RAMA"
    done
  }
  mitad sp-Spk1_man carlos &
  P1=$!
  mitad sp-Spk0_woman liliana
  wait $P1
  subir
  paso "juez final"
  $PY juzgar --carpeta "$E/gen/final" --dispositivo cuda > "$E/logs/juez_final.log" 2>&1
  tail -40 "$E/logs/juez_final.log" | grep -a "^[A-Z][A-Z][A-Z] \|^emocion" > "$E/gen/final/resumen.txt"
  paso "final hecho"
fi
