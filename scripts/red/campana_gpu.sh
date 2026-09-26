#!/usr/bin/env bash
# La campana de la red por dentro en la g4dn (bajo demanda), terminando lo que el Mac dejo en ~/Documents/red-interna.
# Corre DENTRO de la maquina, despues de scripts/lora/gpu_entorno.sh y de bajar la carpeta de S3:
#
#   RED_S3=s3://<bucket>/red-interna nohup bash ~/mejora/scripts/red/campana_gpu.sh > ~/campana.out 2>&1 &
#
# Dos carriles a la vez:
#   sintesis (GPU, torch fp32, RED_DISPOSITIVO=cuda): mecanica en cuda, I0 de largas y cortas, los barridos de
#       dirigir.py (sexo rehecho entero, tono en la condicion en dos ramas, tono en la mejor capa, tono --desde 40)
#       y atencion --detalle. Cada barrido deja <carpeta>/.hecho al acabar.
#   medida (CPU y GPU): sondas.py en cuanto esta I0; juez_lote.py y puerta_dirigir.py sobre inst/ y sobre cada
#       barrido cuando tiene su .hecho; costura.py en el de --desde; sorpresa_wer.py al final.
# Reutilizacion (regla del 26-09): I0 del Mac vale tal cual; de dirigir solo vale una clave con TODOS sus lambda
# del Mac. sexo_ambas estaba a medias (liliana 11 de 32): se aparta a dirigir_mac/ y se rehace entera aqui, para
# que lambda = 0 y lambda > 0 salgan de la misma maquina. Sube a S3 cada 3 min y al salir; se reanuda relanzando.
set -uo pipefail
: "${RED_S3:?falta RED_S3=s3://<bucket>/red-interna}"
source /opt/pytorch/bin/activate
export RED_DISPOSITIVO=cuda PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
R="$HOME/Documents/red-interna"
M="$HOME/.cache/vibevoice-nix/modelo"
L="$R/logs"
DUR=scripts/red/corpus_duraciones.json
CORPUS2="scripts/corpus_mejora.json,$DUR"
V4=(--voz sp-Spk1_man --voz sp-Spk0_woman --voz carlos --voz liliana)
V3=(--voz sp-Spk1_man --voz carlos --voz liliana)
COMUN=(--lambdas 0,0.05,0.1,0.2 --semillas 11,101)
cd "$HOME/mejora"
mkdir -p "$L"
paso() { echo "[gpu] $(date -u +%T) $*"; }
subir() { aws s3 sync --only-show-errors "$R/" "$RED_S3/"; }
( while sleep 180; do subir || true; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; subir; paso "subido"' EXIT

# ---------- 0. la mecanica en cuda: si no da 8/8, no se sintetiza nada ----------
if [[ ! -f "$L/mecanica_cuda.ok" ]]; then
  python scripts/red/probar_mecanica.py --voz "$R/voces/sp-Spk1_man.pt" --modelo "$M" 2>&1 | tee "$L/mecanica_cuda.log" | grep -a "^[a-z]*: \|RESUMEN"
  RES=$(grep -a '^RESUMEN' "$L/mecanica_cuda.log" || true)
  [[ $(grep -o "'pasa'" <<<"$RES" | wc -l) -eq 8 ]] || { paso "MECANICA EN CUDA NO PASA: $RES"; exit 5; }
  touch "$L/mecanica_cuda.ok"
fi

# sexo_ambas del Mac, a medias: fuera de dirigir/ (aqui y en S3) antes de nada
if [[ -d "$R/dirigir/sexo_ambas" && ! -f "$R/dirigir/sexo_ambas/.gpu" ]]; then
  mkdir -p "$R/dirigir_mac" && mv "$R/dirigir/sexo_ambas" "$R/dirigir_mac/sexo_ambas"
  aws s3 mv --only-show-errors --recursive "$RED_S3/dirigir/sexo_ambas/" "$RED_S3/dirigir_mac/sexo_ambas/"
  paso "sexo_ambas del Mac apartado a dirigir_mac/ (se rehace entero en la GPU)"
fi

barrido() {   # barrido <carpeta> <args de dirigir.py...>
  local D="$R/dirigir/$1"; shift
  [[ -f "$D/.hecho" ]] && { paso "dirigir $(basename "$D"): ya estaba"; return; }
  mkdir -p "$D" && touch "$D/.gpu"
  paso "dirigir $(basename "$D")"
  python scripts/red/dirigir.py --modelo "$M" --voces "$R/voces" "${V3[@]}" "${COMUN[@]}" --salida "$D" "$@" \
    >> "$L/dirigir_$(basename "$D").log" 2>&1 && touch "$D/.hecho" || paso "dirigir $(basename "$D") FALLO (ver logs)"
}

sintesis() {
  paso "I0: largas y cortas, 4 voces, semillas 11 y 101"
  python scripts/red/instrumentar.py --modelo "$M" --voces "$R/voces" "${V4[@]}" --corpus "$DUR" --grupos largas,cortas \
    --semillas 11,101 --salida "$R/inst" >> "$L/instrumentar_gpu.log" 2>&1 || paso "I0 FALLO (ver log): se sigue con lo que haya"
  touch "$R/inst/.hecho"
  barrido sexo_ambas --direcciones "$R/prefijos_direcciones.npz" --clave condicion/sexo --rama ambas --grupos es
  while [[ ! -f "$R/sondas/.hecho" && ! -f "$R/sondas/.fallo" ]]; do sleep 20; done
  if [[ -f "$R/sondas/.fallo" ]]; then paso "sin sondas no hay direccion de tono: solo atencion"; else
  CAPA=$(python - "$R/sondas/r2_por_sitio.json" <<'FIN_PY'
import json, sys
r = json.load(open(sys.argv[1]))["r2"]["texto"]
capas = {s: v["f0_st"] for s, v in r.items() if s.startswith("capa") and v.get("f0_st") is not None}
print(max(capas, key=capas.get))
FIN_PY
)
  paso "mejor capa para f0_st por frase apartada: $CAPA"
  local DIRS="$R/sondas/direcciones.npz"
  barrido f0_ambas --direcciones "$DIRS" --clave condicion/f0_st --rama ambas --grupos es
  barrido f0_pos --direcciones "$DIRS" --clave condicion/f0_st --rama pos --grupos es
  barrido "${CAPA}_f0_ambas" --direcciones "$DIRS" --clave "$CAPA/f0_st" --rama ambas --grupos es
  barrido f0_ambas_desde40 --direcciones "$DIRS" --clave condicion/f0_st --rama ambas --desde 40 --corpus "$DUR" --grupos largas
  fi
  for v in sp-Spk1_man sp-Spk0_woman carlos liliana; do
    [[ -f "$R/atencion/${v}_detalle.json" ]] && continue
    python scripts/red/atencion.py --modelo "$M" --voces "$R/voces" --voz "$v" --detalle --salida "$R/atencion/${v}_detalle.json" \
      >> "$L/atencion_detalle.log" 2>&1
  done
  touch "$R/.sintesis_hecha"
  paso "sintesis hecha"
}

juzgar() {   # juzgar <carpeta>: lote, juez_lote (large-v3 fp16, ECAPA, UTMOS en la GPU), puerta
  local D="$1"
  python scripts/red/puerta_dirigir.py lote "$D" --corpus "$CORPUS2" >> "$L/jueces.log" 2>&1
  python scripts/juez_lote.py "$D/lote.json" "$D/juez.json" --identidades "$R/ids" --dispositivo cuda >> "$L/jueces.log" 2>&1
}

medida() {
  while [[ ! -f "$R/inst/.hecho" ]]; do sleep 20; done
  if [[ ! -f "$R/sondas/.hecho" ]]; then
    paso "sondas (I1), pliegues por frase y por voz"
    python scripts/red/sondas.py --inst "$R/inst" --salida "$R/sondas" > "$L/sondas.log" 2>&1 && touch "$R/sondas/.hecho" \
      || { paso "SONDAS FALLARON (ver $L/sondas.log)"; mkdir -p "$R/sondas"; touch "$R/sondas/.fallo"; }
  fi
  # control de ECAPA de sp-Spk0_woman, como el de sp-Spk1_man: sus clips base de en y es_numeros
  mkdir -p "$R/ids/sp-Spk0_woman"
  for f in "$R"/inst/sp-Spk0_woman__en[0-9]__s*.wav "$R"/inst/sp-Spk0_woman__es_numeros[0-9]__s*.wav; do
    [[ -e "$f" ]] && cp -n "$f" "$R/ids/sp-Spk0_woman/"
  done
  paso "jueces de I0 (inst/)"
  juzgar "$R/inst"
  paso "sorpresa frente al WER"
  python scripts/red/sorpresa_wer.py --modelo "$M" --inst "$R/inst" --juez "$R/inst/juez.json" --salida "$R/sorpresa.json" \
    > "$L/sorpresa.log" 2>&1 || paso "sorpresa_wer FALLO (ver log)"
  local hechas=" "
  while :; do
    local fin=0
    [[ -f "$R/.sintesis_hecha" ]] && fin=1   # si ya estaba al empezar la pasada, esta pasada ve todos los .hecho
    for D in "$R"/dirigir/*/; do
      D="${D%/}"
      [[ -f "$D/.hecho" && "$hechas" != *" $(basename "$D") "* ]] || continue
      paso "jueces y puerta de $(basename "$D")"
      juzgar "$D"
      python scripts/red/puerta_dirigir.py puerta "$D" --descriptor hz --signo 1 --corpus "$CORPUS2" > "$D/puerta.txt" 2>&1
      if [[ "$D" == *desde40 ]]; then
        python scripts/red/costura.py "$D" --desde 40 --corpus "$CORPUS2" --dispositivo cuda > "$D/costura.txt" 2>&1
      fi
      hechas="$hechas$(basename "$D") "
      subir || true
    done
    [[ $fin == 1 ]] && break
    sleep 30
  done
  paso "medida hecha"
}

sintesis &
PS=$!
medida &
PM=$!
wait $PS $PM
paso "campana acabada"
