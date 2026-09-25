#!/usr/bin/env bash
# D2 del plan de preferencias en la GPU: barrido de beta (receta de la F8: rango 16, lr 5e-5, 600 pasos,
# acumulacion 4) y la base de la puerta D3, juzgada con whisper large-v3.
# Dos entrenamientos a la vez NO caben en la T4 (25-09: el segundo murio por memoria de CUDA; cada uno llega
# a ~8 GB con la cache de condiciones de referencia): uno entrena y, a la vez, se genera la puerta.
#   setsid -f bash dpo_d2.sh [betas ya hechas...] > ~/dpo/d2.log 2>&1 < /dev/null
set -uo pipefail
source /opt/pytorch/bin/activate
cd ~/mejora/scripts/lora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
entrenar() {
  python dpo_entrenar.py --datos $HOME/dpo/d1,$HOME/dpo/d0 --pasos 600 --acumular 4 --lr 5e-5 --cada 50 \
    --salida ~/dpo/b$1 --beta $1 2>&1 | grep --line-buffered -E "\[dpo\]|Error|Traceback|Killed" > ~/dpo/b$1.log
}
HECHAS=" $* "
[[ -f ~/dpo/pb/lote.0.json ]] || { python dpo_generar.py --salida ~/dpo/pb --semillas 11,22,33 > ~/dpo/pb/gen0.log 2>&1 & }
for b in 500 2000 5000; do
  [[ "$HECHAS" == *" $b "* ]] || entrenar $b
done
wait
python -c "import json,glob; json.dump([c for f in sorted(glob.glob('$HOME/dpo/pb/lote.*.json')) for c in json.load(open(f))], open('$HOME/dpo/pb/lote.json','w'), ensure_ascii=False)"
python ../juez_lote.py ~/dpo/pb/lote.json ~/dpo/pb/medidas_puerta.json --identidades ~/dpo/pb/identidades \
    --dispositivo cuda --rapido 2>&1 | grep "\[juez\]" | tail -2
echo FASE2_LISTA
