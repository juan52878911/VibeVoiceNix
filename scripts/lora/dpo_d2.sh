#!/usr/bin/env bash
# D2 del plan de preferencias en la GPU: barrido de beta (receta de la F8: rango 16, lr 5e-5, 600 pasos,
# acumulacion 4) con dos entrenamientos a la vez, y la base de la puerta D3 generada mientras entrena el
# tercero (en la T4 caben dos procesos grandes). La puerta se juzga con whisper large-v3.
#   setsid -f bash dpo_d2.sh > ~/dpo/d2.log 2>&1 < /dev/null
set -uo pipefail
source /opt/pytorch/bin/activate
cd ~/mejora/scripts/lora
E="python dpo_entrenar.py --datos $HOME/dpo/d1,$HOME/dpo/d0 --pasos 600 --acumular 4 --lr 5e-5 --cada 50"
$E --salida ~/dpo/b500 --beta 500 2>&1 | grep --line-buffered "\[dpo\]" > ~/dpo/b500.log &
$E --salida ~/dpo/b2000 --beta 2000 2>&1 | grep --line-buffered "\[dpo\]" > ~/dpo/b2000.log &
wait
$E --salida ~/dpo/b5000 --beta 5000 2>&1 | grep --line-buffered "\[dpo\]" > ~/dpo/b5000.log &
python dpo_generar.py --salida ~/dpo/pb --semillas 11,22,33 > ~/dpo/pb/gen0.log 2>&1 &
wait
python -c "import json,glob; json.dump([c for f in sorted(glob.glob('$HOME/dpo/pb/lote.*.json')) for c in json.load(open(f))], open('$HOME/dpo/pb/lote.json','w'), ensure_ascii=False)"
python ../juez_lote.py ~/dpo/pb/lote.json ~/dpo/pb/medidas_puerta.json --identidades ~/dpo/pb/identidades \
    --dispositivo cuda --rapido 2>&1 | grep "\[juez\]" | tail -2
echo FASE2_LISTA
