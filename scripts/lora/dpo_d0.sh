#!/usr/bin/env bash
# D0 del plan de preferencias en la instancia GPU: 50 textos x 4 semillas con la base, juzgados con la
# recompensa (whisper medium, ECAPA, UTMOS) y puntuados. Todo queda en ~/dpo/d0.
#   nohup bash ~/mejora/scripts/lora/dpo_d0.sh > ~/dpo/d0.log 2>&1 &
set -euo pipefail
source /opt/pytorch/bin/activate
cd ~/mejora/scripts/lora
D=~/dpo/d0
mkdir -p $D
[[ -f $D/hablantes.json ]] || python dpo_generar.py --salida $D --hablantes en:3,es:3,fr:1,de:1,it:1,pt:1 \
    --por-hablante 5 --preparar 2>&1 | grep -v "newly initialized"
python dpo_generar.py --salida $D --trozo 0/2 > $D/gen0.log 2>&1 &
python dpo_generar.py --salida $D --trozo 1/2 > $D/gen1.log 2>&1 &
wait
python -c "import json,glob; json.dump([c for f in sorted(glob.glob('$D/lote.*.json')) for c in json.load(open(f))], open('$D/lote.json','w'), ensure_ascii=False)"
python ../juez_lote.py $D/lote.json $D/medidas.json --identidades $D/identidades --dispositivo cuda \
    --whisper medium --rapido 2>&1 | grep "\[juez\]"
python dpo_puntuar.py $D
echo D0_LISTO
