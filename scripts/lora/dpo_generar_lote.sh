#!/usr/bin/env bash
# Un lote de DPO en la instancia GPU: elegir lectores y textos, generar en dos procesos (en la T4 caben
# dos grandes), juzgar y puntuar. D0, D1 y la puerta D3 son el mismo guion con otros argumentos.
#   nohup bash dpo_generar_lote.sh ~/dpo/d1 "--hablantes en:5,es:5,fr:3,de:3,it:2,pt:2 --por-hablante 15" \
#         "--whisper medium --rapido" [--solo-generar] "[argumentos de generacion: --lora ... --cuantizar]" > d1.log 2>&1 &
# Con --solo-generar no elige lectores (hablantes.json ya esta en la carpeta: la puerta del LoRA usa la de la base).
set -euo pipefail
source /opt/pytorch/bin/activate
cd ~/mejora/scripts/lora
D="$1"; PREP="$2"; JUEZ="$3"; MODO="${4:-}"; GEN="${5:-}"
mkdir -p "$D"
if [[ "$MODO" != "--solo-generar" && ! -f "$D/hablantes.json" ]]; then
  python dpo_generar.py --salida "$D" $PREP --preparar 2>&1 | grep "\[gen\]"
fi
python dpo_generar.py --salida "$D" --trozo 0/2 $GEN > "$D/gen0.log" 2>&1 &
python dpo_generar.py --salida "$D" --trozo 1/2 $GEN > "$D/gen1.log" 2>&1 &
wait
grep -h "\[gen\].*clips" "$D"/gen*.log
python -c "import json,glob; json.dump([c for f in sorted(glob.glob('$D/lote.*.json')) for c in json.load(open(f))], open('$D/lote.json','w'), ensure_ascii=False)"
M=medidas.json; [[ "$JUEZ" == *medium* ]] || M=medidas_puerta.json
python ../juez_lote.py "$D/lote.json" "$D/$M" --identidades "$D/identidades" --dispositivo cuda $JUEZ 2>&1 | grep "\[juez\]" | tail -2
[[ "$M" == medidas.json ]] && python dpo_puntuar.py "$D"
echo LOTE_LISTO
