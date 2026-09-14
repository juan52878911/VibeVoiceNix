#!/usr/bin/env bash
# 0.2 repetida: LM TTS aislado, 6 rondas, con voz-stream parado, y la regla de agregación del plan.
#
#   ssh -n root@voz 'systemd-run --unit fase0-lm --collect bash /root/lab/cli/fase0_lm.sh'
#
# Para voz-stream y lo devuelve al salir, falle lo que falle.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep; do
  command -v "$orden" >/dev/null || { echo "[lm] falta $orden: aborto"; exit 1; }
done

PY=/nix/store/68d5yzqf9iqgvfwn2lv4hky2khin1i0f-vibevoice-env/bin/python3.12
export OMP_NUM_THREADS=6 MALLOC_ARENA_MAX=2 PYTHONWARNINGS=ignore
cd /root/lab || exit 1

if pgrep -fa 'fase[0-9]_|ws_fidelidad|benchmark_app' | grep -v -e pgrep -e fase0_lm; then
  echo "[lm] hay otra carga en la VM: no se toca nada"
  exit 1
fi

devolver() {
  systemctl start voz-stream voz-stream-sin-swap
  echo "[lm] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT

systemctl stop voz-stream-sin-swap voz-stream
sleep 5
$PY cli/lab_fase0.py lm --rondas 6 --salida lm6.json
$PY cli/lab_fase0.py resumen_lm lm6.json
echo "[lm] FIN"
