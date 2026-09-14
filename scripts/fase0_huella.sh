#!/usr/bin/env bash
# Puerta previa de A1+A2: huella de los tensores torch con la carga vieja y con la nueva.
#
#   ssh -n root@voz 'systemd-run --unit fase0-huella --collect bash /root/lab/cli/fase0_huella.sh'
#
# Para voz-stream (la carga vieja pica 4,3 GB) y lo devuelve al salir.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep; do
  command -v "$orden" >/dev/null || { echo "[huella] falta $orden: aborto"; exit 1; }
done

PY=/nix/store/68d5yzqf9iqgvfwn2lv4hky2khin1i0f-vibevoice-env/bin/python3.12
export VIBEVOICE_MODELO=/nix/store/ixp4bcqfiazpgkfw82x326qjzh26mrsk-vibevoice-realtime-0.5b
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MALLOC_ARENA_MAX=2 PYTHONWARNINGS=ignore
cd /root/lab || exit 1

if pgrep -fa 'fase[0-9]_' | grep -v -e pgrep -e fase0_; then
  echo "[huella] hay un experimento de otra sesion corriendo: no se toca nada"
  exit 1
fi

devolver() {
  systemctl start voz-stream voz-stream-sin-swap
  echo "[huella] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT

systemctl stop voz-stream-sin-swap voz-stream
sleep 3
$PY cli/lab_fase0.py carga --modo viejo --codigo ov --huella viejo.json
$PY cli/lab_fase0.py carga --modo nuevo --codigo ov --huella nuevo.json
$PY cli/lab_fase0.py comparar viejo.json nuevo.json
echo "[huella] FIN"
