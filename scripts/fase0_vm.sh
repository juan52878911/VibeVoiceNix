#!/usr/bin/env bash
# Fase 0 en la VM voz: carga vieja y nueva con su huella, LM TTS aislado y LM de texto.
#
#   scp -r ... root@voz:/root/lab/ && ssh -n root@voz 'systemd-run --unit fase0-lab bash /root/lab/cli/fase0_vm.sh'
#   journalctl -u fase0-lab -f
#
# Para voz-stream (la carga vieja pica 4,4 GB y la VM tiene 4,9) y lo devuelve al salir, falle lo que falle.
set -uo pipefail
# systemd-run arranca sin PATH: sin esto pgrep, sleep y systemctl "no existen", el guardia no guarda
# y el trap no devuelve nada (paso la primera vez, 14-09-2026).
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep free; do
  command -v "$orden" >/dev/null || { echo "[fase0] falta $orden: aborto"; exit 1; }
done

PY=/nix/store/68d5yzqf9iqgvfwn2lv4hky2khin1i0f-vibevoice-env/bin/python3.12
export VIBEVOICE_MODELO=/nix/store/ixp4bcqfiazpgkfw82x326qjzh26mrsk-vibevoice-realtime-0.5b
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MALLOC_ARENA_MAX=2 PYTHONWARNINGS=ignore
cd /root/lab || exit 1

if pgrep -fa 'fase[0-9]_' | grep -v -e pgrep -e fase0_vm; then
  echo "[fase0] hay un experimento de otra sesion corriendo: no se toca nada"
  exit 1
fi

devolver() {
  systemctl start voz-stream voz-stream-sin-swap
  echo "[fase0] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT

systemctl stop voz-stream-sin-swap voz-stream
sleep 3
free -m

echo "[fase0] 0.1 carga vieja"
$PY cli/lab_fase0.py carga --modo viejo --codigo ov --huella viejo.json
echo "[fase0] 0.1 carga nueva"
$PY cli/lab_fase0.py carga --modo nuevo --codigo ov --huella nuevo.json
echo "[fase0] huella"
$PY cli/lab_fase0.py comparar viejo.json nuevo.json
echo "[fase0] 0.2 LM TTS aislado"
$PY cli/lab_fase0.py lm --salida lm.json
echo "[fase0] 0.2b LM de texto"
$PY cli/lab_fase0.py lm_texto --codigo ov
echo "[fase0] FIN"
