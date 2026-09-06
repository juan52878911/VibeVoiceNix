#!/usr/bin/env bash
# Crea la VM `taller` en Proxmox como clon de `voz` y le aplica su
# configuracion. Se ejecuta UNA vez, desde el Mac, con voz apagada.
#
#   scripts/crear_taller.sh          clona 210 -> 220 y la configura
#
# Pasos:
#   1. qm clone 210 220 (clon completo en `local`, qcow2: local-lvm solo
#      tiene 45 GB libres y el disco son 40)
#   2. 11 GB de RAM sin balloon, 6 nucleos, cpuunits 512 (por debajo de
#      AuraCRM, que va con los 1024 por defecto)
#   3. arranca: sube como "voz" en 192.168.2.54 (voz esta apagada, no choca)
#   4. nixos-rebuild switch --flake .#taller sobre esa IP: cambia nombre, IP a
#      .55, apaga los servicios de voz e instala Qwen3. Construye en la propia
#      VM (--build-host): el Mac no puede construir x86_64-linux.
set -euo pipefail

PVE="${TALLER_PVE:-pve}"
ORIGEN="${TALLER_VOZ_VMID:-210}"
TALLER="${TALLER_VMID:-220}"
IP_INICIAL="${TALLER_IP_INICIAL:-192.168.2.54}"
IP_FINAL="${TALLER_IP:-192.168.2.55}"

pve() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$PVE" "$@"; }

if pve "qm status $TALLER" >/dev/null 2>&1; then
  echo "la VM $TALLER ya existe; salto el clon"
else
  [[ "$(pve "qm status $ORIGEN" | awk '{print $2}')" == "stopped" ]] || {
    echo "la VM $ORIGEN tiene que estar apagada para clonarla (scripts/modo_taller.sh on)"; exit 1; }
  pve "qm clone $ORIGEN $TALLER --name taller --full 1 --storage local --format qcow2"
  pve "qm set $TALLER --memory 11264 --balloon 0 --cores 6 --cpuunits 512"
fi

pve "qm start $TALLER"
echo "esperando SSH en $IP_INICIAL..."
for _ in $(seq 1 60); do
  ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no "root@$IP_INICIAL" true 2>/dev/null && break
  sleep 5
done

# La clave del host es la de voz (mismo disco): evitar el aviso de MITM.
ssh-keygen -R "$IP_INICIAL" >/dev/null 2>&1 || true
ssh-keygen -R "$IP_FINAL" >/dev/null 2>&1 || true

nixos-rebuild switch --flake .#taller \
  --target-host "root@$IP_INICIAL" --build-host "root@$IP_INICIAL" \
  --use-remote-sudo=false 2>&1 | tail -20

echo "taller aplicada; deberia responder en $IP_FINAL"
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "root@$IP_FINAL" \
  'hostname; free -g | head -2; nproc; echo QWEN3TTS_MODELO=$QWEN3TTS_MODELO'
