#!/usr/bin/env bash
# La VM voz hace de taller: más RAM y los servicios de voz parados mientras se entrena.
#
#   scripts/modo_taller_voz.sh on       para app-noticias, sube la VM voz a 11 GB, para voz-stream,
#                                       voz-api y whisper (vuelven solos si se reinicia la VM)
#   scripts/modo_taller_voz.sh off      devuelve la VM voz a 5 GB con sus servicios, arranca app-noticias
#   scripts/modo_taller_voz.sh estado   RAM, servicios y qué corre
#
# POR QUE ESTO Y NO LA VM taller DE crear_taller.sh
# El clon completo de la VM voz no cabe en este host tal como está: el almacenamiento `local` (55 GB
# libres) no admite discos de VM, y `local-lvm` tiene ~17 GB libres para un disco que ocupa ~20.
# Llenar ese pool fino puede dejar sin espacio a las demás VM, AuraCRM incluida. Medido el 13-09-2026
# al intentarlo. Aquí no se clona nada: la misma VM, con más RAM mientras dura el entrenamiento.
#
# QUE NO TOCA NUNCA: la VM 200 (AuraCRM).
set -euo pipefail
HOST="${TALLER_PVE:-pve}"
VOZ="${TALLER_VOZ_VMID:-210}"
CT_NOTICIAS="${TALLER_CT_NOTICIAS:-100}"
IP="${TALLER_IP:-192.168.2.54}"
RAM_TALLER="${TALLER_RAM_MB:-11264}"
RAM_VOZ="${TALLER_RAM_VOZ_MB:-5120}"
CLAVE="${TALLER_CLAVE:-$HOME/.ssh/oracle_a1}"
[[ "$VOZ" == "200" || "$CT_NOTICIAS" == "200" ]] && { echo "el VMID 200 es intocable (AuraCRM)" >&2; exit 2; }

pve() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$@"; }
vm() { ssh -i "$CLAVE" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 "root@$IP" "$@"; }

reiniciar_con_ram() {
  local mb=$1
  echo "apagando VM $VOZ..."
  pve "qm shutdown $VOZ --timeout 180 || qm stop $VOZ"
  pve "qm set $VOZ --memory $mb --balloon 0"
  pve "qm start $VOZ"
  echo "esperando SSH en $IP..."
  for _ in $(seq 1 60); do vm true 2>/dev/null && return 0; sleep 5; done
  echo "la VM no responde por SSH" >&2; return 1
}

case "${1:-}" in
  on)
    pve "pct status $CT_NOTICIAS | grep -q running && pct stop $CT_NOTICIAS || true"
    reiniciar_con_ram "$RAM_TALLER"
    vm 'systemctl stop voz-stream voz-api homelab-whisper 2>/dev/null; systemctl is-active voz-stream voz-api homelab-whisper; free -g | sed -n 2p'
    echo "modo taller: VM $VOZ con $RAM_TALLER MB y la voz parada"
    ;;
  off)
    reiniciar_con_ram "$RAM_VOZ"
    for _ in $(seq 1 60); do vm 'curl -sf localhost:8082/health >/dev/null' 2>/dev/null && break; sleep 5; done
    vm 'systemctl is-active voz-stream voz-api; free -g | sed -n 2p'
    pve "pct start $CT_NOTICIAS"
    echo "modo voz: VM $VOZ con $RAM_VOZ MB y servicios arriba; app-noticias arrancado"
    ;;
  estado)
    pve "qm config $VOZ | grep -E '^memory'; pct status $CT_NOTICIAS; free -m | sed -n 2p"
    vm 'systemctl is-active voz-stream voz-api homelab-whisper; free -g | sed -n 2p; pgrep -fa "fase[0-9]_|entrenar" | head -3' || true
    ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
