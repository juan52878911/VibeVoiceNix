#!/usr/bin/env bash
# Conmuta el host Proxmox entre servir voz y entrenar/doblar en segundo plano.
#
#   scripts/modo_taller.sh on       apaga app-noticias y voz, arranca taller
#   scripts/modo_taller.sh off      apaga taller, vuelve voz y app-noticias
#   scripts/modo_taller.sh estado   que corre y cuanta RAM queda
#
# QUE NO TOCA NUNCA: la VM 200 (AuraCRM). Esta en la lista de intocables y el
# script se niega a operar sobre ella aunque se le pida por variable.
#
# POR QUE HACE FALTA: el M920q tiene 16 GB. AuraCRM (~3-4 GB) + host (~1,5)
# + voz (5) + app-noticias (3) no dejan sitio para un taller de 8-11 GB. De
# noche voz y app-noticias no hacen falta; el taller si.
set -euo pipefail

HOST="${TALLER_PVE:-pve}"          # alias de ~/.ssh/config (root@192.168.2.100)
VOZ="${TALLER_VOZ_VMID:-210}"
TALLER="${TALLER_VMID:-220}"
CT_NOTICIAS="${TALLER_CT_NOTICIAS:-100}"
INTOCABLES=(200)

for id in "$VOZ" "$TALLER" "$CT_NOTICIAS"; do
  for i in "${INTOCABLES[@]}"; do
    [[ "$id" == "$i" ]] && { echo "el VMID $id es intocable (AuraCRM)" >&2; exit 2; }
  done
done

pve() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$@"; }

estado_vm() { pve "qm status $1 2>/dev/null | awk '{print \$2}'"; }
estado_ct() { pve "pct status $1 2>/dev/null | awk '{print \$2}'"; }

apagar_vm() {
  local id=$1
  [[ "$(estado_vm "$id")" == "running" ]] || return 0
  echo "apagando VM $id..."
  # shutdown limpio con tope; si el agente no responde, stop.
  pve "qm shutdown $id --timeout 120" || pve "qm stop $id"
}

case "${1:-}" in
  on)
    if [[ "$(estado_ct "$CT_NOTICIAS")" == "running" ]]; then
      echo "parando CT $CT_NOTICIAS..."; pve "pct stop $CT_NOTICIAS"
    fi
    apagar_vm "$VOZ"
    echo "arrancando taller ($TALLER)..."
    pve "qm start $TALLER"
    ;;
  off)
    apagar_vm "$TALLER"
    echo "arrancando voz ($VOZ) y CT $CT_NOTICIAS..."
    pve "qm start $VOZ"
    pve "pct start $CT_NOTICIAS"
    ;;
  estado)
    pve "qm list; pct list; free -m | head -2"
    ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
