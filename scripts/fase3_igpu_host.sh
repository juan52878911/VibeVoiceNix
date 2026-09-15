#!/usr/bin/env bash
# Fase 3 (docs/plan-rendimiento.md): pasa la iGPU UHD 630 del host pve a la VM voz (210), o la devuelve.
#
#   bash scripts/fase3_igpu_host.sh pasar    < /dev/null     # desde el Mac
#   bash scripts/fase3_igpu_host.sh devolver < /dev/null
#
# En caliente: NO reinicia pve y NO toca AuraCRM (VM 200, CT 203). Paso a paso, comprobando entre cada uno
# que pve sigue respondiendo por ssh; si deja de responder, el guion se para (no insiste). El LXC 204 del
# laboratorio pierde su /dev/dri mientras la iGPU esté en la VM (se le quita el dev0 y se le devuelve).
set -uo pipefail

PVE=(ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 pve)
GPU=0000:00:02.0

vivo() {
  local _
  for _ in 1 2 3; do
    "${PVE[@]}" true 2>/dev/null && return 0
    sleep 5
  done
  echo "[igpu] pve NO responde por ssh tras $1: me paro aqui. Revisar consola del host." >&2
  exit 2
}

paso() {  # $1 descripcion  $2 orden remota
  echo "[igpu] $1"
  "${PVE[@]}" "$2" || { echo "[igpu] fallo en: $1" >&2; vivo "$1"; exit 1; }
  vivo "$1"
}

case "${1:-}" in
  pasar)
    "${PVE[@]}" "qm config 210 | grep -q '^hostpci0:'" && { echo "[igpu] la VM 210 ya tiene hostpci0"; exit 1; }
    "${PVE[@]}" "readlink /sys/bus/pci/devices/$GPU/driver | grep -q i915" || { echo "[igpu] la iGPU no esta en i915: no toco nada"; exit 1; }
    paso "para el LXC 204 y le quita el dev0" "pct stop 204 2>/dev/null; pct set 204 --delete dev0; pct config 204 | grep -c '^dev0' || true"
    paso "nadie mas usa /dev/dri" "! fuser -s /dev/dri/* 2>/dev/null"
    paso "suelta la consola del framebuffer" "for v in /sys/class/vtconsole/vtcon*; do grep -qi 'frame buffer' \$v/name && echo 0 > \$v/bind; done; cat /sys/class/vtconsole/vtcon*/bind | tr '\n' ' '"
    paso "desliga la iGPU de i915" "timeout 60 sh -c 'echo $GPU > /sys/bus/pci/drivers/i915/unbind'"
    paso "la liga a vfio-pci" "modprobe vfio-pci && echo vfio-pci > /sys/bus/pci/devices/$GPU/driver_override && echo $GPU > /sys/bus/pci/drivers_probe && lspci -nnk -s 00:02.0 | grep 'driver in use'"
    paso "hostpci0 en la VM 210" "qm set 210 --hostpci0 $GPU,pcie=0 && qm config 210 | grep '^hostpci0'"
    paso "apaga la VM 210" "qm shutdown 210 --timeout 180 || qm stop 210"
    paso "arranca la VM 210 con la iGPU" "qm start 210 && sleep 20 && qm status 210"
    echo "[igpu] hecho: iGPU en la VM 210; AuraCRM sin tocar"
    "${PVE[@]}" "qm status 200; pct status 203; uptime"
    ;;
  devolver)
    paso "apaga la VM 210" "qm shutdown 210 --timeout 180 || qm stop 210"
    paso "quita hostpci0" "qm set 210 --delete hostpci0; qm config 210 | grep -c '^hostpci0' || true"
    paso "arranca la VM 210 sin iGPU" "qm start 210 && qm status 210"
    paso "desliga la iGPU de vfio-pci" "echo > /sys/bus/pci/devices/$GPU/driver_override; if [ -e /sys/bus/pci/drivers/vfio-pci/$GPU ]; then echo $GPU > /sys/bus/pci/drivers/vfio-pci/unbind; fi; true"
    paso "la vuelve a ligar a i915" "timeout 60 sh -c 'echo $GPU > /sys/bus/pci/drivers/i915/bind'; lspci -nnk -s 00:02.0 | grep 'driver in use'"
    paso "vuelve la consola del framebuffer" "for v in /sys/class/vtconsole/vtcon*; do grep -qi 'frame buffer' \$v/name && echo 1 > \$v/bind; done; true"
    paso "devuelve el dev0 al LXC 204" "pct set 204 --dev0 /dev/dri/renderD128,mode=0666 && ls -la /dev/dri/renderD128"
    echo "[igpu] hecho: iGPU de vuelta en el host"
    "${PVE[@]}" "qm status 200; pct status 203; uptime"
    ;;
  *)
    echo "uso: $0 pasar|devolver" >&2
    exit 1
    ;;
esac
