#!/usr/bin/env bash
# Temperatura, frecuencia y potencia del host (pve) una vez por segundo, en CSV.
#
#   scp scripts/vigilar_host.sh pve:/root/ && ssh pve 'bash /root/vigilar_host.sh 600 > /root/host.csv'
#
# Columnas: epoch, temperatura del paquete (C), MHz medios y minimos de las 12 CPU,
# vatios del paquete, de los nucleos, del uncore (la iGPU) y de la DRAM (RAPL), y el
# contador de estrangulamiento termico del paquete. Solo lee sysfs y /proc: no toca nada.
set -euo pipefail

SEGUNDOS=${1:-300}
RAPL=/sys/class/powercap

temp_paquete() {
  local h
  for h in /sys/class/hwmon/hwmon*; do
    if [ "$(cat "$h/name" 2>/dev/null)" = coretemp ]; then
      echo $(( $(cat "$h/temp1_input") / 1000 ))
      return
    fi
  done
  echo nan
}

energia() {  # uJ de una zona RAPL, o 0 si no existe
  cat "$RAPL/$1/energy_uj" 2>/dev/null || echo 0
}

zonas=(intel-rapl:0 intel-rapl:0:0 intel-rapl:0:1 intel-rapl:0:2)
declare -A prev
for z in "${zonas[@]}"; do prev[$z]=$(energia "$z"); done
t_prev=$(date +%s.%N)

echo "epoch,temp_c,mhz_media,mhz_min,w_paquete,w_nucleos,w_uncore,w_dram,estrangulado"
for _ in $(seq 1 "$SEGUNDOS"); do
  sleep 1
  t=$(date +%s.%N)
  dt=$(awk -v a="$t" -v b="$t_prev" 'BEGIN{print a-b}')
  vatios=()
  for z in "${zonas[@]}"; do
    e=$(energia "$z")
    vatios+=("$(awk -v e="$e" -v p="${prev[$z]}" -v dt="$dt" 'BEGIN{d=e-p; if(d<0)d=0; printf "%.2f", d/1e6/dt}')")
    prev[$z]=$e
  done
  t_prev=$t
  mhz=$(awk -F: '/^cpu MHz/{s+=$2; n++; if(min==""||$2<min)min=$2} END{printf "%.0f,%.0f", s/n, min}' /proc/cpuinfo)
  estr=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo nan)
  echo "$(date +%s),$(temp_paquete),$mhz,${vatios[0]},${vatios[1]},${vatios[2]},${vatios[3]},$estr"
done
