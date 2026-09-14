#!/usr/bin/env bash
# Segunda mitad del umbral 0.3: cuánto frena la iGPU a la CPU con la MISMA carga de CPU.
#
#   bash scripts/fase0_igpu_frecuencia.sh < /dev/null      # desde el Mac
#
# Durante 180 s: la VM voz corre el decodificador int8 en CPU a 6 hilos sin parar (carga fija), pve
# registra temperatura, frecuencia y RAPL cada segundo, y el LXC 204 pone la GPU al 100 % solo entre
# los segundos 60 y 120 (decodificador int8 en f32). Compara las ventanas [15, 55] (sin GPU) y
# [75, 115] (con GPU): MHz medios del host y ms por llamada de la carga de CPU. Umbral fijado antes:
# la frecuencia media baja <= 15 % con la GPU al 100 %.
set -euo pipefail

SSH_VM=(ssh -n -i "$HOME/.ssh/oracle_a1" -o IdentitiesOnly=yes root@192.168.2.54)
SSH_PVE=(ssh -n -o ConnectTimeout=10 pve)
PY_VM=/nix/store/68d5yzqf9iqgvfwn2lv4hky2khin1i0f-vibevoice-env/bin/python3.12
DIR=${TMPDIR:-/tmp}/igpu_frecuencia_$$
mkdir -p "$DIR"

# Nada de otra sesión en la VM, y voz-stream libre
if "${SSH_VM[@]}" 'pgrep -fa "fase[0-9]_|ws_fidelidad|benchmark_app" | grep -v pgrep'; then
  echo "hay otra carga en la VM: aborto"; exit 1
fi
"${SSH_VM[@]}" 'curl -s -m5 127.0.0.1:8082/health | grep -q "\"ocupado\":false"' || { echo "voz-stream ocupado: aborto"; exit 1; }

t0=$(( $(date +%s) + 5 ))
"${SSH_PVE[@]}" "sleep \$(( $t0 - \$(date +%s) )); bash /root/vigilar_host.sh 180" > "$DIR/host.csv" &
"${SSH_VM[@]}" "sleep \$(( $t0 - \$(date +%s) )); $PY_VM /root/lab/cli/lab_igpu.py bucle --ir /var/lib/voz/ov/decoder_mm_int8.xml --dispositivo CPU --hilos 6 --segundos 180" > "$DIR/cpu.csv" &
"${SSH_PVE[@]}" "sleep \$(( $t0 + 60 - \$(date +%s) )); pct exec 204 -- /opt/ov/bin/python /root/lab_igpu.py bucle --ir /root/ir/decoder_mm_int8.xml --dispositivo GPU --precision f32 --segundos 60" > "$DIR/gpu.csv" &
wait

python3 - "$DIR" "$t0" <<'PY'
import csv, statistics, sys
d, t0 = sys.argv[1], int(sys.argv[2])
host = [r for r in csv.DictReader(open(f"{d}/host.csv")) if r.get("epoch", "").isdigit()]
cpu = [l.strip().split(",") for l in open(f"{d}/cpu.csv") if l[:1].isdigit()]
gpu = [l.strip().split(",") for l in open(f"{d}/gpu.csv") if l[:1].isdigit()]
def ventana(a, b):
    h = [r for r in host if t0 + a <= int(r["epoch"]) < t0 + b]
    c = [float(x[2]) for x in cpu if t0 + a <= int(x[0]) < t0 + b]
    return {"n_host": len(h), "mhz": statistics.mean(float(r["mhz_media"]) for r in h),
            "temp": statistics.mean(float(r["temp_c"]) for r in h),
            "w_paquete": statistics.mean(float(r["w_paquete"]) for r in h),
            "w_uncore": statistics.mean(float(r["w_uncore"]) for r in h),
            "cpu_ms": statistics.median(c) if c else float("nan")}
sin, con = ventana(15, 55), ventana(75, 115)
caida = 100 * (1 - con["mhz"] / sin["mhz"])
print("sin GPU ", {k: round(v, 2) for k, v in sin.items()})
print("con GPU ", {k: round(v, 2) for k, v in con.items()})
print("GPU ms mediana en su ventana", statistics.median(float(x[2]) for x in gpu) if gpu else "sin datos")
print(f"caida de frecuencia {caida:.1f} %  ->  {'CUMPLE' if caida <= 15 else 'NO CUMPLE'} (umbral <= 15 %)")
print(f"carga de CPU: {sin['cpu_ms']:.1f} -> {con['cpu_ms']:.1f} ms por llamada "
      f"({100 * (con['cpu_ms'] / sin['cpu_ms'] - 1):+.1f} %)")
PY
echo "datos en $DIR"
