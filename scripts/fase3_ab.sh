#!/usr/bin/env bash
# Fase 3 (docs/plan-rendimiento.md): el decodificador en la iGPU (asíncrono) frente a producción, en la VM voz
# con la iGPU ya pasada (scripts/fase3_igpu_host.sh pasar).
#
#   ssh -n root@voz 'systemd-run --unit fase3-ab --collect bash /root/lab/cli/fase3_ab.sh'
#
# Tandas base -> gpu -> base -> gpu, cada una un proceso nuevo de voz-stream en 127.0.0.1:8092 con el entorno,
# el python y el voz-stream.py de producción en marcha. La variante usa el código OpenVINO de producción con
# SOLO motor.py cambiado (dispositivo del decodificador por entorno) y
#   VIBEVOICE_ACUSTICO_DISPOSITIVO=GPU  VIBEVOICE_SOLAPAR_DECODER=1
# banco_md5.py a 3 rondas guardando los WAV. Como el decodificador en GPU cambia la numérica (f32 en otro
# dispositivo), no se exige md5: se comparan duración exacta y SNR de cada clip contra la base (puertas 1 y
# 4). ws_fidelidad.py completo contra la primera tanda gpu. Para voz-stream y lo devuelve al salir.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep curl awk sha256sum swapoff; do
  command -v "$orden" >/dev/null || { echo "[gpu] falta $orden: aborto"; exit 1; }
done

LAB=/root/lab
PUERTO=8092
SALIDA=$LAB/fase3
rm -rf "$SALIDA"
mkdir -p "$SALIDA"
cd "$LAB" || exit 1

if pgrep -fa 'fase[0-9]_|ws_fidelidad|evaluar_clones|benchmark_app|ablacion' | grep -v -e pgrep -e fase3_ab; then
  echo "[gpu] hay otra carga en la VM: no se toca nada"; exit 1
fi
[ -e /dev/dri/renderD128 ] || { echo "[gpu] no hay /dev/dri/renderD128 en la VM: la iGPU no esta pasada"; exit 1; }

PID_PROD=$(systemctl show -p MainPID --value voz-stream)
[ "${PID_PROD:-0}" -gt 0 ] || { echo "[gpu] voz-stream no esta en marcha: aborto"; exit 1; }
mapfile -d '' -t CMD < "/proc/$PID_PROD/cmdline"
PY=${CMD[0]}
CLI=${CMD[1]}
while IFS= read -r -d '' par; do
  case "$par" in
    VIBEVOICE_*|OMP_*|MALLOC_*|HF_*|VOZ_*) export "${par?}" ;;
  esac
done < "/proc/$PID_PROD/environ"
[ -n "${VOZ_TOKEN:-}" ] || { echo "[gpu] sin VOZ_TOKEN: aborto"; exit 1; }
BASE_OV=$VIBEVOICE_OV_CODIGO
VAR_OV=$LAB/ov-gpu
rm -rf "$VAR_OV"
mkdir -p "$VAR_OV"
cp "$BASE_OV"/*.py "$VAR_OV/"
cp "$LAB/ov/motor.py" "$VAR_OV/motor.py"

echo "[gpu] OpenVINO ve: $(cd "$VAR_OV" && "$PY" -c 'import openvino as ov; c = ov.Core(); print([(d, c.get_property(d, "FULL_DEVICE_NAME")) for d in c.available_devices])' 2>&1 | tail -1)"
if ! "$PY" -c 'import openvino as ov, sys; sys.exit(0 if "GPU" in ov.Core().available_devices else 1)'; then
  echo "[gpu] OpenVINO no ve la GPU: aborto"; exit 1
fi

VOCES=$LAB/voces
rm -rf "$VOCES"
mkdir -p "$VOCES"
for f in /run/voz-stream/voces/*.pt; do
  [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$VOCES/"
done
export VIBEVOICE_VOCES=$VOCES VOZ_STREAM_HOST=127.0.0.1 VOZ_STREAM_PUERTO=$PUERTO

SERVIDOR=""
devolver() {
  [ -n "$SERVIDOR" ] && kill "$SERVIDOR" 2>/dev/null
  sleep 2
  systemctl start voz-stream voz-stream-sin-swap
  echo "[gpu] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT
systemctl stop voz-stream-sin-swap voz-stream
sleep 3

tanda() {  # $1 etiqueta  $2 base|gpu
  local etiqueta=$1
  if [ "$2" = gpu ]; then
    VIBEVOICE_OV_CODIGO=$VAR_OV VIBEVOICE_ACUSTICO_DISPOSITIVO=GPU VIBEVOICE_SOLAPAR_DECODER=1 \
      "$PY" "$CLI" > "$SALIDA/$etiqueta.log" 2>&1 &
  else
    VIBEVOICE_OV_CODIGO=$BASE_OV "$PY" "$CLI" > "$SALIDA/$etiqueta.log" 2>&1 &
  fi
  SERVIDOR=$!
  for _ in $(seq 1 240); do
    curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1 && break
    kill -0 "$SERVIDOR" 2>/dev/null || break
    sleep 3
  done
  if ! curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1; then
    echo "[gpu] $etiqueta no arranco"; tail -20 "$SALIDA/$etiqueta.log"; exit 1
  fi
  tr '\r' '\n' < "$SALIDA/$etiqueta.log" | grep -E '\[carga\]|modelo listo|solapado'
  usado=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print (t-f)}' /proc/meminfo)
  libre=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$usado" -gt 0 ] && [ "$usado" -lt $((libre - 524288)) ]; then swapoff -a && swapon -a; fi
  "$PY" cli/banco_md5.py --url "http://127.0.0.1:$PUERTO" --etiqueta "$etiqueta" --rondas 3 \
      --pid "$SERVIDOR" --salida "$SALIDA/$etiqueta.json" --wav "$SALIDA/wav"
  if [ "$etiqueta" = gpu-1 ]; then
    "$PY" cli/ws_fidelidad.py --url "http://127.0.0.1:$PUERTO" --token "$VOZ_TOKEN" > "$SALIDA/ws_gpu.log" 2>&1
    echo "[gpu] ws_fidelidad codigo $?: $(tail -1 "$SALIDA/ws_gpu.log")"
  fi
  kill "$SERVIDOR"
  wait "$SERVIDOR" 2>/dev/null
  SERVIDOR=""
  sleep 3
}

tanda base-1 base
tanda gpu-1 gpu
tanda base-2 base
tanda gpu-2 gpu

"$PY" - "$SALIDA" <<'PYEOF'
import json, statistics, sys
import numpy as np
d = sys.argv[1]
def leer(e, r, f):
    # banco_md5.py guarda el cuerpo del flujo tal cual: cabecera WAV de streaming de 44 bytes (con tamaños de
    # relleno, que wave.open no acepta) y PCM16 detrás
    with open(f"{d}/wav/{e}_r{r}_f{f}.wav", "rb") as fh:
        return np.frombuffer(fh.read()[44:], dtype=np.int16).astype(np.float64)
duraciones_ok, snrs = True, []
for r in range(3):
    for f in range(8):
        b, g = leer("base-1", r, f), leer("gpu-1", r, f)
        if len(b) != len(g):
            duraciones_ok = False
            print(f"DURACION DISTINTA r{r} f{f}: {len(b)} frente a {len(g)} muestras")
            continue
        err = ((b - g) ** 2).sum()
        snrs.append(99.0 if err == 0 else 10 * np.log10((b ** 2).sum() / err))
print(f"duraciones identicas en los 24 clips: {duraciones_ok}; SNR gpu frente a base: min {min(snrs):.1f} dB, mediana {statistics.median(snrs):.1f} dB")
def rtf(pref):
    return [r["rtf"] for e in ("1", "2") for r in json.load(open(f"{d}/{pref}-{e}.json"))["rondas"][1:]]
rb, rg = rtf("base"), rtf("gpu")
mb, mg = statistics.median(rb), statistics.median(rg)
for e in ("base-1", "gpu-1", "base-2", "gpu-2"):
    j = json.load(open(f"{d}/{e}.json"))
    print(e, "memoria", j.get("memoria"), "reparto ronda 2", j["rondas"][-1].get("reparto"))
print(f"RTF base {rb} mediana {mb:.4f} · gpu {rg} mediana {mg:.4f} · gpu/base {mg / mb:.4f} (puerta <= 0,93)")
print("PUERTAS 1 Y 4:", "PASAN" if duraciones_ok and min(snrs) >= 25 and mg <= 0.93 * mb else "NO PASAN")
PYEOF
echo "[gpu] FIN"
