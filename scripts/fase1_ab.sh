#!/usr/bin/env bash
# Puerta de la fase 1 (docs/plan-rendimiento.md): base -> variante -> base -> variante.
#
#   ssh -n root@voz 'systemd-run --unit fase1-ab --collect bash /root/lab/cli/fase1_ab.sh'
#
# Cada tanda es un proceso nuevo de voz-stream en 127.0.0.1:8092 (fuera de la LAN), con el MISMO
# entorno, el mismo python y el mismo voz-stream.py que la unidad de produccion EN MARCHA. La base usa
# su codigo OpenVINO; la variante, una copia de ese codigo con solo motor.py cambiado por el de
# /root/lab/ov. Si el motor.py de produccion no es el de partida de la rama, aborta: la diferencia
# dejaria de ser solo A1+A2+A4. banco_md5.py: 3 rondas por tanda (la 0 no cuenta), md5, RTF y VmHWM.
# ws_fidelidad.py completo contra la primera variante, despues de su banco.
# Para voz-stream y lo devuelve al salir, falle lo que falle.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep curl awk sha256sum swapoff; do
  command -v "$orden" >/dev/null || { echo "[ab] falta $orden: aborto"; exit 1; }
done

# sha256 de pkgs/vibevoice-ov/motor.py en el merge-base de la rama con main (d32c686)
MOTOR_ORIGEN=d38b1f612c246e1f6c360483afffd147502b3dab6f1bebdec1ea8924609c6aa9
LAB=/root/lab
PUERTO=8092
SALIDA=$LAB/fase1
mkdir -p "$SALIDA"
cd "$LAB" || exit 1

if pgrep -fa 'fase[0-9]_|benchmark_app' | grep -v -e pgrep -e fase1_ab; then
  echo "[ab] hay un experimento de otra sesion corriendo: no se toca nada"
  exit 1
fi

# Lo que corre produccion, leido de la unidad en marcha y no de rutas fijas (puede redesplegarse)
PID_PROD=$(systemctl show -p MainPID --value voz-stream)
[ "${PID_PROD:-0}" -gt 0 ] || { echo "[ab] voz-stream no esta en marcha: aborto"; exit 1; }
mapfile -d '' -t CMD < "/proc/$PID_PROD/cmdline"
PY=${CMD[0]}
BASE_CLI=${CMD[1]}
# Solo las variables del servicio. NO el PATH de la unidad: en NixOS no trae curl ni awk, y
# importarlo dejo el bucle de espera ciego dos veces (14-09-2026).
while IFS= read -r -d '' par; do
  case "$par" in
    VIBEVOICE_*|OMP_*|MALLOC_*|HF_*|VOZ_*) export "${par?}" ;;
  esac
done < "/proc/$PID_PROD/environ"
BASE_OV=$VIBEVOICE_OV_CODIGO
[ -n "${VOZ_TOKEN:-}" ] || { echo "[ab] sin VOZ_TOKEN en el entorno de produccion: aborto"; exit 1; }
[ "$(sha256sum "$BASE_OV/motor.py" | cut -d' ' -f1)" = "$MOTOR_ORIGEN" ] || {
  echo "[ab] el motor.py de produccion no es el de partida de la rama: aborto"; exit 1; }
VAR_OV=$LAB/ov-variante
rm -rf "$VAR_OV"
mkdir -p "$VAR_OV"
cp "$BASE_OV"/*.py "$VAR_OV/"
cp "$LAB/ov/motor.py" "$VAR_OV/motor.py"
echo "[ab] base: $BASE_CLI + $BASE_OV; variante: el mismo voz-stream.py + $VAR_OV (solo motor.py)"

# /run/voz-stream/voces desaparece al parar la unidad: se enlaza a lo que apunta ahora
VOCES=$LAB/voces
rm -rf "$VOCES"
mkdir -p "$VOCES"
n=0
for f in /run/voz-stream/voces/*.pt; do
  [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$VOCES/" && n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "[ab] sin voces en /run/voz-stream/voces: aborto"; exit 1; }
echo "[ab] $n voces enlazadas"
export VIBEVOICE_VOCES=$VOCES VOZ_STREAM_HOST=127.0.0.1 VOZ_STREAM_PUERTO=$PUERTO

# AB_SECO=1: comprueba todo lo de arriba y sale sin parar nada
if [ "${AB_SECO:-0}" = 1 ]; then
  echo "[ab] en seco: PY=$PY OMP=${OMP_NUM_THREADS:-} semilla=${VIBEVOICE_SEMILLA:-} IR_LM=${VIBEVOICE_IR_LM:-}"
  echo "[ab] en seco: curl $(command -v curl), token ${#VOZ_TOKEN} caracteres; listo para medir"
  exit 0
fi

SERVIDOR=""
devolver() {
  [ -n "$SERVIDOR" ] && kill "$SERVIDOR" 2>/dev/null
  sleep 2
  systemctl start voz-stream voz-stream-sin-swap
  echo "[ab] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT

systemctl stop voz-stream-sin-swap voz-stream
sleep 3

tanda() {  # $1 etiqueta  $2 codigo OV
  local etiqueta=$1
  VIBEVOICE_OV_CODIGO=$2 "$PY" "$BASE_CLI" > "$SALIDA/$etiqueta.log" 2>&1 &
  SERVIDOR=$!
  # Hasta 12 min, como voz-stream-sin-swap
  for _ in $(seq 1 240); do
    curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1 && break
    kill -0 "$SERVIDOR" 2>/dev/null || break
    sleep 3
  done
  if ! curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1; then
    echo "[ab] $etiqueta no arranco"; tail -20 "$SALIDA/$etiqueta.log"; exit 1
  fi
  tr '\r' '\n' < "$SALIDA/$etiqueta.log" | grep -E '\[carga\]|modelo listo'
  # Lo mismo que voz-stream-sin-swap en produccion, en las dos variantes: la carga vieja empuja paginas
  # del modelo al swap y la nueva no, y medir la base con fallos de pagina mayores no seria su RTF real.
  usado=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print (t-f)}' /proc/meminfo)
  libre=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  echo "[ab] $etiqueta: swap usado $((usado / 1024)) MB, disponible $((libre / 1024)) MB"
  if [ "$usado" -gt 0 ] && [ "$usado" -lt $((libre - 524288)) ]; then
    swapoff -a && swapon -a && echo "[ab] $etiqueta: swap devuelto a RAM"
  fi
  "$PY" cli/banco_md5.py --url "http://127.0.0.1:$PUERTO" --etiqueta "$etiqueta" --rondas 3 \
      --pid "$SERVIDOR" --salida "$SALIDA/$etiqueta.json"
  if [ "$etiqueta" = variante-1 ]; then
    echo "[ab] ws_fidelidad contra la variante"
    "$PY" cli/ws_fidelidad.py --url "http://127.0.0.1:$PUERTO" --token "$VOZ_TOKEN" \
        > "$SALIDA/ws_fidelidad.log" 2>&1
    echo "[ab] ws_fidelidad codigo de salida $?"
    tail -25 "$SALIDA/ws_fidelidad.log"
  fi
  kill "$SERVIDOR"
  wait "$SERVIDOR" 2>/dev/null
  SERVIDOR=""
  sleep 3
}

tanda base-1 "$BASE_OV"
tanda variante-1 "$VAR_OV"
tanda base-2 "$BASE_OV"
tanda variante-2 "$VAR_OV"

"$PY" cli/banco_md5.py comparar "$SALIDA"/base-1.json "$SALIDA"/variante-1.json \
    "$SALIDA"/base-2.json "$SALIDA"/variante-2.json
echo "[ab] FIN"
