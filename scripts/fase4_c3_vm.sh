#!/usr/bin/env bash
# Fase 4, C3 (docs/plan-rendimiento.md): corpus del banco A/B con el LM TTS en int8 y el control (decodificador
# int4), frente a producción (LM int4). Se puntúa aparte (banco_ab.py, en el CT de lotes de ascci).
#
#   ssh -n root@voz 'systemd-run --unit fase4-c3 --collect bash /root/lab/cli/fase4_c3_vm.sh'
#
# Pasos, cada uno con un proceso nuevo de voz-stream en 127.0.0.1:8092, con el entorno, el python y el código
# de producción EN MARCHA y solo el IR que toque cambiado:
#   1. base17: producción, 17 clips (andres, semilla 101). Si su PCM coincide con el manifiesto del corpus
#      `difusion` del 13-09 (/root/lab/manifiesto_difusion.json), ese corpus sirve de base; si no, se genera
#      la base entera aquí mismo.
#   2. lm_int8: VIBEVOICE_IR_LM = tts_lm_estado_int8.xml, corpus entero (238 clips).
#   3. control_int4: VIBEVOICE_IR_ACUSTICO = decoder_mm_int4.xml, corpus entero.
# Para voz-stream y lo devuelve al salir, falle lo que falle.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep curl awk swapoff; do
  command -v "$orden" >/dev/null || { echo "[c3] falta $orden: aborto"; exit 1; }
done
LAB=/root/lab
PUERTO=8092
SAL=$LAB/fase4
OV=/var/lib/voz/ov
mkdir -p "$SAL"
cd "$LAB" || exit 1

if pgrep -fa 'fase[0-9]_|ws_fidelidad|evaluar_clones|benchmark_app' | grep -v -e pgrep -e fase4_c3; then
  echo "[c3] hay otra carga en la VM: no se toca nada"; exit 1
fi
for f in "$OV/tts_lm_estado_int8.xml" "$OV/decoder_mm_int4.xml" "$LAB/manifiesto_difusion.json" "$LAB/cli/banco_ab_generar.py"; do
  [ -e "$f" ] || { echo "[c3] falta $f: aborto"; exit 1; }
done

PID_PROD=$(systemctl show -p MainPID --value voz-stream)
[ "${PID_PROD:-0}" -gt 0 ] || { echo "[c3] voz-stream no esta en marcha: aborto"; exit 1; }
mapfile -d '' -t CMD < "/proc/$PID_PROD/cmdline"
PY=${CMD[0]}
CLI=${CMD[1]}
while IFS= read -r -d '' par; do
  case "$par" in
    VIBEVOICE_*|OMP_*|MALLOC_*|HF_*|VOZ_*) export "${par?}" ;;
  esac
done < "/proc/$PID_PROD/environ"
[ -n "${VOZ_TOKEN:-}" ] || { echo "[c3] sin VOZ_TOKEN: aborto"; exit 1; }
IR_LM_PROD=$VIBEVOICE_IR_LM
IR_AC_PROD=$VIBEVOICE_IR_ACUSTICO
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
  echo "[c3] voz-stream devuelto: $(systemctl is-active voz-stream)"
}
trap devolver EXIT
systemctl stop voz-stream-sin-swap voz-stream
sleep 3

servidor() {  # $1 etiqueta  $2 IR del LM  $3 IR del decodificador
  VIBEVOICE_IR_LM=$2 VIBEVOICE_IR_ACUSTICO=$3 "$PY" "$CLI" > "$SAL/servidor_$1.log" 2>&1 &
  SERVIDOR=$!
  for _ in $(seq 1 240); do
    curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1 && break
    kill -0 "$SERVIDOR" 2>/dev/null || break
    sleep 3
  done
  if ! curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" > "$SAL/health_$1.json" 2>/dev/null; then
    echo "[c3] servidor $1 no arranco"; tail -20 "$SAL/servidor_$1.log"; exit 1
  fi
  echo "[c3] servidor $1: $(grep -oE '"ir":\{[^}]*\}' "$SAL/health_$1.json")"
  usado=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print (t-f)}' /proc/meminfo)
  libre=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$usado" -gt 0 ] && [ "$usado" -lt $((libre - 524288)) ]; then swapoff -a && swapon -a; fi
}
parar() {
  kill "$SERVIDOR"
  wait "$SERVIDOR" 2>/dev/null
  SERVIDOR=""
  sleep 3
}
generar() {  # $1 etiqueta, el resto se pasa a banco_ab_generar.py
  local e=$1
  shift
  rm -rf "${SAL:?}/$e"
  curl -s "http://127.0.0.1:$PUERTO/crono?reset=true" >/dev/null
  "$PY" cli/banco_ab_generar.py --url "http://127.0.0.1:$PUERTO" --token "$VOZ_TOKEN" --etiqueta "$e" \
      --salida "$SAL/$e" "$@" || { echo "[c3] fallo al generar $e"; exit 1; }
  curl -s "http://127.0.0.1:$PUERTO/crono" > "$SAL/$e/crono.json"
}

echo "[c3] === base de hoy $(date +%H:%M:%S)"
servidor base "$IR_LM_PROD" "$IR_AC_PROD"
generar base17 --voces andres --semillas 101
if "$PY" - "$SAL/base17" "$LAB/manifiesto_difusion.json" <<'PYEOF'
import hashlib, json, os, sys, wave
d, m = sys.argv[1], json.load(open(sys.argv[2]))
distintos = []
for f, h in m.items():
    with wave.open(os.path.join(d, f)) as w:
        if hashlib.md5(w.readframes(w.getnframes())).hexdigest() != h:
            distintos.append(f)
print(f"[c3] base de hoy frente al corpus difusion del 13-09: {len(m) - len(distintos)}/{len(m)} identicos {distintos[:3]}")
sys.exit(1 if distintos else 0)
PYEOF
then
  echo "[c3] la base del 13-09 (difusion) vale: no se regenera"
else
  echo "[c3] la base cambio desde el 13-09: se genera la base entera"
  generar base
fi
parar

echo "[c3] === LM int8 $(date +%H:%M:%S)"
servidor lm_int8 "$OV/tts_lm_estado_int8.xml" "$IR_AC_PROD"
generar lm_int8
parar

echo "[c3] === control: decodificador int4 $(date +%H:%M:%S)"
servidor control_int4 "$IR_LM_PROD" "$OV/decoder_mm_int4.xml"
generar control_int4
parar
echo "[c3] FIN $(date +%H:%M:%S)"
