#!/usr/bin/env bash
# Puerta de A5 (docs/plan-rendimiento.md): whisper-server a 4 hilos transcribe EXACTAMENTE lo mismo que a 6.
#
#   ssh -n root@voz 'systemd-run --unit fase1-whisper --collect bash /root/lab/cli/fase1_whisper.sh'
#
# Genera los 8 audios de banco_md5.py con voz-stream de producción (semilla 101), levanta una segunda
# instancia de whisper-server con el MISMO binario, modelo e idioma que la unidad pero --threads 4 en
# 127.0.0.1:8091, y transcribe cada audio con producción (6 hilos, :8081) y con la de 4, tres veces cada
# uno. Pasa si los 8 textos son idénticos. Anota el tiempo de cada transcripción. No para nada.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl curl sha256sum; do
  command -v "$orden" >/dev/null || { echo "[stt] falta $orden: aborto"; exit 1; }
done
LAB=/root/lab
SALIDA=$LAB/fase1_whisper
rm -rf "$SALIDA"
mkdir -p "$SALIDA"
cd "$LAB" || exit 1

if pgrep -fa 'fase[0-9]_|ws_fidelidad|benchmark_app' | grep -v -e pgrep -e fase1_whisper; then
  echo "[stt] hay otra carga en la VM: no se toca nada"; exit 1
fi

PID_TTS=$(systemctl show -p MainPID --value voz-stream)
mapfile -d '' -t CMD_TTS < "/proc/$PID_TTS/cmdline"
PY=${CMD_TTS[0]}
VOZ_TOKEN=$(tr '\0' '\n' < "/proc/$PID_TTS/environ" | sed -n 's/^VOZ_TOKEN=//p')
export VOZ_TOKEN

PID_STT=$(systemctl show -p MainPID --value homelab-whisper)
mapfile -d '' -t CMD_STT < "/proc/$PID_STT/cmdline"
ARGS=()
salta=0
for ((i = 1; i < ${#CMD_STT[@]}; i++)); do
  if [ "$salta" = 1 ]; then salta=0; continue; fi
  case "${CMD_STT[$i]}" in
    --threads|--port|--host) salta=1 ;;
    *) ARGS+=("${CMD_STT[$i]}") ;;
  esac
done
echo "[stt] produccion: ${CMD_STT[*]}"

"$PY" cli/banco_md5.py --url http://127.0.0.1:8082 --etiqueta stt --rondas 1 --wav "$SALIDA/wav" \
    --salida "$SALIDA/tts.json" || { echo "[stt] no se generaron los audios"; exit 1; }

"${CMD_STT[0]}" "${ARGS[@]}" --threads 4 --host 127.0.0.1 --port 8091 > "$SALIDA/whisper4.log" 2>&1 &
W4=$!
trap 'kill $W4 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  curl -fsS -m 3 http://127.0.0.1:8091/ >/dev/null 2>&1 && break
  sleep 2
done

transcribir() {  # $1 puerto  $2 wav
  curl -sS -m 300 "http://127.0.0.1:$1/inference" -F "file=@$2" -F response_format=json -F temperature=0
}

iguales=0
total=0
for wav in "$SALIDA"/wav/*.wav; do
  n=$(basename "$wav" .wav)
  for h in 6 4; do
    puerto=8081
    [ "$h" = 4 ] && puerto=8091
    for r in 1 2 3; do
      ini=$(date +%s.%N)
      transcribir "$puerto" "$wav" > "$SALIDA/$n.h$h.r$r.json"
      fin=$(date +%s.%N)
      echo "$n h$h r$r $(awk -v a="$ini" -v b="$fin" 'BEGIN{printf "%.2f", b-a}') s" >> "$SALIDA/tiempos.txt"
    done
  done
  total=$((total + 1))
  h6=$(sha256sum "$SALIDA/$n".h6.r*.json | cut -d' ' -f1 | sort -u | wc -l)
  h4=$(sha256sum "$SALIDA/$n".h4.r*.json | cut -d' ' -f1 | sort -u | wc -l)
  if cmp -s "$SALIDA/$n.h6.r1.json" "$SALIDA/$n.h4.r1.json"; then
    iguales=$((iguales + 1)); estado=IGUAL
  else
    estado=DISTINTO
  fi
  echo "[stt] $n: $estado (variantes entre repeticiones: 6h=$h6 4h=$h4) $(tr -d '\n' < "$SALIDA/$n.h6.r1.json" | cut -c1-120)"
done
awk '{s[$2]+=$4; n[$2]++} END{for (h in s) printf "[stt] %s: %.2f s de media por transcripcion\n", h, s[h]/n[h]}' "$SALIDA/tiempos.txt"
echo "[stt] $iguales/$total transcripciones identicas entre 6 y 4 hilos"
[ "$iguales" = "$total" ] && echo "[stt] A5 PASA" || echo "[stt] A5 NO PASA"
echo "[stt] FIN"
