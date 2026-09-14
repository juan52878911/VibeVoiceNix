#!/usr/bin/env bash
# Puerta de la fase 1 (docs/plan-rendimiento.md): base -> variante -> base -> variante.
#
#   ssh -n root@voz 'systemd-run --unit fase1-ab --collect bash /root/lab/cli/fase1_ab.sh'
#
# Cada tanda es un proceso nuevo de voz-stream en 127.0.0.1:8092 (fuera de la LAN), con el MISMO
# entorno que la unidad de produccion. La base es el codigo del store; la variante, /root/lab/ov con el
# mismo voz_stream.py. banco_md5.py: 3 rondas por tanda (la 0 no cuenta), md5, RTF y VmHWM del
# servidor. ws_fidelidad.py completo contra la primera variante, despues de su banco.
# Para voz-stream y lo devuelve al salir, falle lo que falle.
set -uo pipefail
export PATH=/run/current-system/sw/bin:$PATH
for orden in pgrep systemctl sleep curl; do
  command -v "$orden" >/dev/null || { echo "[ab] falta $orden: aborto"; exit 1; }
done

PY=/nix/store/68d5yzqf9iqgvfwn2lv4hky2khin1i0f-vibevoice-env/bin/python3.12
BASE_CLI=/nix/store/wgfzr635kdqg4c3lfg6crda6lqjpqy8i-vibevoice-cli/bin/voz-stream.py
BASE_OV=/nix/store/c4ag5lb1v5l4786ryjxj4xcgr3104g9b-vibevoice-ov-codigo
LAB=/root/lab
PUERTO=8092
SALIDA=$LAB/fase1
mkdir -p "$SALIDA"
cd "$LAB" || exit 1

if pgrep -fa 'fase[0-9]_' | grep -v -e pgrep -e fase1_ab; then
  echo "[ab] hay un experimento de otra sesion corriendo: no se toca nada"
  exit 1
fi

# El entorno de produccion, tal cual lo tiene la unidad
for par in $(systemctl show -p Environment --value voz-stream); do
  export "${par?}"
done
set -a
# shellcheck disable=SC1091
. /var/lib/voz/token.env
set +a
[ -n "${VOZ_TOKEN:-}" ] || { echo "[ab] sin VOZ_TOKEN: aborto"; exit 1; }

# /run/voz-stream/voces desaparece al parar la unidad: se rehace igual que su ExecStartPre
VOCES=$LAB/voces
rm -rf "$VOCES"
mkdir -p "$VOCES"
voces_store=$(dirname "$(readlink -f /run/voz-stream/voces/sp-Spk1_man.pt 2>/dev/null || true)")
[ -d "$voces_store" ] || { echo "[ab] no encuentro las voces del store (voz-stream parado?): aborto"; exit 1; }
n=0
for f in "$voces_store"/*.pt /var/lib/voz/voces-propias/*.pt; do
  [ -e "$f" ] && ln -sf "$f" "$VOCES/" && n=$((n + 1))
done
echo "[ab] $n voces enlazadas desde $voces_store"
export VIBEVOICE_VOCES=$VOCES VOZ_STREAM_HOST=127.0.0.1 VOZ_STREAM_PUERTO=$PUERTO

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

tanda() {  # $1 etiqueta  $2 voz-stream.py  $3 codigo OV
  local etiqueta=$1
  VIBEVOICE_OV_CODIGO=$3 $PY "$2" > "$SALIDA/$etiqueta.log" 2>&1 &
  SERVIDOR=$!
  for _ in $(seq 1 100); do
    curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1 && break
    sleep 3
  done
  if ! curl -fsS -m 3 "http://127.0.0.1:$PUERTO/health" >/dev/null 2>&1; then
    echo "[ab] $etiqueta no arranco"; tail -20 "$SALIDA/$etiqueta.log"; exit 1
  fi
  tr '\r' '\n' < "$SALIDA/$etiqueta.log" | grep -E '\[carga\]|modelo listo'
  $PY cli/banco_md5.py --url "http://127.0.0.1:$PUERTO" --etiqueta "$etiqueta" --rondas 3 \
      --pid "$SERVIDOR" --salida "$SALIDA/$etiqueta.json"
  if [ "$etiqueta" = variante-1 ]; then
    echo "[ab] ws_fidelidad contra la variante"
    $PY cli/ws_fidelidad.py --url "http://127.0.0.1:$PUERTO" --token "$VOZ_TOKEN" \
        > "$SALIDA/ws_fidelidad.log" 2>&1
    echo "[ab] ws_fidelidad codigo de salida $?"
    tail -25 "$SALIDA/ws_fidelidad.log"
  fi
  kill "$SERVIDOR"
  wait "$SERVIDOR" 2>/dev/null
  SERVIDOR=""
  sleep 3
}

tanda base-1 "$BASE_CLI" "$BASE_OV"
tanda variante-1 "$LAB/cli/voz_stream.py" "$LAB/ov"
tanda base-2 "$BASE_CLI" "$BASE_OV"
tanda variante-2 "$LAB/cli/voz_stream.py" "$LAB/ov"

$PY cli/banco_md5.py comparar "$SALIDA"/base-1.json "$SALIDA"/variante-1.json \
    "$SALIDA"/base-2.json "$SALIDA"/variante-2.json
echo "[ab] FIN"
