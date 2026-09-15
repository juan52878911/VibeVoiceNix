#!/usr/bin/env bash
# Despliega HEAD en la VM voz y verifica que el audio no cambia.
#
#   bash scripts/desplegar_vm_voz.sh < /dev/null            # desde el Mac, en la rama a desplegar
#
# 1. Comprueba que no corre nada de otra sesión (fase*, ws_fidelidad, evaluar_clones, benchmark_app).
# 2. `git archive HEAD` a /root/despliegue-rend y `nixos-rebuild switch --flake path:...#voz` en la VM
#    (en el Mac no hay nixos-rebuild).
# 3. Verifica:
#    - voz-stream activo, con el motor.py de HEAD;
#    - VmHWM tras arrancar;
#    - md5 de las 8 frases de banco_md5.py igual al de la base del A/B de la fase 1
#      (/root/lab/fase1/base-1.json);
#    - ws_fidelidad.py completo.
set -euo pipefail

VM=root@192.168.2.54
SSHV=(ssh -i "$HOME/.ssh/oracle_a1" -o IdentitiesOnly=yes)
DESTINO=/root/despliegue-rend
REV=$(git rev-parse --short HEAD)
MOTOR_SHA=$(git show HEAD:pkgs/vibevoice-ov/motor.py | shasum -a 256 | cut -d' ' -f1)
[ -z "$(git status --porcelain -- pkgs nix flake.nix flake.lock)" ] || { echo "hay cambios sin commit en pkgs/nix: aborto"; exit 1; }

if "${SSHV[@]}" -n "$VM" 'pgrep -fa "fase[0-9]_|ws_fidelidad|evaluar_clones|benchmark_app|nixos-rebuild|musica|arranque|ablacion" | grep -v pgrep'; then
  echo "hay otra carga en la VM: aborto"; exit 1
fi

echo "[despliegue] $REV (motor.py $MOTOR_SHA)"
git archive --format=tar HEAD | "${SSHV[@]}" "$VM" "rm -rf $DESTINO && mkdir -p $DESTINO && tar x -C $DESTINO"
"${SSHV[@]}" -n "$VM" "cd $DESTINO && nixos-rebuild switch --flake path:$DESTINO#voz 2>&1 | tail -15"

# Sin -n: el heredoc ES la entrada de este ssh (con -n llegaria /dev/null y no se ejecutaria nada).
# El guion entero se lanza con < /dev/null, asi que ningun otro ssh se come esta entrada.
"${SSHV[@]}" "$VM" "MOTOR_SHA=$MOTOR_SHA bash -s" <<'EOF'
set -uo pipefail
for _ in $(seq 1 120); do
  curl -fsS -m 3 http://127.0.0.1:8082/health >/dev/null 2>&1 && break
  sleep 5
done
systemctl is-active voz-stream
P=$(systemctl show -p MainPID --value voz-stream)
OV=$(tr '\0' '\n' < /proc/$P/environ | sed -n 's/^VIBEVOICE_OV_CODIGO=//p')
[ "$(sha256sum "$OV/motor.py" | cut -d' ' -f1)" = "$MOTOR_SHA" ] && echo "motor.py desplegado = HEAD" || { echo "motor.py desplegado DISTINTO"; exit 1; }
grep -E 'VmHWM|VmRSS|VmSwap' /proc/$P/status
journalctl -u voz-stream -b --no-pager | grep -E '\[carga\]|modelo listo|\[arranque\] codigo' | tail -6 | cut -c30-200
PY=$(tr '\0' '\n' < /proc/$P/cmdline | head -1)
export VOZ_TOKEN=$(tr '\0' '\n' < /proc/$P/environ | sed -n 's/^VOZ_TOKEN=//p')
cd /root/lab
$PY cli/banco_md5.py --url http://127.0.0.1:8082 --etiqueta desplegado --rondas 2 --pid "$P" --salida fase1/desplegado.json
$PY cli/banco_md5.py comparar fase1/base-1.json fase1/desplegado.json
$PY cli/ws_fidelidad.py --url http://127.0.0.1:8082 --token "$VOZ_TOKEN" > fase1/ws_desplegado.log 2>&1
echo "ws_fidelidad codigo $?"; tail -3 fase1/ws_desplegado.log
EOF
