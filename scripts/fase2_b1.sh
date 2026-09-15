#!/usr/bin/env bash
# Fase 2, B1 (docs/plan-rendimiento.md): la VM voz con su topología real (6 núcleos x 2 hilos) frente a
# la de hoy (12 núcleos), en 4 tandas alternas base -> b1 -> base -> b1, cada una con la VM arrancada
# desde cero.
#
#   bash scripts/fase2_b1.sh < /dev/null        # desde el Mac
#
# En cada tanda:
#   - para la VM 210 (ACPI; si no responde en 180 s, stop);
#   - pone o quita `args: -smp 12,sockets=1,cores=6,threads=2` y la arranca;
#   - espera a /health y a que voz-stream-sin-swap termine;
#   - corre banco_md5.py a 3 rondas (la 0 no cuenta).
# ws_fidelidad.py completo en la primera tanda b1. Al salir, falle lo que falle, la VM queda sin `args`
# y arrancada. Solo la deja con B1 una ejecución aparte, si la puerta pasa.
set -uo pipefail

PVE=(ssh -n -o ConnectTimeout=10 pve)
VMS=(ssh -n -i "$HOME/.ssh/oracle_a1" -o IdentitiesOnly=yes -o ConnectTimeout=10 root@192.168.2.54)
TOPO="-smp 12,sockets=1,cores=6,threads=2,maxcpus=12"
REMOTO=/root/lab/fase2_b1
LOCAL=${TMPDIR:-/tmp}/fase2_b1
mkdir -p "$LOCAL"

revertir() {
  echo "[b1] dejando la VM sin args y arrancada"
  "${PVE[@]}" "if qm config 210 | grep -q '^args:'; then qm shutdown 210 --timeout 180 || qm stop 210; qm set 210 --delete args; fi; qm status 210 | grep -q running || qm start 210; qm config 210 | grep '^args:' || echo 'sin args'; qm status 210"
}

if "${VMS[@]}" 'pgrep -fa "fase[0-9]_|ws_fidelidad|evaluar_clones|benchmark_app" | grep -v pgrep'; then
  echo "[b1] hay otra carga en la VM: aborto"; exit 1
fi
if "${PVE[@]}" 'qm config 210 | grep -q "^args:"'; then
  echo "[b1] la VM 210 ya tiene args puestos: no se toca nada"; exit 1
fi
trap revertir EXIT
"${VMS[@]}" "rm -rf $REMOTO && mkdir -p $REMOTO"

arrancar_con() {  # $1 = base | b1
  "${PVE[@]}" "qm shutdown 210 --timeout 180 || qm stop 210"
  if [ "$1" = b1 ]; then
    "${PVE[@]}" "qm set 210 --args '$TOPO'"
  else
    "${PVE[@]}" "qm config 210 | grep -q '^args:' && qm set 210 --delete args; true"
  fi
  "${PVE[@]}" "qm start 210" || return 1
  for _ in $(seq 1 90); do
    sleep 10
    # shellcheck disable=SC2016  # se expande en la VM, no aqui
    if "${VMS[@]}" 'curl -fsS -m 3 127.0.0.1:8082/health >/dev/null && [ "$(systemctl show -p ActiveState --value voz-stream-sin-swap)" = active ]' 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

tanda() {  # $1 etiqueta  $2 modo
  echo "[b1] === $1 ($2) $(date +%H:%M:%S)"
  arrancar_con "$2" || { echo "[b1] la VM no volvió en la tanda $1"; exit 1; }
  "${PVE[@]}" "qm showcmd 210 --pretty | grep -E '^\s*-smp'"
  "${VMS[@]}" "echo \"guest: \$(nproc) CPU, \$(lscpu | grep -E '^(Thread|Core)' | tr -s ' ' | tr '\n' ';')\""
  "${VMS[@]}" "P=\$(systemctl show -p MainPID --value voz-stream)
    PY=\$(tr '\\0' '\\n' < /proc/\$P/cmdline | head -1)
    export VOZ_TOKEN=\$(tr '\\0' '\\n' < /proc/\$P/environ | sed -n 's/^VOZ_TOKEN=//p')
    cd /root/lab && \$PY cli/banco_md5.py --url http://127.0.0.1:8082 --etiqueta $1 --rondas 3 --pid \$P --salida $REMOTO/$1.json
    if [ $1 = b1-1 ]; then
      \$PY cli/ws_fidelidad.py --url http://127.0.0.1:8082 --token \"\$VOZ_TOKEN\" > $REMOTO/ws_b1.log 2>&1
      echo \"ws_fidelidad codigo \$?: \$(tail -1 $REMOTO/ws_b1.log)\"
    fi"
}

tanda base-1 base
tanda b1-1 b1
tanda base-2 base
tanda b1-2 b1

scp -q -i "$HOME/.ssh/oracle_a1" -o IdentitiesOnly=yes "root@192.168.2.54:$REMOTO/*.json" "root@192.168.2.54:$REMOTO/ws_b1.log" "$LOCAL/"
python3 - "$LOCAL" <<'PY'
import json, statistics, sys
d = sys.argv[1]
bancos = {e: json.load(open(f"{d}/{e}.json")) for e in ("base-1", "b1-1", "base-2", "b1-2")}
ref = bancos["base-1"]["rondas"][0]["clips"]
md5_ok = all(c["md5"] == ref[c["frase"]]["md5"] for b in bancos.values() for r in b["rondas"] for c in r["clips"])
def rtf(modo):
    return [r["rtf"] for e, b in bancos.items() if e.startswith(modo) for r in b["rondas"][1:]]
def iqr_rel(v):
    v = sorted(v); n = len(v)
    q = lambda p: v[min(n - 1, int(p * (n - 1) + 0.5))]
    return (q(0.75) - q(0.25)) / statistics.median(v)
base, b1 = rtf("base"), rtf("b1")
mb, m1 = statistics.median(base), statistics.median(b1)
ws_ok = "todo correcto" in open(f"{d}/ws_b1.log").read()
rtf_ok = m1 <= 0.97 * mb
var_ok = iqr_rel(b1) <= 0.5 * iqr_rel(base)
print(f"md5 identico en todo: {md5_ok}")
print(f"ws_fidelidad con B1: {'todo correcto' if ws_ok else 'FALLA'}")
print(f"RTF base {base} mediana {mb:.4f} IQR {100*iqr_rel(base):.1f} %")
print(f"RTF b1   {b1} mediana {m1:.4f} IQR {100*iqr_rel(b1):.1f} %")
print(f"B1/base = {m1/mb:.4f} (puerta <= 0,97: {rtf_ok}) · IQR b1 <= mitad del de base: {var_ok}")
for e, b in bancos.items():
    print(e, "VmHWM", b.get("memoria"), "reparto ronda 2", b["rondas"][-1].get("reparto"))
print("B1 PASA" if md5_ok and ws_ok and (rtf_ok or var_ok) else "B1 NO PASA")
PY
echo "[b1] FIN"
