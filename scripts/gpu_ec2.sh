#!/usr/bin/env bash
# Una maquina GPU en EC2 para el plan de mejora, con el gasto bajo control.
#
#   bash scripts/gpu_ec2.sh lanzar [tipo] [horas]   # g4dn.xlarge y 3 h por defecto
#   bash scripts/gpu_ec2.sh ip                       # IP publica de la que este viva
#   bash scripts/gpu_ec2.sh gasto                    # lo gastado y lo que queda del tope
#   bash scripts/gpu_ec2.sh terminar                 # la termina ya (y apunta el gasto)
#
# Salvaguardas, por orden:
#   1. No arranca si el gasto apuntado mas el de esta tanda (horas x tarifa) pasa el TOPE.
#   2. La instancia se apaga sola a las N horas (shutdown dentro de la maquina) y, como se lanza
#      con InstanceInitiatedShutdownBehavior=terminate, apagarse es TERMINARSE: no queda disco ni
#      factura. Aunque el Mac se apague o esta sesion muera, a las N horas deja de costar.
#   3. Una sola a la vez (la cuota de GPU es de 8 vCPU).
# El gasto se apunta en $LIBRO AL LANZAR con el peor caso (las horas enteras) y se corrige al
# terminar con el tiempo real: si la maquina se termina sola, el libro ya tiene lo que costo.
set -euo pipefail
REGION=us-east-1
TOPE="${TOPE:-20}"                     # USD para todo el plan (Juan, 22-09-2026)
AMI="${AMI:-ami-012ba162b9cd2729c}"    # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
CLAVE=mejora-modelo
LLAVE="$HOME/.ssh/$CLAVE.pem"
SG_NOMBRE=mejora-modelo-ssh
LIBRO="$HOME/Documents/mejora-modelo/aws_gasto.json"
tarifa() {   # USD/h bajo demanda en us-east-1, consultadas en la API de precios el 22-09-2026
  case "$1" in g4dn.xlarge) echo 0.526;; g4dn.2xlarge) echo 0.752;; g6.xlarge) echo 0.8048;;
               g5.xlarge) echo 1.006;; *) echo "";; esac; }
A=(aws --region "$REGION")
mkdir -p "$(dirname "$LIBRO")"
[[ -f "$LIBRO" ]] || echo '{"tandas": []}' > "$LIBRO"

gastado() { python3 -c "import json;print(round(sum(t['usd'] for t in json.load(open('$LIBRO'))['tandas']),3))"; }
viva() { "${A[@]}" ec2 describe-instances --filters "Name=tag:proyecto,Values=mejora-modelo" \
           "Name=instance-state-name,Values=pending,running,stopping" \
           --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,PublicIpAddress]' --output text; }

preparar_acceso() {
  if [[ ! -f "$LLAVE" ]]; then
    "${A[@]}" ec2 create-key-pair --key-name "$CLAVE" --query KeyMaterial --output text > "$LLAVE"
    chmod 600 "$LLAVE"
  fi
  SG=$("${A[@]}" ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NOMBRE" \
         --query 'SecurityGroups[0].GroupId' --output text)
  if [[ "$SG" == "None" ]]; then
    SG=$("${A[@]}" ec2 create-security-group --group-name "$SG_NOMBRE" \
           --description "SSH solo desde la IP de Juan (plan de mejora del modelo)" --query GroupId --output text)
  fi
  IP=$(curl -s -m 5 https://checkip.amazonaws.com)
  # El proveedor rota la IP de salida dentro de su /24 (.36, .58...): se permite el bloque, no una IP
  IP="${IP%.*}.0"
  "${A[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
     --cidr "$IP/24" >/dev/null 2>&1 || true            # ya estaba: no pasa nada
  echo "$SG"
}

case "${1:-}" in
  lanzar)
    TIPO="${2:-g4dn.xlarge}"; HORAS="${3:-3}"
    T=$(tarifa "$TIPO"); [[ -n "$T" ]] || { echo "tipo sin tarifa apuntada: $TIPO"; exit 1; }
    [[ -z "$(viva)" ]] || { echo "ya hay una viva:"; viva; exit 1; }
    PREVISTO=$(python3 -c "print(round($T * $HORAS + 0.05, 3))")   # +0,05 de disco y red
    YA=$(gastado)
    if python3 -c "import sys; sys.exit(0 if $YA + $PREVISTO <= $TOPE else 1)"; then :; else
      echo "[gpu] NO: gastado $YA + previsto $PREVISTO pasa el tope de $TOPE USD. Se sigue en el Mac y los servidores."
      exit 3
    fi
    SG=$(preparar_acceso)
    ID=$("${A[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$TIPO" --count 1 \
          --key-name "$CLAVE" --security-group-ids "$SG" \
          --instance-initiated-shutdown-behavior terminate \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
          --tag-specifications "ResourceType=instance,Tags=[{Key=proyecto,Value=mejora-modelo},{Key=Name,Value=mejora-modelo-gpu}]" \
          --user-data "#!/bin/bash
shutdown -h +$((HORAS * 60))" \
          --query 'Instances[0].InstanceId' --output text)
    python3 - "$LIBRO" "$ID" "$TIPO" "$HORAS" "$PREVISTO" <<'PY'
import json, sys
from datetime import datetime, timezone
libro, id_, tipo, horas, usd = sys.argv[1:6]
d = json.load(open(libro))
d["tandas"].append({"id": id_, "tipo": tipo, "inicio": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "horas_max": float(horas), "usd": float(usd), "cerrada": False})
json.dump(d, open(libro, "w"), indent=1)
PY
    echo "[gpu] $ID $TIPO lanzada; se termina sola a las $HORAS h. Apuntado el peor caso ($PREVISTO USD); gastado con eso $(gastado) de $TOPE."
    "${A[@]}" ec2 wait instance-running --instance-ids "$ID"
    viva ;;
  ip) viva | awk '{print $4}' ;;
  gasto)
    echo "gastado $(gastado) USD de $TOPE"
    viva | while read -r id tipo ini ip; do
      python3 -c "
from datetime import datetime, timezone
h=(datetime.now(timezone.utc)-datetime.fromisoformat('$ini'.replace('Z','+00:00'))).total_seconds()/3600
print(f'viva $id $tipo: {h:.2f} h, {h*$(tarifa "$tipo"):.3f} USD en curso')"
    done ;;
  terminar)
    viva | while read -r id tipo ini ip; do
      "${A[@]}" ec2 terminate-instances --instance-ids "$id" >/dev/null
      python3 - "$LIBRO" "$id" "$ini" "$(tarifa "$tipo")" <<'PY'
import json, sys
from datetime import datetime, timezone
libro, id_, ini, tarifa = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
h = (datetime.now(timezone.utc) - datetime.fromisoformat(ini.replace("Z", "+00:00"))).total_seconds() / 3600
d = json.load(open(libro))
for t in d["tandas"]:
    if t["id"] == id_:
        t.update(horas=round(h, 3), usd=round(h * tarifa + 0.05, 3), cerrada=True)
json.dump(d, open(libro, "w"), indent=1)
print(f"[gpu] {id_} terminada: {h:.2f} h, {h * tarifa + 0.05:.3f} USD")
PY
    done
    echo "gastado $(gastado) USD de $TOPE" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
