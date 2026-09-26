#!/usr/bin/env bash
# Una maquina de CPU en SPOT para la campana de la red por dentro (scripts/red), con el gasto bajo control.
# Calcado de scripts/gpu_ec2.sh: mismo tope, mismo libro de gasto, misma etiqueta; cambia que es spot
# de una sola vez, Ubuntu 22.04 a secas (sin CUDA) y la tarifa se lee del mercado al lanzar.
#
#   bash scripts/red/spot_ec2.sh lanzar [tipo] [horas]   # c8a.2xlarge y 3 h por defecto
#   bash scripts/red/spot_ec2.sh ip                       # IP publica de la que este viva
#   bash scripts/red/spot_ec2.sh subir                    # copia scripts/ y el lock de pkgs/vibevoice a ~/mejora
#   bash scripts/red/spot_ec2.sh credenciales [horas]     # credenciales TEMPORALES de STS para el s3 sync (12 h)
#   bash scripts/red/spot_ec2.sh gasto                    # lo gastado y lo que queda del tope
#   bash scripts/red/spot_ec2.sh terminar                 # la termina ya (y apunta el gasto)
#
# Salvaguardas, por orden:
#   1. No arranca si el gasto apuntado mas el de esta tanda (horas x tarifa spot mas alta de la region en
#      ese momento) pasa el TOPE.
#   2. La instancia se apaga sola a las N horas (shutdown dentro de la maquina) y, como se lanza con
#      InstanceInitiatedShutdownBehavior=terminate, apagarse es TERMINARSE: no queda disco ni factura.
#   3. Solicitud spot de UNA SOLA VEZ (one-time, terminate al interrumpir): si AWS la reclama, muere y no
#      renace. Reanudar es volver a `lanzar`: campana.sh sigue por donde iba (el trabajo vive en S3).
#   4. Una sola del proyecto a la vez (comparte la etiqueta proyecto=mejora-modelo con la de GPU).
# El gasto se apunta en $LIBRO AL LANZAR con el peor caso y se corrige con el tiempo real en `terminar`,
# `gasto` o el siguiente `lanzar`: si la maquina murio sola (tope de horas o interrupcion de spot), el libro
# se cierra con la hora de fin que AWS conserva de ella (~1 h) y se marca si fue una interrupcion.
set -euo pipefail
REGION=us-east-1
TOPE="${TOPE:-20}"                     # USD para todo el plan (Juan, 22-09-2026), compartido con gpu_ec2.sh
CLAVE=mejora-modelo
LLAVE="$HOME/.ssh/$CLAVE.pem"
SG_NOMBRE=mejora-modelo-ssh
LIBRO="${LIBRO:-$HOME/Documents/mejora-modelo/aws_gasto.json}"   # el de gpu_ec2.sh; otro con LIBRO=...
NOMBRE=mejora-modelo-spot
DISCO="${DISCO:-40}"                   # GB: modelo 2 GB, codificador 1,4 GB, entorno ~4 GB, jueces ~2 GB, campana
A=(aws --region "$REGION")
SSH_OPC=(-i "$LLAVE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)
# Sin libro no hay tope: ~/Documents/mejora-modelo es un enlace a un disco externo (/Volumes/VIDEOS) y, si no
# esta montado, crear un libro vacio haria creer que no se ha gastado nada. Se para; con LIBRO=<otro> se sigue,
# y ese libro tiene que llevar el gasto de antes (una tanda "arrastre") hasta fusionarlo con el de siempre.
if [[ ! -f "$LIBRO" ]]; then
  echo "[spot] NO: no encuentro el libro de gasto $LIBRO (¿disco externo sin montar?)." >&2
  echo "       Monta el disco, o pasa LIBRO=<ruta> con un libro que ya lleve el gasto acumulado." >&2
  exit 4
fi

ami_ubuntu() {   # la Ubuntu 22.04 (jammy, amd64) mas reciente de Canonical; no hace falta CUDA ni la DLAMI
  "${A[@]}" ec2 describe-images --owners 099720109477 \
     --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
     --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text; }
tarifa() {   # USD/h spot AHORA para el tipo: la mas alta de las zonas de la region (peor caso para el tope)
  "${A[@]}" ec2 describe-spot-price-history --instance-types "$1" --product-descriptions "Linux/UNIX" \
     --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'max_by(SpotPriceHistory,&to_number(SpotPrice)).SpotPrice' --output text; }
zona_barata() {   # la zona con el spot mas barato ahora mismo para el tipo (ZONA=... la fija a mano)
  "${A[@]}" ec2 describe-spot-price-history --instance-types "$1" --product-descriptions "Linux/UNIX" \
     --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'min_by(SpotPriceHistory,&to_number(SpotPrice)).AvailabilityZone' --output text; }
tarifa_zona() {   # USD/h spot ahora en la zona de la instancia: lo que de verdad se paga
  "${A[@]}" ec2 describe-spot-price-history --instance-types "$1" --availability-zone "$2" --product-descriptions "Linux/UNIX" \
     --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query 'SpotPriceHistory[0].SpotPrice' --output text; }
gastado() { python3 -c "import json;print(round(sum(t['usd'] for t in json.load(open('$LIBRO'))['tandas']),3))"; }
alguna() { "${A[@]}" ec2 describe-instances --filters "Name=tag:proyecto,Values=mejora-modelo" \
             "Name=instance-state-name,Values=pending,running,stopping" \
             --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text; }
viva() { "${A[@]}" ec2 describe-instances --filters "Name=tag:proyecto,Values=mejora-modelo" "Name=tag:Name,Values=$NOMBRE" \
           "Name=instance-state-name,Values=pending,running,stopping" \
           --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,PublicIpAddress,Placement.AvailabilityZone]' --output text; }
ip_viva() { local ip; ip=$(viva | awk '{print $4}'); [[ -n "$ip" && "$ip" != "None" ]] || { echo "no hay ninguna viva" >&2; exit 1; }; echo "$ip"; }

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
  IP="${IP%.*}.0"    # el proveedor rota la IP de salida dentro de su /24: se permite el bloque
  "${A[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
     --cidr "$IP/24" >/dev/null 2>&1 || true            # ya estaba: no pasa nada
  echo "$SG"
}

# Cierra en el libro las tandas spot abiertas cuya instancia ya no vive: horas hasta la hora de fin que AWS
# conserva, a la tarifa spot mas alta que tuvo su zona durante la tanda; marca si fue una interrupcion.
cerrar_muertas() {
  python3 - "$LIBRO" "$REGION" <<'FIN_PY'
import json, re, subprocess, sys
from datetime import datetime, timezone
libro, region = sys.argv[1], sys.argv[2]
aws = ["aws", "--region", region, "ec2"]
d = json.load(open(libro))
cambio = False
for t in d["tandas"]:
    if t.get("cerrada") or t.get("mercado") != "spot":
        continue
    r = subprocess.run(aws + ["describe-instances", "--instance-ids", t["id"], "--output", "json"], capture_output=True, text=True)
    if r.returncode:
        print(f"[spot] {t['id']}: AWS ya no la conoce; la tanda queda con el peor caso apuntado ({t['usd']} USD)")
        t.update(cerrada=True, nota="sin datos de AWS: peor caso")
        cambio = True
        continue
    i = json.loads(r.stdout)["Reservations"][0]["Instances"][0]
    if i["State"]["Name"] not in ("terminated", "shutting-down"):
        continue
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", i.get("StateTransitionReason", ""))
    fin = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if m else datetime.now(timezone.utc)
    zona = i["Placement"]["AvailabilityZone"]
    p = subprocess.run(aws + ["describe-spot-price-history", "--instance-types", t["tipo"], "--availability-zone", zona,
                              "--product-descriptions", "Linux/UNIX", "--start-time", t["inicio"],
                              "--query", "SpotPriceHistory[].SpotPrice", "--output", "json"], capture_output=True, text=True)
    precios = [float(x) for x in json.loads(p.stdout or "[]")] or [t["tarifa"]]
    tarifa = max(precios)
    h = max(0.0, (fin - datetime.fromisoformat(t["inicio"])).total_seconds() / 3600)
    inter = i.get("StateReason", {}).get("Code") == "Server.SpotInstanceTermination"
    t.update(horas=round(h, 3), tarifa=tarifa, zona=zona, usd=round(h * tarifa + 0.02, 3), cerrada=True,
             interrumpida=inter, fin=fin.isoformat(timespec="seconds"))
    cambio = True
    print(f"[spot] {t['id']} {'INTERRUMPIDA por AWS' if inter else 'terminada'}: {h:.2f} h a {tarifa} $/h, {t['usd']} USD")
if cambio:
    json.dump(d, open(libro, "w"), indent=1)
FIN_PY
}

case "${1:-}" in
  lanzar)
    TIPO="${2:-c8a.2xlarge}"; HORAS="${3:-3}"
    T=$(tarifa "$TIPO"); [[ -n "$T" && "$T" != "None" ]] || { echo "sin precio spot para $TIPO en $REGION"; exit 1; }
    cerrar_muertas
    [[ -z "$(alguna)" ]] || { echo "ya hay una viva del proyecto (spot o GPU):"; alguna; exit 1; }
    PREVISTO=$(python3 -c "print(round($T * $HORAS + 0.02, 3))")   # +0,02 de disco y S3
    YA=$(gastado)
    if python3 -c "import sys; sys.exit(0 if $YA + $PREVISTO <= $TOPE else 1)"; then :; else
      echo "[spot] NO: gastado $YA + previsto $PREVISTO pasa el tope de $TOPE USD. Se sigue en el Mac y los servidores."
      exit 3
    fi
    AMI="${AMI:-$(ami_ubuntu)}"
    ZONA="${ZONA:-$(zona_barata "$TIPO")}"   # sin esto AWS elige la subred por defecto, que puede ser la zona cara
    SG=$(preparar_acceso)
    ID=$("${A[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$TIPO" --count 1 \
          --key-name "$CLAVE" --security-group-ids "$SG" --placement "AvailabilityZone=$ZONA" \
          --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
          --instance-initiated-shutdown-behavior terminate \
          --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISCO,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
          --tag-specifications "ResourceType=instance,Tags=[{Key=proyecto,Value=mejora-modelo},{Key=Name,Value=$NOMBRE}]" \
          --user-data "#!/bin/bash
shutdown -h +$((HORAS * 60))" \
          --query 'Instances[0].InstanceId' --output text)
    python3 - "$LIBRO" "$ID" "$TIPO" "$HORAS" "$PREVISTO" "$T" "$AMI" <<'FIN_PY'
import json, sys
from datetime import datetime, timezone
libro, id_, tipo, horas, usd, tarifa, ami = sys.argv[1:8]
d = json.load(open(libro))
d["tandas"].append({"id": id_, "tipo": tipo, "mercado": "spot", "ami": ami, "tarifa": float(tarifa),
                    "inicio": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "horas_max": float(horas), "usd": float(usd), "cerrada": False})
json.dump(d, open(libro, "w"), indent=1)
FIN_PY
    echo "[spot] $ID $TIPO lanzada en $ZONA a $(tarifa_zona "$TIPO" "$ZONA") \$/h (spot; peor de la region $T; $AMI); se termina sola a las $HORAS h."
    echo "[spot] apuntado el peor caso ($PREVISTO USD); gastado con eso $(gastado) de $TOPE."
    "${A[@]}" ec2 wait instance-running --instance-ids "$ID"
    IP=$(ip_viva)
    for _ in $(seq 30); do ssh -n "${SSH_OPC[@]}" "ubuntu@$IP" true 2>/dev/null && break; sleep 5; done
    viva ;;
  ip) ip_viva ;;
  subir)
    IP=$(ip_viva)
    RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
    ssh -n "${SSH_OPC[@]}" "ubuntu@$IP" "mkdir -p ~/mejora/pkgs/vibevoice"
    rsync -az -e "ssh ${SSH_OPC[*]}" --exclude '__pycache__' --exclude '.venv' --exclude '*.wav' --exclude '*.npz' \
      "$RAIZ/scripts" "ubuntu@$IP:mejora/"
    rsync -az -e "ssh ${SSH_OPC[*]}" "$RAIZ/pkgs/vibevoice/pyproject.toml" "$RAIZ/pkgs/vibevoice/uv.lock" \
      "ubuntu@$IP:mejora/pkgs/vibevoice/"
    echo "[spot] subido a ubuntu@$IP:~/mejora (scripts/ y pkgs/vibevoice/{pyproject.toml,uv.lock})" ;;
  credenciales)
    # Credenciales temporales de STS (caducan solas, 12 h por defecto): la maquina las usa para `aws s3 sync`.
    # Nunca las de larga vida del Mac. Se escriben en ~/.aws de la maquina, que muere con ella.
    IP=$(ip_viva)
    SEG=$(( ${2:-12} * 3600 ))
    CRED=$("${A[@]}" sts get-session-token --duration-seconds "$SEG" \
        --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken,Expiration]' --output text)
    read -r AK SK ST CAD <<<"$CRED"
    printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\naws_session_token = %s\n' "$AK" "$SK" "$ST" |
      ssh "${SSH_OPC[@]}" "ubuntu@$IP" "mkdir -p ~/.aws && umask 077 && cat > ~/.aws/credentials && printf '[default]\nregion = $REGION\n' > ~/.aws/config"
    echo "[spot] credenciales temporales en la maquina hasta $CAD" ;;
  gasto)
    cerrar_muertas
    echo "gastado $(gastado) USD de $TOPE"
    viva | while read -r id tipo ini ip zona; do
      TZ_=$(tarifa_zona "$tipo" "$zona")
      python3 -c "
from datetime import datetime, timezone
h=(datetime.now(timezone.utc)-datetime.fromisoformat('$ini'.replace('Z','+00:00'))).total_seconds()/3600
print(f'viva $id $tipo en $zona ($ip): {h:.2f} h, {h*$TZ_:.3f} USD en curso a $TZ_ \$/h')"
    done ;;
  terminar)
    IDS=$(viva | awk '{print $1}')
    if [[ -n "$IDS" ]]; then
      "${A[@]}" ec2 terminate-instances --instance-ids $IDS >/dev/null
      "${A[@]}" ec2 wait instance-terminated --instance-ids $IDS
    fi
    cerrar_muertas
    echo "gastado $(gastado) USD de $TOPE" ;;
  *) sed -n 5,11p "$0"; exit 1 ;;
esac
