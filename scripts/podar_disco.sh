#!/usr/bin/env bash
# Poda de disco de la VM voz (palanca A3 del plan de rendimiento). EN SECO por defecto.
#
#   bash podar_disco.sh            # lista lo que borraria y cuanto libera
#   bash podar_disco.sh --borrar   # lo borra
#
# Solo IR que no usa nadie y la cache de nix. Se conservan, a proposito:
#   decoder_mm_int8, tts_lm_estado_int4, cabeza_int8, difusion_p6_int8   produccion
#   decoder_mm_int4                                                      control del banco A/B
#   decoder_mm_fp16                                                      iGPU (C1) y referencia de SNR
#   tts_lm_estado_fp16, cabeza_fp16                                      fuentes que relee comprimir()
# tts_lm_estado_int8 se conservaba para el banco C3; C3 se cerró el 15-09-2026 (el int8 no suena mejor,
# docs/plan-rendimiento.md), así que ahora está en la lista de lo que sobra.
# Los marcadores .convertir_*.hecho NO se tocan: el oneshot (nix/modules/vibevoice-ov.nix) ya no
# genera decoder_estado_* (su paso es convertir_decoder:2 = decoder_mm_*), y cabeza_int4 y
# difusion_p6_fp16 no los usa ninguna ruta de voz-stream.nix.
# No toca /var/lib/taller ni /root/.cache/vibevoice-nix: son del taller y se deciden aparte.
set -euo pipefail

DIR=/var/lib/voz/ov
BORRAR=0
[ "${1:-}" = "--borrar" ] && BORRAR=1

SOBRAN=(
  decoder_estado_fp16.bin decoder_estado_fp16.xml
  decoder_estado_int8.bin decoder_estado_int8.xml
  decoder_estado_int4.bin decoder_estado_int4.xml
  cabeza_int4.bin cabeza_int4.xml
  difusion_p6_fp16.bin difusion_p6_fp16.xml
  tts_lm_estado_int8.bin tts_lm_estado_int8.xml
)

# Seguro: nada de la lista puede estar en uso por el servicio
en_uso=$(systemctl show -p Environment voz-stream | tr ' ' '\n' | grep -o '[^/=]*\.xml' || true)
for f in "${SOBRAN[@]}"; do
  if printf '%s\n' "$en_uso" | grep -qx "$f"; then
    echo "ABORTO: $f lo usa voz-stream" >&2
    exit 1
  fi
done

total=0
for f in "${SOBRAN[@]}"; do
  [ -e "$DIR/$f" ] || continue
  b=$(stat -c %s "$DIR/$f")
  total=$((total + b))
  printf '%8d MB  %s\n' $((b / 1048576)) "$DIR/$f"
done
cache_nix=$(du -sb /root/.cache/nix 2>/dev/null | cut -f1 || echo 0)
printf '%8d MB  /root/.cache/nix\n' $((cache_nix / 1048576))
echo "total: $(( (total + cache_nix) / 1048576 )) MB, mas lo que libere nix-collect-garbage --delete-older-than 14d"

if [ "$BORRAR" = 1 ]; then
  for f in "${SOBRAN[@]}"; do rm -f "${DIR:?}/$f"; done
  rm -rf /root/.cache/nix
  nix-collect-garbage --delete-older-than 14d
  df -h /
else
  echo "(en seco: nada borrado; --borrar para hacerlo)"
fi
