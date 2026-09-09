#!/usr/bin/env bash
# La puerta barata: cuanto tarda Qwen3-TTS-0.6B con el motor C en ESTA CPU.
#
#   scripts/banco_rtf_vm.sh banco/ref_h0.wav [salida]
#
# Corre en la VM (voz o taller) con QWEN3TTS_BIN y QWEN3TTS_MODELO puestos
# (nix/taller.nix los exporta), o en cualquier caja con el binario y los
# pesos. Mide, por combinacion de cuantizacion e hilos, el RTF que imprime el
# propio motor (solo generacion, sin la carga) y el pico de RSS (VmHWM), sobre
# las tres frases en espanol del banco con la voz clonada de la referencia.
#
# Umbral del plan: RTF <= 1,0 y RSS <= 3,5 GB. Si no se cumple con ninguna
# combinacion, el modelo no va a esta VM en tiempo real y pasa al lote.
set -euo pipefail

REF=${1:?referencia wav}
SALIDA=${2:-banco_rtf}
BIN=${QWEN3TTS_BIN:-qwen_tts}
MODELO=${QWEN3TTS_MODELO:?QWEN3TTS_MODELO no definido}
mkdir -p "$SALIDA"

FRASES=(
  "El backup de anoche termino sin errores y los tres servicios responden con normalidad."
  "En serio? No me lo puedo creer! Eso si que no me lo esperaba para nada, de verdad."
  "El problema no es el precio, es que nadie te explica lo que estas comprando. Te ensenan una tabla con veinte filas, te sonrien, y cuando preguntas por la letra pequena te dicen que eso ya lo veremos mas adelante, cuando firmes."
)

echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || sysctl -n machdep.cpu.brand_string)"
echo "flags: $(grep -m1 flags /proc/cpuinfo 2>/dev/null | grep -o -E 'avx2|avx512f|avx512_vnni|fma' | sort -u | tr '\n' ' ')"
echo "ram: $(free -m 2>/dev/null | awk '/Mem:/{print $2" MB"}' || echo '?')"
echo

printf "%-8s %-6s %-6s %-8s %-8s %-8s %-9s\n" cuant hilos frase audio_s gen_s rtf rss_mb | tee "$SALIDA/tabla.txt"
for cuant in int8 int4; do
  for hilos in 4 6; do
    i=0
    for frase in "${FRASES[@]}"; do
      i=$((i + 1))
      log="$SALIDA/${cuant}_j${hilos}_f${i}.log"
      # Pico de RSS: GNU time -v en Linux (kB), BSD time -l en macOS (bytes).
      if [[ "$(uname)" == "Darwin" ]]; then bandera=-l; else bandera=-v; fi
      /usr/bin/time $bandera "$BIN" -d "$MODELO" --ref-audio "$REF" -l Spanish "--$cuant" -j "$hilos" \
        --seed 11 --text "$frase" -o "$SALIDA/${cuant}_j${hilos}_f${i}.wav" >"$log" 2>&1 || true
      audio=$(grep -o -E 'Audio: [0-9.]+s' "$log" | grep -o -E '[0-9.]+' | head -1)
      gen=$(grep -o -E 'generated in [0-9.]+s' "$log" | grep -o -E '[0-9.]+' | head -1)
      rtf=$(grep -o -E 'RTF [0-9.]+' "$log" | grep -o -E '[0-9.]+' | head -1)
      rss=$(grep -o -E 'Maximum resident set size \(kbytes\): [0-9]+' "$log" | grep -o -E '[0-9]+$' | awk '{printf "%d", $1/1024}')
      [[ -z "$rss" ]] && rss=$(grep -E 'maximum resident set size' "$log" | awk '{printf "%d", $1/1048576}')
      printf "%-8s %-6s %-6s %-8s %-8s %-8s %-9s\n" "$cuant" "$hilos" "$i" "${audio:-?}" "${gen:-?}" "${rtf:-?}" "${rss:-?}" | tee -a "$SALIDA/tabla.txt"
    done
  done
done
echo; echo "tabla en $SALIDA/tabla.txt; logs y wav al lado"
