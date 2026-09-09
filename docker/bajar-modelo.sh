#!/usr/bin/env bash
# Descarga el modelo de whisper que monta el contenedor.
#
# No va dentro de la imagen a proposito: son 466 MB y ataria la version del
# modelo a la del contenedor. Se verifica el sha256 para que un fallo de red o
# una descarga distinta no pase inadvertida.
set -euo pipefail

MODELO="${1:-small}"
DESTINO="$(dirname "$0")/modelos"

case "$MODELO" in
  small)
    URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    SHA="1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
    ;;
  base)
    URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    SHA="60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
    ;;
  medium)
    # el que usa el doblaje: con small, una reunion de Teams salia
    # ininteligible ("en las zonas de aseo ir al a los otanos") y con medium
    # se entiende; cuesta el 5% del tiempo total del doblaje (medido)
    URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
    SHA="6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208"
    ;;
  *)
    echo "modelo desconocido: $MODELO (usa 'small', 'base' o 'medium')" >&2
    exit 1
    ;;
esac

mkdir -p "$DESTINO"
FICHERO="$DESTINO/ggml-$MODELO.bin"

if [ -f "$FICHERO" ]; then
  echo "ya estaba: $FICHERO"
else
  echo "bajando ggml-$MODELO.bin ..."
  curl -fL --progress-bar "$URL" -o "$FICHERO.parcial"
  mv "$FICHERO.parcial" "$FICHERO"
fi

echo -n "verificando... "
if command -v sha256sum >/dev/null; then
  REAL=$(sha256sum "$FICHERO" | cut -d' ' -f1)
else
  REAL=$(shasum -a 256 "$FICHERO" | cut -d' ' -f1)
fi

if [ "$REAL" != "$SHA" ]; then
  echo "MAL"
  echo "  esperado: $SHA" >&2
  echo "  obtenido: $REAL" >&2
  echo "Borra $FICHERO y vuelve a intentarlo." >&2
  exit 1
fi

echo "ok"
echo
echo "Listo. Si usas 'base' en vez de 'small', cambia tambien el --model del"
echo "servicio whisper en compose.yaml."
