#!/usr/bin/env bash
# whisper.cpp NATIVO en el Mac, sobre Metal. El sumando gordo de la latencia.
#
# POR QUE, MEDIDO 2026-08-07 en este mismo Mac (M4), doce locuciones de
# VibeVoice del largo de una orden real, contra la MISMA batería y midiendo
# también la tasa de error de palabra:
#
#     whisper small en Docker (lo de hoy)   2177 ms de media   WER 15,0 %
#     whisper small NATIVO con Metal         277 ms            WER 15,0 %
#     whisper base  NATIVO con Metal         101 ms            WER 28,3 %
#
# O sea: el MISMO modelo, la MISMA transcripción palabra por palabra (los
# nueve errores son exactamente los mismos nueve), y 1,9 segundos menos. No es
# que small sea lento: es que en Docker no hay Metal. Los contenedores en
# macOS corren dentro de una VM Linux que no ve la GPU, y whisper.cpp acaba
# codificando en CPU. El coste apenas depende del largo del audio porque es
# arranque del codificador, no proceso.
#
# BAJAR A `base` NO SALE A CUENTA. Ahorra otros 176 ms y casi dobla los
# errores: «Vuelve atrás» -> «Juelve atrás», «Espera» -> «Espira», «Ponme un
# temporizador» -> «Pomeo temporizador». Con frases de una palabra -- que son
# justo las de interrumpir -- eso importa: interrupcion.clasificar() casa la
# frase entera y una palabra mal la tira al LLM. Queda como opción para
# máquinas más justas:
#
#     ./scripts/whisper-mac.sh base
#
# Y CÓMO SE USA: se le apunta al puente con --whisper-url. Sin esa opción todo
# sigue yendo por voz-api como siempre, así que esto no rompe nada si no está
# levantado.
#
#     ./scripts/whisper-mac.sh &
#     pkgs/vibevoice/.venv/bin/python scripts/asistente_web.py \
#         --whisper-url http://127.0.0.1:8083
#
# El botón de micrófono de la página SIGUE yendo por voz-api: manda lo que
# grabe MediaRecorder (webm/opus, mp4/aac) y eso necesita el ffmpeg de
# voz-api. La escucha continua construye WAV s16 ella misma y whisper.cpp lo
# lee solo -- se comprobó con el mismo audio a 16, 22 y 48 kHz: transcripción
# idéntica --, así que ahí no hace falta convertir nada.
set -euo pipefail

MODELO="${1:-small}"
PUERTO="${WHISPER_NATIVO_PUERTO:-8083}"
raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FICHERO="$raiz/docker/modelos/ggml-$MODELO.bin"

if ! command -v whisper-server >/dev/null; then
  echo "falta whisper.cpp nativo. En un Mac:" >&2
  echo "    brew install whisper-cpp" >&2
  echo "El bottle de Homebrew ya trae Metal embebido (MTL: EMBED_LIBRARY=1)." >&2
  exit 1
fi

if [[ ! -f "$FICHERO" ]]; then
  echo "falta el modelo $FICHERO; bájalo con:" >&2
  echo "    cd docker && ./bajar-modelo.sh $MODELO" >&2
  exit 1
fi

# 4 hilos y no 6: en el Mac el codificador va por Metal y los hilos solo
# atienden al decodificador, que es barato. Subirlos no cambia nada medible.
echo "==> whisper $MODELO nativo (Metal) en http://127.0.0.1:$PUERTO/inference"
exec whisper-server \
  --model "$FICHERO" \
  --language es \
  --threads 4 \
  --host 127.0.0.1 \
  --port "$PUERTO"
