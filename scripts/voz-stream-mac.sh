#!/usr/bin/env bash
# VibeVoice en un Mac con Apple Silicon, NATIVO y sobre la GPU (Metal).
#
# Por qué no va por Docker: los contenedores en macOS corren dentro de una VM
# Linux que NO tiene acceso a Metal. Da igual lo que se configure -- desde un
# contenedor, torch.backends.mps.is_available() siempre da False. La GPU del
# Mac solo se alcanza desde un proceso nativo, y de ahí este script.
#
# El resto del stack (voz-api y whisper) sí va bien por Docker, y se compila
# nativo para ARM. Lo normal es combinarlos:
#
#     cd docker && docker compose up -d      # Piper + whisper en el 8080
#     ./scripts/voz-stream-mac.sh            # VibeVoice en el 8082
#
# La consola del 8080 encuentra el 8082 sola.
set -euo pipefail

raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${VIBEVOICE_CACHE:-$HOME/.cache/vibevoice-nix}"

command -v uv >/dev/null || { echo "hace falta uv: brew install uv"; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || echo "[aviso] esto es para macOS; en Linux usa docker compose"

# ------------------------------------------------------------------ deps --
# El lock resuelve DOS torch segun plataforma: en darwin cae la rueda normal,
# que trae MPS de serie. No hay que forzar nada.
echo "==> dependencias"
cd "$raiz/pkgs/vibevoice"
uv sync --frozen --no-dev

export HF_HOME="$cache/hf"
voces="$cache/voces"
preparado="$cache/modelo"

# ----------------------------------------------------------------- modelo --
echo "==> modelo y tokenizador (~2 GB la primera vez)"
uv run python - "$preparado" <<'PY'
import json, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

modelo = Path(snapshot_download("microsoft/VibeVoice-Realtime-0.5B"))
# El repo del modelo no trae tokenizador: el procesador cae a Qwen2.5-1.5B.
qwen = Path(snapshot_download("Qwen/Qwen2.5-1.5B",
            allow_patterns=["tokenizer*", "vocab.json", "merges.txt"]))

# Se prepara una copia del modelo con el tokenizador apuntado a una ruta
# LOCAL. Sin esto el procesador pide "Qwen/Qwen2.5-1.5B" al hub por su
# identificador, y con HF_HUB_OFFLINE=1 la resolucion desde cache no le
# devuelve vocab.json: revienta con
#
#   TypeError: expected str, bytes or os.PathLike object, not NoneType
#
# Es lo mismo que hace el paquete de Nix, que parchea este campo para apuntar
# al tokenizador del store. Ojo: el procesador exige que la ruta contenga
# "qwen", que es como decide el tipo de tokenizador.
destino = Path(sys.argv[1])
destino.mkdir(parents=True, exist_ok=True)
if not (destino / "config.json").exists():
    shutil.copy2(modelo / "config.json", destino / "config.json")

# Los pesos se copian PODADOS: fuera model.tts_language_model.embed_tokens
# (260 MiB que nadie lee -- compartir_embeddings_muertos() en voz_stream.py
# la sustituye al cargar por la tabla del language_model, de forma identica).
# Si hay una copia anterior sin podar, se rehace desde la cache del hub.
MUERTA = "model.tts_language_model.embed_tokens.weight"
pesos = destino / "model.safetensors"
hace_falta = True
if pesos.exists():
    with safe_open(str(pesos), framework="pt") as f:
        hace_falta = MUERTA in f.keys()
if hace_falta:
    with safe_open(str(modelo / "model.safetensors"), framework="pt") as f:
        claves = [k for k in f.keys() if k != MUERTA]
        assert len(claves) == len(f.keys()) - 1, \
            "la tabla muerta no esta en el checkpoint: revisar la poda"
        tensores = {k: f.get_tensor(k) for k in claves}
        meta = f.metadata() or {"format": "pt"}
    # A un temporal y luego rename: si esto muere a medias no deja unos
    # pesos corruptos con el nombre bueno.
    tmp = pesos.with_name("model.safetensors.tmp")
    save_file(tensores, str(tmp), metadata=meta)
    tmp.replace(pesos)
    print(f"pesos podados: {pesos.stat().st_size} bytes")

cfg = json.loads((modelo / "preprocessor_config.json").read_text())
cfg["language_model_pretrained_name"] = str(qwen)
(destino / "preprocessor_config.json").write_text(json.dumps(cfg, indent=2))
print(f"modelo preparado en {destino}")
PY

# ------------------------------------------------------------------ voces --
# Los .pt no van en el paquete de Python: el pyproject de VibeVoice empaqueta
# el codigo, no demo/.
#
# Solo las sp-* (9,4 MB frente a 96): son las unicas en espanol y las que usa
# todo el stack. Si quieres las 25, borra "$voces" y cambia el patron.
if [[ ! -d "$voces" ]] || [[ -z "$(ls -A "$voces" 2>/dev/null)" ]]; then
  echo "==> voces"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  git clone --quiet --filter=blob:none --sparse \
    https://github.com/microsoft/VibeVoice.git "$tmp/vv"
  git -C "$tmp/vv" checkout --quiet 94da20d98b2fa7688e9cbfaf7692ddb4954f7600
  git -C "$tmp/vv" sparse-checkout set demo/voices/streaming_model
  mkdir -p "$voces"
  cp "$tmp/vv"/demo/voices/streaming_model/sp-*.pt "$voces/"
fi

# ---------------------------------------------------------------- arranque --
[[ -f "$preparado/model.safetensors" ]] || { echo "falta el modelo en $preparado"; exit 1; }
export VIBEVOICE_MODELO="$preparado"
export VIBEVOICE_VOCES="$voces"
export VIBEVOICE_PASOS="${VIBEVOICE_PASOS:-6}"
export VIBEVOICE_VOZ="${VIBEVOICE_VOZ:-sp-Spk1_man}"
export VOZ_STREAM_PUERTO="${VOZ_STREAM_PUERTO:-8082}"
export VOZ_STREAM_HOST="${VOZ_STREAM_HOST:-127.0.0.1}"
export HF_HUB_OFFLINE=1
# "auto" ya elegiria mps; se puede forzar cpu para comparar:
#     VIBEVOICE_DISPOSITIVO=cpu ./scripts/voz-stream-mac.sh
export VIBEVOICE_DISPOSITIVO="${VIBEVOICE_DISPOSITIVO:-auto}"

# El token lo comparte con el resto del stack si existe el .env de docker.
if [[ -z "${VOZ_TOKEN:-}" && -f "$raiz/docker/.env" ]]; then
  VOZ_TOKEN="$(grep -E '^VOZ_TOKEN=' "$raiz/docker/.env" | cut -d= -f2- | tr -d '"'"'"'' || true)"
  export VOZ_TOKEN
fi

echo "==> arrancando en http://$VOZ_STREAM_HOST:$VOZ_STREAM_PUERTO"
exec uv run python "$raiz/pkgs/vibevoice-cli/voz_stream.py"
