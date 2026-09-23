#!/usr/bin/env bash
# Entorno de entrenamiento y evaluacion en la instancia GPU (DLAMI PyTorch, Ubuntu 22.04).
#   bash ~/mejora/scripts/lora/gpu_entorno.sh
set -euo pipefail
sudo apt-get -qq update >/dev/null && sudo apt-get -qq install -y espeak-ng ffmpeg >/dev/null
source /opt/pytorch/bin/activate
# el mismo commit de VibeVoice que produccion (pkgs/vibevoice/uv.lock), sin tocar el torch de la imagen
pip install -q --no-deps "git+https://github.com/microsoft/VibeVoice.git@94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
pip install -q "transformers==4.57.6" diffusers accelerate "datasets>=3" soundfile librosa faster-whisper \
    "speechbrain==1.1.1" phonemizer num2words sentencepiece protobuf nvidia-ml-py 2>&1 | grep -v -i "notice" | tail -3
python - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
d = Path(snapshot_download("microsoft/VibeVoice-Realtime-0.5B"))
dest = Path.home() / ".cache/vibevoice-nix/modelo"
dest.mkdir(parents=True, exist_ok=True)
for f in ("config.json", "model.safetensors", "preprocessor_config.json"):
    if (d / f).exists():
        shutil.copy(d / f, dest / f)
print("modelo:", sorted(p.name for p in dest.iterdir()))
import sys
sys.path.insert(0, str(Path.home() / "mejora/scripts"))
from auditar_encoder import encoder_comunitario
print("codificador:", len(encoder_comunitario(Path.home() / ".cache/vibevoice-nix")), "tensores")
PY
echo ENTORNO_LISTO
