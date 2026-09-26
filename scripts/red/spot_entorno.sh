#!/usr/bin/env bash
# Entorno de la campana de la red por dentro en una maquina de CPU (Ubuntu 22.04 en spot, sin CUDA).
# Corre DENTRO de la maquina, despues de `spot_ec2.sh subir` y `spot_ec2.sh credenciales`:
#
#   VOCES_S3=s3://<bucket>/voces RED_S3=s3://<bucket>/<prefijo> bash ~/mejora/scripts/red/spot_entorno.sh
#
#   VOCES_S3  el prefijo voces/ del bucket de dobla (refs reales de control para ECAPA). Opcional.
#   RED_S3    el prefijo de la campana: de ahi salen patron/ (el clip de paridad del Mac) y voces_pt/ (prefijos
#             .pt propios, p. ej. clones con consentimiento), y ahi deja entorno/<instancia>.log. Opcional.
#   Ninguna ruta de bucket va en el repo: se pasan por el entorno.
#
# Lo mismo que scripts/lora/gpu_entorno.sh (VibeVoice al commit de produccion, transformers 4.57.6, modelo y
# codificador comunitario de Hugging Face) pero con torch de CPU: se usa el uv.lock de pkgs/vibevoice, que
# ya resuelve torch 2.13.0+cpu desde download.pytorch.org/whl/cpu, y se anaden los jueces de CPU.
#
# La trampa del bf16 (docs/ec2-y-coste.md §4): Zen 5 tiene AVX512_BF16. OpenVINO pasaba a bf16 solo; torch no
# deberia, pero se fuerza igual: sitecustomize pone float32_matmul_precision("highest") y la precision ieee de
# oneDNN en cada proceso del entorno, se mide el error de un matmul contra f64 (bf16 daria ~1e-3; f32 ~1e-6) y
# la paridad frente al clip del Mac la hace campana.sh (paso paridad).
#
# Al final: probar_mecanica.py --modelo con los pesos reales tiene que dar las ocho pruebas "pasa".
set -euo pipefail
RAIZ="$HOME/mejora"
CACHE="$HOME/.cache/vibevoice-nix"
VOCES="$HOME/voces"
COMMIT_VV=94da20d98b2fa7688e9cbfaf7692ddb4954f7600
VENV="$RAIZ/pkgs/vibevoice/.venv"
PY="$VENV/bin/python"
ID=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" |
     xargs -I{} curl -s -m 2 -H "X-aws-ec2-metadata-token: {}" http://169.254.169.254/latest/meta-data/instance-id || hostname)
LOG="$HOME/entorno-$ID.log"
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
echo "[entorno] $ID $(date -u +%FT%TZ)"

# 1. sistema: awscli para el sync, espeak-ng y ffmpeg como en gpu_entorno.sh, uv para el lock
sudo apt-get -qq update >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y espeak-ng ffmpeg unzip libsndfile1 >/dev/null
if ! command -v aws >/dev/null; then
  curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  unzip -qo /tmp/awscli.zip -d /tmp && sudo /tmp/aws/install >/dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
export PATH="$HOME/.local/bin:$PATH"

# 2. python: el lock de produccion (torch 2.13.0+cpu, transformers 4.57.6, VibeVoice al commit fijado) y los jueces
cd "$RAIZ/pkgs/vibevoice"
uv python install -q 3.12
uv sync -q --frozen --no-install-project --python 3.12
uv pip install -q --python "$PY" scikit-learn librosa soundfile faster-whisper "speechbrain==1.1.1" num2words phonemizer
"$PY" - <<'FIN_PY'
import torch, transformers, vibevoice, sklearn, librosa, faster_whisper
print(f"[entorno] torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}, "
      f"sklearn {sklearn.__version__}, librosa {librosa.__version__}, faster-whisper {faster_whisper.__version__}")
assert "+cpu" in torch.__version__, "torch no es la rueda de CPU"
assert transformers.__version__ == "4.57.6", "transformers no es el de produccion"
FIN_PY

# 3. f32 estricto en todo proceso de este entorno (sin tocar los .py de scripts/red)
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cat > "$SITE/sitecustomize.py" <<'FIN_PY'
# spot_entorno.sh: f32 de verdad en CPUs con AVX512_BF16/AMX (docs/ec2-y-coste.md §4). RED_F32=0 lo desactiva.
import os
if os.environ.get("RED_F32", "1") == "1":
    try:
        import torch
        torch.set_float32_matmul_precision("highest")
        try:   # torch >= 2.9: la precision de oneDNN en matmul, aparte
            torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        except AttributeError:
            pass
    except ImportError:
        pass
FIN_PY
cat > "$HOME/red.env" <<EOF
export PATH="$VENV/bin:\$HOME/.local/bin:\$PATH"
export ONEDNN_DEFAULT_FPMATH_MODE=STRICT
export RED_F32=1
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)}
export HF_HUB_DISABLE_PROGRESS_BARS=1
EOF
source "$HOME/red.env"
echo "[entorno] CPU: $(lscpu | sed -n 's/^Model name: *//p'); nucleos fisicos $OMP_NUM_THREADS; banderas: $(grep -o -w 'avx512_bf16\|amx_bf16\|avx512f\|avx2' /proc/cpuinfo | sort -u | tr '\n' ' ')"
"$PY" - <<'FIN_PY'
import torch
torch.manual_seed(0)
a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
err = ((a @ b).double() - a.double() @ b.double()).abs().max().item() / (a.double() @ b.double()).abs().max().item()
lin = torch.nn.Linear(896, 896)
x = torch.randn(64, 896)
err_lin = ((lin(x)).double() - (x.double() @ lin.weight.double().T + lin.bias.double())).abs().max().item()
print(f"[entorno] f32: precision {torch.get_float32_matmul_precision()}, capacidad {torch.backends.cpu.get_cpu_capability()}, "
      f"error relativo matmul {err:.1e}, error absoluto lineal {err_lin:.1e}")
assert torch.get_float32_matmul_precision() == "highest"
assert err < 1e-5 and err_lin < 1e-4, "el matmul f32 no es f32: ¿bf16 silencioso?"
FIN_PY

# 4. modelo y codificador comunitario, como gpu_entorno.sh; voces oficiales del repo de VibeVoice al commit fijado
mkdir -p "$CACHE/modelo" "$VOCES"
"$PY" - "$RAIZ/scripts" "$CACHE" <<'FIN_PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
scripts, cache = Path(sys.argv[1]), Path(sys.argv[2])
dest = cache / "modelo"
if not (dest / "model.safetensors").exists():
    d = Path(snapshot_download("microsoft/VibeVoice-Realtime-0.5B"))
    for f in ("config.json", "model.safetensors", "preprocessor_config.json"):
        if (d / f).exists():
            shutil.copy(d / f, dest / f)
print("[entorno] modelo:", sorted(p.name for p in dest.iterdir()))
sys.path.insert(0, str(scripts))
from auditar_encoder import encoder_comunitario
print("[entorno] codificador:", len(encoder_comunitario(cache)), "tensores")
FIN_PY
for v in sp-Spk1_man sp-Spk0_woman; do
  [[ -s "$VOCES/$v.pt" ]] || curl -sSfL -o "$VOCES/$v.pt" \
    "https://github.com/microsoft/VibeVoice/raw/$COMMIT_VV/demo/voices/streaming_model/$v.pt"
done

# 5. datos de S3: voces propias (.pt), refs reales de control y el clip patron
if [[ -n "${RED_S3:-}" ]]; then
  aws s3 sync --only-show-errors "$RED_S3/voces_pt/" "$VOCES/"
  aws s3 sync --only-show-errors "$RED_S3/patron/" "$HOME/patron/"
fi
if [[ -n "${VOCES_S3:-}" ]]; then
  aws s3 sync --only-show-errors "${VOCES_S3%/}/" "$HOME/voces_refs/" --exclude "*" --include "*/refs/*" --include "identidades.json"
fi
echo "[entorno] voces: $(cd "$VOCES" && ls *.pt | tr '\n' ' ')· patron: $(ls "$HOME/patron" 2>/dev/null | tr '\n' ' ')· refs: $(ls "$HOME/voces_refs" 2>/dev/null | grep -vc json || true) identidades"

# 6. la mecanica con los pesos reales: las ocho pruebas
cd "$RAIZ"
"$PY" scripts/red/probar_mecanica.py --voz "$VOCES/sp-Spk1_man.pt" --modelo "$CACHE/modelo" | tee "$HOME/mecanica-$ID.log"
RES=$(grep '^RESUMEN' "$HOME/mecanica-$ID.log" || true)
PASAN=$(grep -o "'pasa'" <<<"$RES" | wc -l)
if [[ "$PASAN" -ne 8 || "$RES" == *FALLA* ]]; then
  echo "[entorno] MECANICA NO PASA ($PASAN/8): $RES"
  [[ -n "${RED_S3:-}" ]] && aws s3 cp --only-show-errors "$LOG" "$RED_S3/entorno/$ID.log" || true
  exit 5
fi
echo "[entorno] mecanica 8/8 · $(( $(date +%s) - t0 )) s en montar y probar"
touch "$HOME/entorno.ok"
[[ -n "${RED_S3:-}" ]] && aws s3 cp --only-show-errors "$LOG" "$RED_S3/entorno/$ID.log" || true
echo ENTORNO_LISTO
