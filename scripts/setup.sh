#!/usr/bin/env bash
# One-time setup: Python environment (.venv via uv) + FoundationStereo in third_party/.
# Safe to re-run: every step is skipped when already done.
#
#   bash scripts/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FS_DIR="$REPO/third_party/FoundationStereo"
FS_URL="https://github.com/NVlabs/FoundationStereo"
FS_COMMIT="6e8806816b533e4d13ddbb95ffa907b797060a62"   # pinned, bump deliberately
FS_CKPT_DRIVE="https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
FS_CKPT_DIR="$FS_DIR/pretrained_models/23-51-11"

cd "$REPO"

# --- 1. prerequisites -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. Post-processing (FoundationStereo) needs an NVIDIA GPU + driver."
fi

# --- 2. Python environment --------------------------------------------------------------
echo "==> uv sync (creates .venv from uv.lock)"
uv sync

# --- 3. FoundationStereo code -----------------------------------------------------------
if [ ! -d "$FS_DIR/.git" ]; then
    echo "==> cloning FoundationStereo into third_party/"
    mkdir -p "$REPO/third_party"
    git clone "$FS_URL" "$FS_DIR"
fi
if [ "$(git -C "$FS_DIR" rev-parse HEAD)" != "$FS_COMMIT" ]; then
    echo "==> checking out pinned FoundationStereo commit ${FS_COMMIT:0:8}"
    git -C "$FS_DIR" fetch --quiet origin
    git -C "$FS_DIR" checkout --quiet -- .   # drop the patch below so the checkout can't conflict
    git -C "$FS_DIR" checkout --quiet "$FS_COMMIT"
fi
# Patch: torch.cuda.amp.autocast is deprecated (FutureWarning on every call) -> torch.amp.autocast("cuda", ...).
# Same behaviour. Upstream master still uses the old call. Idempotent: patched lines no longer match.
sed -i 's/^\(\s*\)autocast = torch\.cuda\.amp\.autocast$/\1autocast = lambda **kw: torch.amp.autocast("cuda", **kw)/' \
    "$FS_DIR/core/foundation_stereo.py"
sed -i 's/torch\.cuda\.amp\.autocast(/torch.amp.autocast("cuda", /g' "$FS_DIR/core/geometry.py" "$FS_DIR/core/submodule.py"

# --- 4. FoundationStereo checkpoint (Google Drive) ---------------------------------------
if [ ! -f "$FS_CKPT_DIR/model_best_bp2.pth" ] || [ ! -f "$FS_CKPT_DIR/cfg.yaml" ]; then
    echo "==> downloading the FoundationStereo checkpoints (Google Drive, ~several hundred MB)"
    mkdir -p "$FS_DIR/pretrained_models"
    if ! uv run --with gdown gdown --folder "$FS_CKPT_DRIVE" -O "$FS_DIR/pretrained_models"; then
        echo "Download failed (Google Drive sometimes rate-limits). Download the folder by hand from"
        echo "    $FS_CKPT_DRIVE"
        echo "and put its 23-51-11/ folder at $FS_CKPT_DIR"
        exit 1
    fi
    # gdown sometimes nests the folder: pretrained_models/pretrained_models/23-51-11
    if [ -d "$FS_DIR/pretrained_models/pretrained_models" ]; then
        mv "$FS_DIR/pretrained_models/pretrained_models/"* "$FS_DIR/pretrained_models/"
        rmdir "$FS_DIR/pretrained_models/pretrained_models"
    fi
    if [ ! -f "$FS_CKPT_DIR/model_best_bp2.pth" ] || [ ! -f "$FS_CKPT_DIR/cfg.yaml" ]; then
        echo "Checkpoint not where expected: need $FS_CKPT_DIR/{model_best_bp2.pth,cfg.yaml}"
        ls -R "$FS_DIR/pretrained_models" | head -20
        exit 1
    fi
fi

# --- 5. smoke test ----------------------------------------------------------------------
echo "==> smoke test"
uv run python - "$FS_DIR" <<'EOF'
import sys
import torch
print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
sys.path.insert(0, sys.argv[1])
from core.foundation_stereo import FoundationStereo  # noqa: F401
import projectaria_tools  # noqa: F401
print("  FoundationStereo + projectaria_tools import OK")
EOF
echo "Setup done."
