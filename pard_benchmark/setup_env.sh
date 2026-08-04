#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${PARD_VENV:-"$ROOT/.venv-pard"}
BASE_PYTHON=${PARD_BASE_PYTHON:-/opt/venv/bin/python3}
if [[ ! -x "$BASE_PYTHON" ]]; then
  BASE_PYTHON=$(command -v python3)
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$VENV"
fi

# A venv created from the image's /opt/venv does not automatically inherit the
# parent venv.  Add that site-packages directory after this venv's own packages,
# so ROCm PyTorch is reused while Transformers 4.51.3 remains isolated here.
BASE_SITE=$($BASE_PYTHON -c 'import site; print(site.getsitepackages()[0])')
VENV_SITE=$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')
$VENV/bin/python - "$VENV_SITE/rocm_parent_venv.pth" "$BASE_SITE" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(sys.argv[2] + "\n", encoding="utf-8")
PY

# Keep new downloads in a user-writable cache. Reuse the existing gated Base
# snapshot without touching its old root-squashed cache or lock directory.
if [[ -n "${HF_HOME:-}" ]]; then
  mkdir -p "$HF_HOME/hub"
  LEGACY_BASE="$ROOT/../.hf_cache/hub/models--meta-llama--Llama-3.1-8B"
  LINKED_BASE="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B"
  if [[ -d "$LEGACY_BASE" && ! -e "$LINKED_BASE" ]]; then
    ln -s "$LEGACY_BASE" "$LINKED_BASE"
  fi
fi

if ! "$VENV/bin/python" -c 'import torch, transformers; assert transformers.__version__ == "4.51.3"' >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install --disable-pip-version-check \
    -r "$ROOT/requirements-pard-inference.txt"
fi

"$VENV/bin/python" - <<'PY'
import torch
import transformers

print(f"PARD environment ready: torch={torch.__version__}, transformers={transformers.__version__}")
if transformers.__version__ != "4.51.3":
    raise SystemExit("isolated environment did not activate Transformers 4.51.3")
PY
