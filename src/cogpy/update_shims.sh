#!/usr/bin/env bash
# Run from: code/lib/cogpy/src/cogpy
set -euo pipefail

if [[ ! -d "./core" || ! -f "./__init__.py" ]]; then
  echo "Run this inside src/cogpy (must contain ./core and ./__init__.py)" >&2
  exit 1
fi

# auto-detect subpackages in ./core that have __init__.py
mapfile -t SUBPKGS < <(find ./core -mindepth 1 -maxdepth 1 -type d \
  -not -name "__pycache__" -exec test -f "{}/__init__.py" \; -print |
  xargs -I{} basename "{}" | sort)

echo "Updating shims for: ${SUBPKGS[*]}"

for name in "${SUBPKGS[@]}"; do
  SHIM_DIR="./$name"
  SHIM_INIT="$SHIM_DIR/__init__.py"
  mkdir -p "$SHIM_DIR"

  [[ -f "$SHIM_INIT" && ! -f "$SHIM_INIT.bak" ]] && cp -p "$SHIM_INIT" "$SHIM_INIT.bak"

  cat > "$SHIM_INIT" <<PY
# Auto-generated shim: exposes cogpy.core.${name} as cogpy.${name}
from cogpy.core import ${name} as _impl
from cogpy.core.${name} import *
__all__ = getattr(_impl, "__all__", [])
PY

  echo "✓ wrote $SHIM_INIT"
done

echo "Done."
