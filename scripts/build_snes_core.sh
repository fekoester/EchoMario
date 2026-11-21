#!/bin/bash
# Builds stable-retro's snes9x core with a render/audio switch (third_party/snes9x_render_switch.patch)
# and installs it in ~/.cache/echomario/snes_core, where src/echomario/envs/smw_env.py picks it up.
# The look-ahead teacher then skips drawing the picture in its rollouts: ~1.5x more emulated frames
# per second, identical game RAM (checked by tests/test_smw.py).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("stable-retro"))')"
DEST="${ECHOMARIO_SNES_CORE:-$HOME/.cache/echomario/snes_core}"
CORE_INFO="$("$PY" -c 'import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), "cores", "snes9x.json"))')"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -c advice.detachedHead=false clone -q --depth 1 --branch "v$VERSION" https://github.com/Farama-Foundation/stable-retro.git "$WORK/stable-retro"
cd "$WORK/stable-retro"
git apply "$REPO/third_party/snes9x_render_switch.patch"
make -C cores/snes/libretro -j"${JOBS:-4}" > "$WORK/build.log" 2>&1 || { tail -20 "$WORK/build.log"; exit 1; }
mkdir -p "$DEST"
cp cores/snes/libretro/snes9x_libretro.so "$DEST/"
cp "$CORE_INFO" "$DEST/"
echo "installed $DEST/snes9x_libretro.so (stable-retro v$VERSION + render switch)"
