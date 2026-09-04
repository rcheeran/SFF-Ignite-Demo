#!/usr/bin/env bash
# Fetch the AgileX Piper meshes (32 MB, MIT) from google-deepmind/mujoco_menagerie into sim/assets/agilex_piper/assets/.
# The MJCF itself (piper.xml, lightly edited) and its LICENSE are tracked; the meshes are gitignored for size.
#     bash sim/fetch_assets.sh
set -euo pipefail
dst="$(cd "$(dirname "$0")" && pwd)/assets/agilex_piper/assets"
[ -f "$dst/link6.stl" ] && { echo "meshes already present in $dst"; exit 0; }
tmp="$(mktemp -d)"
git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$tmp"
git -C "$tmp" sparse-checkout set --quiet agilex_piper/assets
mkdir -p "$dst" && cp "$tmp"/agilex_piper/assets/* "$dst"/ && rm -rf "$tmp"
echo "fetched $(ls "$dst" | wc -l) mesh files into $dst"
