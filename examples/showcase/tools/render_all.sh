#!/usr/bin/env bash
# Build showcase scenes and render them (1080p60 H.264) into renders/showcase/.
#   bash examples/showcase/tools/render_all.sh s01_patagonia [s02_hurtigruten ...]
# The scene script examples/showcase/tools/<scene>.py writes <scene>/scenes/<scene>.json.
# VERSION (default _v1) is appended to the file name, so earlier renders are kept.
set -u
cd "$(dirname "$0")/../../.." || exit 1
V="${VERSION:-_v1}"
OUT=renders/showcase
mkdir -p "$OUT"
for name in "$@"; do
  python -m uv run python "examples/showcase/tools/$name.py" > /dev/null || { echo "build failed: $name"; continue; }
  res=$(python -c "import json;print(json.load(open('examples/showcase/$name/scenes/$name.json'))['properties'].get('export.resolution','1920x1080'))")
  python -m uv run python -m earthling.cli render "examples/showcase/$name" --scene "scenes/$name.json" \
    --preset h264 --resolution "$res" --out "$OUT/${name}_1080p60${V}.mp4" 2>&1 \
    | tr '\r' '\n' | grep -v "^  frame" > "$OUT/${name}${V}_render_log.txt"
  echo "$name: $(grep wrote "$OUT/${name}${V}_render_log.txt" | tail -1)"
  python -m uv run earthling qa "$OUT/${name}_1080p60${V}.mp4"
done
