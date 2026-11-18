#!/usr/bin/env bash
# Download MiVOLO v2 and the face + person detector it needs, from HuggingFace at pinned
# commits, and check their SHA-256. Run at image build time so that a container start does
# not need the network. The revisions are the ones age_gender.py reports in /health.
#
# Licences: the MiVOLO v2 weights are CC-BY-NC-4.0 (non-commercial use only); the
# detector weights are AGPL-3.0. See the README.
set -euo pipefail

MODELS_DIR="${MIVOLO_MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models/mivolo}"
HF="https://huggingface.co"

MIVOLO_REV="e8bdd8aec21ae468cbfe73b41f52550105ad8fbf"
MIVOLO="$HF/iitolstykh/mivolo_v2/resolve/$MIVOLO_REV"
DETECTOR_REV="b3d071aedccd46a3b2d4b40609da6880a815f395"
DETECTOR="$HF/iitolstykh/YOLO-Face-Person-Detector/resolve/$DETECTOR_REV"

mkdir -p "$MODELS_DIR"

fetch() {
    local url="$1" dest="$2" want="$3"
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        echo "ok (cached): $(basename "$dest")"
        return
    fi
    echo "downloading: $(basename "$dest")"
    # HuggingFace redirects LFS files to a CDN that has dropped connections mid-build
    # before; retry instead of failing the whole image on the first hiccup.
    local attempt
    for attempt in 1 2 3 4 5; do
        if curl -fsSL --retry 3 --retry-delay 5 -o "$dest.tmp" "$url"; then
            break
        fi
        echo "download failed (attempt $attempt): $url" >&2
        rm -f "$dest.tmp"
        if [ "$attempt" = 5 ]; then
            exit 1
        fi
        sleep 15
    done
    local got
    got="$(sha256sum "$dest.tmp" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$dest.tmp"
        echo "ERROR: SHA256 mismatch for $url" >&2
        echo "  expected: $want" >&2
        echo "  got:      $got" >&2
        exit 1
    fi
    mv "$dest.tmp" "$dest"
    echo "ok: $(basename "$dest")"
}

fetch "$MIVOLO/model.safetensors" "$MODELS_DIR/mivolo_v2.safetensors" \
    "96efb47051c038ebeec74b73b4253c5fd000433e5afcab7deee0bd8f3fa7bf18"
fetch "$MIVOLO/config.json" "$MODELS_DIR/mivolo_v2_config.json" \
    "aa273722873361e4e77b0cea20d02311eb5bf39e521d8a09c66a5365d4ec4e9c"
fetch "$DETECTOR/yolov8x_person_face.pt" "$MODELS_DIR/yolov8x_person_face.pt" \
    "2620f45609a65f909eb876bd7401308b5a8f3843ad5a03cb7416066a3e492989"
