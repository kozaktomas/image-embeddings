#!/usr/bin/env bash
# Stáhne OCR modely (PP-OCRv5) z HuggingFace a připraví slovník znaků.
# ModelScope, odkud RapidOCR stahuje ve výchozím stavu, je na této síti
# blokovaný resolverem DNS4EU Protective — proto oficiální PaddlePaddle repa na HF.
set -euo pipefail

MODELS_DIR="${OCR_MODELS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
HF="https://huggingface.co"

DET_URL="$HF/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx"
DET_SHA="a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d"
REC_URL="$HF/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"
REC_SHA="7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498"
YML_URL="$HF/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"
YML_SHA="0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84"

mkdir -p "$MODELS_DIR"

fetch() {
    local url="$1" dest="$2" want="$3"
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        echo "ok (cached): $(basename "$dest")"
        return
    fi
    echo "stahuji: $(basename "$dest")"
    curl -fsSL -o "$dest.tmp" "$url"
    local got
    got="$(sha256sum "$dest.tmp" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$dest.tmp"
        echo "CHYBA: SHA256 nesouhlasi pro $url" >&2
        echo "  ocekavano: $want" >&2
        echo "  ziskano:   $got" >&2
        exit 1
    fi
    mv "$dest.tmp" "$dest"
    echo "ok: $(basename "$dest")"
}

fetch "$DET_URL" "$MODELS_DIR/PP-OCRv5_mobile_det.onnx" "$DET_SHA"
fetch "$REC_URL" "$MODELS_DIR/latin_PP-OCRv5_mobile_rec.onnx" "$REC_SHA"
fetch "$YML_URL" "$MODELS_DIR/latin_rec_inference.yml" "$YML_SHA"

python3 - "$MODELS_DIR" <<'PY'
import sys, pathlib, yaml

models_dir = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load((models_dir / "latin_rec_inference.yml").read_text(encoding="utf-8"))
chars = cfg["PostProcess"]["character_dict"]
out = models_dir / "latin_dict.txt"
out.write_text("\n".join(chars) + "\n", encoding="utf-8")
print(f"slovnik: {len(chars)} znaku -> {out}")
missing = [c for c in "ěščřžýáíéůúňťď" if c not in chars]
if missing:
    raise SystemExit(f"CHYBA: ve slovniku chybi ceska diakritika: {missing}")
print("ceska diakritika ve slovniku OK")
PY
