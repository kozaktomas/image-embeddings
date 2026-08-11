"""Measure OCR throughput and print the recognised text for manual review.

Usage (on the GPU machine):
    OCR_USE_CUDA=1 ./venv/bin/python scripts/bench_ocr.py
    OCR_USE_CUDA=0 ./venv/bin/python scripts/bench_ocr.py
"""

import logging
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr  # noqa: E402

REPEATS = 3
PHOTOS_DIR = Path(__file__).resolve().parent.parent / "testdata" / "photos"
SYNTHETIC_SIZES = [(1200, 900), (2400, 1800), (4000, 3000)]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def synthetic_images():
    for width, height in SYNTHETIC_SIZES:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, height // 12)
        draw.text((width // 20, height // 3), "VESELICE 1978", fill="black", font=font)
        yield f"synthetic-{width}x{height}", image


def main():
    logging.disable(logging.WARNING)
    ocr.load_engine()
    provider = ocr.engine_info()["provider"]
    print(f"provider: {provider}\n")

    items = [(path.name, Image.open(path)) for path in sorted(PHOTOS_DIR.glob("*.jpg"))]
    items.extend(synthetic_images())

    all_times = []
    for name, image in items:
        times = []
        result = None
        for _ in range(REPEATS):
            started = time.perf_counter()
            result = ocr.extract_text(image)
            times.append(time.perf_counter() - started)
        all_times.extend(times)

        preview = result["text"].replace("\n", " | ")[:110]
        print(
            f"{name:45s} {image.size[0]:5d}x{image.size[1]:<5d} "
            f"median {statistics.median(times):6.3f}s  blocks {result['blocks_count']:3d}"
        )
        print(f"    {preview}")

    ordered = sorted(all_times)
    print(f"\ntotal {len(all_times)} runs")
    print(f"median      {statistics.median(all_times):.3f} s/photo")
    print(f"p95         {ordered[int(len(ordered) * 0.95) - 1]:.3f} s/photo")
    print(f"throughput  {1 / statistics.median(all_times):.2f} photos/s")


if __name__ == "__main__":
    main()
