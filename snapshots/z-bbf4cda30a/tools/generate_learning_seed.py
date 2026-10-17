#!/usr/bin/env python3
"""Create a compact BACTERIALSCOPE visual-memory seed from an app training ZIP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

from PIL import Image


GRID_SIZE = 10


def extract_feature(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB").resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR)
        pixels = list(rgb.getdata())

    luminance = [red * 0.299 + green * 0.587 + blue * 0.114 for red, green, blue in pixels]
    mean = sum(luminance) / len(luminance)
    variance = sum((value - mean) ** 2 for value in luminance) / len(luminance)
    standard_deviation = max(math.sqrt(variance), 10.0)
    feature = bytearray(
        max(0, min(255, int((((value - mean) / standard_deviation + 2.5) / 5.0) * 255.0)))
        for value in luminance
    )

    total_red = sum(pixel[0] for pixel in pixels)
    total_green = sum(pixel[1] for pixel in pixels)
    total_blue = sum(pixel[2] for pixel in pixels)
    color_total = max(total_red + total_green + total_blue, 1)
    feature.extend(
        max(0, min(255, int(channel_total / color_total * 255.0)))
        for channel_total in (total_red, total_green, total_blue)
    )
    return bytes(feature)


def epoch_from_sample(sample: str) -> int:
    match = re.search(r"_(\d{13})$", sample)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("labels_path", type=Path)
    args = parser.parse_args()

    expected_labels = [line.strip() for line in args.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples: list[dict[str, object]] = []

    with zipfile.ZipFile(args.zip_path) as archive:
        exported_labels = [line.strip() for line in archive.read("classes.txt").decode("utf-8").splitlines() if line.strip()]
        if exported_labels != expected_labels:
            raise SystemExit("Exported classes.txt does not match app bacterial_labels.txt")

        image_entries = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.startswith("images/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        metadata_names = sorted(
            name for name in archive.namelist() if name.startswith("metadata/") and name.endswith(".json")
        )
        for metadata_name in metadata_names:
            details = json.loads(archive.read(metadata_name).decode("utf-8"))
            sample = str(details.get("sample") or Path(metadata_name).stem)
            image_name = image_entries.get(sample)
            if not image_name:
                continue
            predicted = str(details.get("predicted_label") or "").strip()
            correct = str(details.get("correct_label") or "").strip()
            if not predicted or not correct:
                continue
            if correct != "No bacteria / false detection" and correct not in expected_labels:
                raise SystemExit(f"Unknown correct label in {metadata_name}: {correct}")
            feature = extract_feature(archive.read(image_name))
            examples.append(
                {
                    "sample": sample,
                    "predicted": predicted,
                    "correct": correct,
                    "feature": base64.b64encode(feature).decode("ascii"),
                    "created": int(details.get("created_epoch") or epoch_from_sample(sample)),
                }
            )

    payload = {
        "schema": 1,
        "source": args.zip_path.name,
        "class_count": len(expected_labels),
        "classes_sha256": hashlib.sha256(("\n".join(expected_labels) + "\n").encode("utf-8")).hexdigest(),
        "example_count": len(examples),
        "examples": examples,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {args.output_path}")


if __name__ == "__main__":
    main()
