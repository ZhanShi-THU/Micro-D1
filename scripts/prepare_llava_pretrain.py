"""
Preprocess LLaVA-Pretrain dataset for adapter warm-up training.

Converts the raw JSON format to project-compatible JSONL format:
  {"image": "path/to/image.jpg", "text": "caption text"}

Also extracts images if not already extracted.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LLaVA-Pretrain dataset")
    parser.add_argument(
        "--phase1_dir",
        type=str,
        default="/data1/LLaVA-Pretrain/phase1",
        help="Path to LLaVA-Pretrain/phase1 directory",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="/data1/LLaVA-Pretrain/phase1/pretrain.jsonl",
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--extract_images",
        action="store_true",
        help="Extract images.zip if not already extracted",
    )
    return parser.parse_args()


def extract_images(phase1_dir: Path) -> Path:
    """Extract images.zip to images directory."""
    images_zip = phase1_dir / "images.zip"
    images_dir = phase1_dir / "images"

    if images_dir.exists():
        print(f"Images already extracted at {images_dir}")
        return images_dir

    if not images_zip.exists():
        raise FileNotFoundError(f"images.zip not found at {images_zip}")

    print(f"Extracting images from {images_zip}...")
    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(phase1_dir)
    print(f"Extracted to {images_dir}")
    return images_dir


def convert_to_jsonl(
    json_path: Path,
    images_dir: Path,
    output_path: Path,
) -> None:
    """Convert LLaVA-Pretrain JSON to JSONL format."""
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Converting {len(data)} records to JSONL...")
    with open(output_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(data):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(data)}...")

            # Extract caption from conversations
            # Format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            conversations = record.get("conversations", [])
            caption = None
            for conv in conversations:
                if conv.get("from") == "gpt":
                    caption = conv.get("value", "").strip()
                    break

            if caption is None:
                continue

            # Image path is relative, e.g., "00453/004539375.jpg"
            image_rel = record.get("image", "")
            if not image_rel:
                continue

            # Build absolute image path
            image_path = images_dir / image_rel

            # Output record in project format
            out_record = {
                "image": str(image_path),
                "text": caption,
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Saved {i + 1} records to {output_path}")


def main() -> None:
    args = parse_args()
    phase1_dir = Path(args.phase1_dir)

    # Extract images if requested
    if args.extract_images:
        images_dir = extract_images(phase1_dir)
    else:
        images_dir = phase1_dir / "images"
        if not images_dir.exists():
            print(f"Warning: images directory not found at {images_dir}")
            print("Run with --extract_images to extract from zip.")

    # Convert JSON to JSONL
    json_path = phase1_dir / "blip_laion_cc_sbu_558k.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    convert_to_jsonl(json_path, images_dir, Path(args.output_manifest))


if __name__ == "__main__":
    main()
