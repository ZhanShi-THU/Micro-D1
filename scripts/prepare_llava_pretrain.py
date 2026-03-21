"""
Preprocess LLaVA-Pretrain dataset for adapter warm-up training.

Converts the raw JSON format to project-compatible JSONL format:
  {
    "image": "path/to/image.jpg",
    "text": "human prompt",
    "target_text": "assistant response"
  }

Also extracts images if not already extracted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any


FALLBACK_PROMPT = "Describe the image in detail."
IMAGE_PLACEHOLDER_RE = re.compile(r"<image>", flags=re.IGNORECASE)


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

    print(f"Extracting images to {images_dir}...")
    os.makedirs(images_dir, exist_ok=True)

    if not images_zip.exists():
        raise FileNotFoundError(f"images.zip not found at {images_zip}")

    print(f"Extracting images from {images_zip}...")
    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(images_dir)
    print(f"Extracted to {images_dir}")
    return images_dir


def clean_human_prompt(prompt: str) -> str:
    prompt = IMAGE_PLACEHOLDER_RE.sub(" ", prompt)
    prompt = " ".join(prompt.split())
    return prompt.strip()


def extract_prompt_and_target(record: dict[str, Any]) -> tuple[str, str] | None:
    conversations = record.get("conversations", [])
    prompt_text: str | None = None
    target_text: str | None = None

    for conv in conversations:
        speaker = str(conv.get("from", "")).strip().lower()
        value = str(conv.get("value", "")).strip()

        if speaker == "human" and prompt_text is None:
            prompt_text = clean_human_prompt(value)
        elif speaker == "gpt" and target_text is None and value:
            target_text = value

        if prompt_text is not None and target_text is not None:
            break

    if not target_text:
        return None
    if not prompt_text:
        prompt_text = FALLBACK_PROMPT
    return prompt_text, target_text


def convert_to_jsonl(
    json_path: Path,
    images_dir: Path,
    output_path: Path,
) -> None:
    """Convert LLaVA-Pretrain JSON to prompt-conditioned JSONL format."""
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Converting {len(data)} records to JSONL...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    skipped_missing_target = 0
    skipped_missing_image = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(data):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(data)}...")

            prompt_and_target = extract_prompt_and_target(record)
            if prompt_and_target is None:
                skipped_missing_target += 1
                continue
            prompt_text, target_text = prompt_and_target

            image_rel = str(record.get("image", "")).strip()
            if not image_rel:
                skipped_missing_image += 1
                continue

            image_path = (images_dir / image_rel).resolve()
            if not image_path.exists():
                skipped_missing_image += 1
                continue

            out_record = {
                "image": str(image_path),
                "text": prompt_text,
                "target_text": target_text,
                "metadata": {
                    "source_dataset": "llava_pretrain",
                    "original_image_rel": image_rel,
                    "conversation_count": len(record.get("conversations", [])),
                },
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written_count += 1

    print(f"Saved {written_count} records to {output_path}")
    print(f"Skipped {skipped_missing_target} records without a usable GPT target")
    print(f"Skipped {skipped_missing_image} records without a usable image")


def main() -> None:
    args = parse_args()
    phase1_dir = Path(args.phase1_dir)

    if args.extract_images:
        images_dir = extract_images(phase1_dir)
    else:
        images_dir = phase1_dir / "images"
        if not images_dir.exists():
            print(f"Warning: images directory not found at {images_dir}")
            print("Run with --extract_images to extract from zip.")

    json_path = phase1_dir / "blip_laion_cc_sbu_558k.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    convert_to_jsonl(json_path, images_dir, Path(args.output_manifest))


if __name__ == "__main__":
    main()
