"""
Prepare LLaVA-Instruct-150K into the project's caption-style JSONL format.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FALLBACK_PROMPT = "Describe the image in detail."
IMAGE_PLACEHOLDER_RE = re.compile(r"<image>", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LLaVA-Instruct-150K JSONL manifest")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to the raw LLaVA-Instruct JSON file.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing the extracted training images.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Output JSONL manifest path.",
    )
    return parser.parse_args()


def clean_human_prompt(prompt: str) -> str:
    prompt = IMAGE_PLACEHOLDER_RE.sub(" ", prompt)
    prompt = " ".join(prompt.split())
    return prompt.strip()


def extract_prompt_and_target(record: dict[str, Any]) -> tuple[str, str] | None:
    prompt_text: str | None = None
    target_text: str | None = None

    for turn in record.get("conversations", []):
        speaker = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()

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


def convert_to_jsonl(input_json: Path, images_dir: Path, output_manifest: Path) -> None:
    with open(input_json, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_target = 0
    skipped_missing_image = 0

    with open(output_manifest, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if index > 0 and index % 50000 == 0:
                print(f"processed {index}/{len(records)} records...")

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

            output_record = {
                "image": str(image_path),
                "text": prompt_text,
                "target_text": target_text,
                "metadata": {
                    "source_dataset": "llava_instruct_150k",
                    "original_image_rel": image_rel,
                    "conversation_count": len(record.get("conversations", [])),
                },
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} records to {output_manifest}")
    print(f"Skipped {skipped_missing_target} records without a usable GPT target")
    print(f"Skipped {skipped_missing_image} records without a usable image")


def main() -> None:
    args = parse_args()
    convert_to_jsonl(
        input_json=Path(args.input_json),
        images_dir=Path(args.images_dir),
        output_manifest=Path(args.output_manifest),
    )


if __name__ == "__main__":
    main()
