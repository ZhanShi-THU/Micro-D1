"""
Prepare VQAv2 into the project's caption-style JSONL format.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VQAv2 JSONL manifest")
    parser.add_argument(
        "--questions-json",
        type=str,
        required=True,
        help="Path to the VQAv2 questions JSON file.",
    )
    parser.add_argument(
        "--annotations-json",
        type=str,
        required=True,
        help="Path to the VQAv2 annotations JSON file.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing COCO images for the corresponding split.",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="train2014",
        help="COCO split token used in image filenames, e.g. train2014 or val2014.",
    )
    return parser.parse_args()


def choose_target_answer(annotation: Dict[str, Any]) -> str | None:
    multiple_choice = str(annotation.get("multiple_choice_answer", "")).strip()
    if multiple_choice:
        return multiple_choice

    answers = annotation.get("answers", [])
    normalized = [str(answer.get("answer", "")).strip() for answer in answers]
    normalized = [answer for answer in normalized if answer]
    if not normalized:
        return None
    return Counter(normalized).most_common(1)[0][0]


def build_image_filename(image_id: int, coco_split: str) -> str:
    return f"COCO_{coco_split}_{image_id:012d}.jpg"


def convert_to_jsonl(
    questions_json: Path,
    annotations_json: Path,
    images_dir: Path,
    output_manifest: Path,
    coco_split: str,
) -> None:
    with open(questions_json, "r", encoding="utf-8") as handle:
        questions_payload = json.load(handle)
    with open(annotations_json, "r", encoding="utf-8") as handle:
        annotations_payload = json.load(handle)

    questions = questions_payload.get("questions", [])
    annotations = {
        int(annotation["question_id"]): annotation
        for annotation in annotations_payload.get("annotations", [])
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_annotation = 0
    skipped_missing_answer = 0
    skipped_missing_image = 0

    with open(output_manifest, "w", encoding="utf-8") as handle:
        for index, question in enumerate(questions):
            if index > 0 and index % 50000 == 0:
                print(f"processed {index}/{len(questions)} questions...")

            question_id = int(question["question_id"])
            annotation = annotations.get(question_id)
            if annotation is None:
                skipped_missing_annotation += 1
                continue

            target_text = choose_target_answer(annotation)
            if not target_text:
                skipped_missing_answer += 1
                continue

            image_id = int(question["image_id"])
            image_path = (images_dir / build_image_filename(image_id, coco_split)).resolve()
            if not image_path.exists():
                skipped_missing_image += 1
                continue

            output_record = {
                "image": str(image_path),
                "text": str(question["question"]).strip(),
                "target_text": target_text,
                "metadata": {
                    "source_dataset": "vqav2",
                    "question_id": question_id,
                    "image_id": image_id,
                    "coco_split": coco_split,
                },
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} records to {output_manifest}")
    print(f"Skipped {skipped_missing_annotation} records without annotations")
    print(f"Skipped {skipped_missing_answer} records without a usable answer")
    print(f"Skipped {skipped_missing_image} records without a usable image")


def main() -> None:
    args = parse_args()
    convert_to_jsonl(
        questions_json=Path(args.questions_json),
        annotations_json=Path(args.annotations_json),
        images_dir=Path(args.images_dir),
        output_manifest=Path(args.output_manifest),
        coco_split=args.coco_split,
    )


if __name__ == "__main__":
    main()
