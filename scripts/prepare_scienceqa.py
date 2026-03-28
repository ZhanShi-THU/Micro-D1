"""
Prepare ScienceQA parquet splits into the project's microvqa-style JSONL format.

The conversion now supports:
  - filtering to more microscopy-adjacent science subsets
  - richer reasoning-style supervision targets
  - optional lecture inclusion in the target rationale

Only image-backed multiple-choice samples are retained so the output is suitable
for visual-language training.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import build_multiple_choice_target


DEFAULT_ALLOWED_SUBJECTS = ("natural science",)
DEFAULT_ALLOWED_TOPICS = (
    "biology",
    "chemistry",
    "physics",
    "earth-science",
    "science-and-engineering-practices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ScienceQA JSONL manifests")
    parser.add_argument(
        "--train-parquet",
        type=str,
        required=True,
        help="Path to the ScienceQA training parquet split.",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        required=True,
        help="Path to the ScienceQA validation parquet split.",
    )
    parser.add_argument(
        "--test-parquet",
        type=str,
        required=True,
        help="Path to the ScienceQA test parquet split.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory where manifests and extracted images will be written.",
    )
    parser.add_argument(
        "--allowed-subjects",
        nargs="*",
        default=list(DEFAULT_ALLOWED_SUBJECTS),
        help=(
            "Allowed ScienceQA subjects. Defaults to the more relevant science subset: "
            f"{', '.join(DEFAULT_ALLOWED_SUBJECTS)}"
        ),
    )
    parser.add_argument(
        "--allowed-topics",
        nargs="*",
        default=list(DEFAULT_ALLOWED_TOPICS),
        help=(
            "Allowed ScienceQA topics. Defaults to a microscopy-friendlier subset: "
            f"{', '.join(DEFAULT_ALLOWED_TOPICS)}"
        ),
    )
    parser.add_argument(
        "--disable-default-filter",
        action="store_true",
        help="Disable subject/topic filtering and keep all image-backed MCQ samples.",
    )
    parser.add_argument(
        "--include-lecture-in-target",
        action="store_true",
        help="Append lecture text before the solution in the target rationale when present.",
    )
    return parser.parse_args()


def normalize_choices(raw_choices: Any) -> list[str]:
    if raw_choices is None:
        return []
    if hasattr(raw_choices, "tolist"):
        raw_choices = raw_choices.tolist()
    if not isinstance(raw_choices, Iterable) or isinstance(raw_choices, (str, bytes)):
        return []

    choices: list[str] = []
    for choice in raw_choices:
        text = str(choice).strip()
        if text:
            choices.append(text)
    return choices


def normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_filter_values(values: Sequence[str] | None) -> set[str]:
    normalized: set[str] = set()
    for value in values or []:
        cleaned = normalize_optional_text(value).lower()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def build_question_text(question: str, hint: str | None) -> str:
    question = str(question).strip()
    hint = str(hint or "").strip()
    if not hint:
        return question
    return f"{question}\nContext: {hint}"


def build_scienceqa_target(
    *,
    answer_index: int,
    choices: Sequence[str],
    solution: str | None = None,
    lecture: str | None = None,
    include_lecture: bool = False,
) -> str:
    rationale_parts: list[str] = []
    cleaned_solution = normalize_optional_text(solution)
    cleaned_lecture = normalize_optional_text(lecture)

    if include_lecture and cleaned_lecture:
        rationale_parts.append(cleaned_lecture)
    if cleaned_solution:
        rationale_parts.append(cleaned_solution)
    elif cleaned_lecture and not include_lecture:
        rationale_parts.append(cleaned_lecture)

    rationale = "\n\n".join(rationale_parts) if rationale_parts else None
    return build_multiple_choice_target(
        answer_index,
        choice_text=choices[answer_index],
        rationale=rationale,
    )


def is_retained_scienceqa_record(
    row: dict[str, Any],
    *,
    allowed_subjects: set[str],
    allowed_topics: set[str],
    disable_default_filter: bool,
) -> bool:
    if disable_default_filter:
        return True

    subject = normalize_optional_text(row.get("subject")).lower()
    topic = normalize_optional_text(row.get("topic")).lower()

    if allowed_subjects and subject not in allowed_subjects:
        return False
    if allowed_topics and topic not in allowed_topics:
        return False
    return True


def extract_image_bytes(image_field: Any) -> bytes | None:
    if not isinstance(image_field, dict):
        return None
    image_bytes = image_field.get("bytes")
    if not image_bytes:
        return None
    return bytes(image_bytes)


def save_image(image_bytes: bytes, image_path: Path) -> None:
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(image_path)


def convert_split(
    parquet_path: Path,
    output_root: Path,
    split_name: str,
    *,
    allowed_subjects: set[str],
    allowed_topics: set[str],
    disable_default_filter: bool,
    include_lecture_in_target: bool,
) -> None:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    images_dir = output_root / "images" / split_name
    manifest_path = output_root / f"{split_name}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_filtered_out = 0
    skipped_missing_image = 0
    skipped_missing_choices = 0
    skipped_invalid_answer = 0

    with open(manifest_path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(df.to_dict(orient="records")):
            if not is_retained_scienceqa_record(
                row,
                allowed_subjects=allowed_subjects,
                allowed_topics=allowed_topics,
                disable_default_filter=disable_default_filter,
            ):
                skipped_filtered_out += 1
                continue

            choices = normalize_choices(row.get("choices"))
            if len(choices) < 2:
                skipped_missing_choices += 1
                continue

            answer = row.get("answer")
            if answer is None:
                skipped_invalid_answer += 1
                continue
            answer_index = int(answer)
            if answer_index < 0 or answer_index >= len(choices):
                skipped_invalid_answer += 1
                continue

            image_bytes = extract_image_bytes(row.get("image"))
            if image_bytes is None:
                skipped_missing_image += 1
                continue

            sample_id = f"scienceqa_{split_name}_{index:06d}"
            image_path = (images_dir / f"{sample_id}.png").resolve()
            save_image(image_bytes, image_path)

            record = {
                "sample_id": sample_id,
                "source_dataset": "scienceqa",
                "split": split_name,
                "image": str(image_path),
                "question": build_question_text(row.get("question", ""), row.get("hint")),
                "choices": choices,
                "correct_index": answer_index,
                "correct_answer": choices[answer_index],
                "target_text": build_scienceqa_target(
                    answer_index=answer_index,
                    choices=choices,
                    solution=row.get("solution"),
                    lecture=row.get("lecture"),
                    include_lecture=include_lecture_in_target,
                ),
                "metadata": {
                    "scienceqa_task": row.get("task"),
                    "grade": row.get("grade"),
                    "subject": row.get("subject"),
                    "topic": row.get("topic"),
                    "category": row.get("category"),
                    "skill": row.get("skill"),
                    "lecture": row.get("lecture"),
                    "solution": row.get("solution"),
                    "hint": row.get("hint"),
                    "filter_subject_match": normalize_optional_text(row.get("subject")).lower()
                    in allowed_subjects
                    if not disable_default_filter
                    else True,
                    "filter_topic_match": normalize_optional_text(row.get("topic")).lower()
                    in allowed_topics
                    if not disable_default_filter
                    else True,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{split_name}] saved {written} samples to {manifest_path}")
    print(f"[{split_name}] skipped_filtered_out={skipped_filtered_out}")
    print(f"[{split_name}] skipped_missing_image={skipped_missing_image}")
    print(f"[{split_name}] skipped_missing_choices={skipped_missing_choices}")
    print(f"[{split_name}] skipped_invalid_answer={skipped_invalid_answer}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    allowed_subjects = normalize_filter_values(args.allowed_subjects)
    allowed_topics = normalize_filter_values(args.allowed_topics)

    convert_split(
        Path(args.train_parquet),
        output_root,
        "train",
        allowed_subjects=allowed_subjects,
        allowed_topics=allowed_topics,
        disable_default_filter=bool(args.disable_default_filter),
        include_lecture_in_target=bool(args.include_lecture_in_target),
    )
    convert_split(
        Path(args.val_parquet),
        output_root,
        "val",
        allowed_subjects=allowed_subjects,
        allowed_topics=allowed_topics,
        disable_default_filter=bool(args.disable_default_filter),
        include_lecture_in_target=bool(args.include_lecture_in_target),
    )
    convert_split(
        Path(args.test_parquet),
        output_root,
        "test",
        allowed_subjects=allowed_subjects,
        allowed_topics=allowed_topics,
        disable_default_filter=bool(args.disable_default_filter),
        include_lecture_in_target=bool(args.include_lecture_in_target),
    )


if __name__ == "__main__":
    main()
