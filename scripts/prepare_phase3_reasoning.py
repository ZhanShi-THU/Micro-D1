from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import build_multiple_choice_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reasoning-supervised Phase 3 manifests from generated MCQ+reason JSONL."
    )
    parser.add_argument(
        "--source-jsonl",
        type=str,
        default="/data1/staging_datasets/unified_vqa/sources/mmsci++/generated_mcq_with_reason.jsonl",
        help="Path to the generated MCQ with reason JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/staging_datasets/phase3_reasoning/mmsci_reasoning",
        help="Directory to write all/train/val manifests and summary.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mmsci_reasoning",
        help="source_dataset name written into the output manifests.",
    )
    parser.add_argument(
        "--unified-images-root",
        type=str,
        default="/data1/staging_datasets/unified_vqa/images/mmsci++",
        help="Root directory containing prepared MMSCI++ images organized by split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for the generated manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_image_path(
    image_value: str,
    *,
    source_root: Path,
    unified_images_root: Path,
) -> Path:
    raw_path = Path(image_value)
    candidate_paths = [
        source_root / raw_path,
        unified_images_root / raw_path,
        unified_images_root / raw_path.name,
        unified_images_root / "train" / raw_path.name,
        unified_images_root / "val" / raw_path.name,
        unified_images_root / "test" / raw_path.name,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unable to resolve image path for {image_value!r}. Tried: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def build_target_text(reason: str, correct_index: int, choice_text: str) -> str:
    cleaned_reason = str(reason).strip()
    return build_multiple_choice_target(
        correct_index,
        choice_text=choice_text,
        rationale=cleaned_reason or None,
    )


def convert_record(
    row: Dict[str, Any],
    *,
    source_jsonl: Path,
    unified_images_root: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    options = list(row["options"])
    answer = str(row["answer"])
    correct_index = options.index(answer)
    image_path = resolve_image_path(
        row["image"],
        source_root=source_jsonl.parent,
        unified_images_root=unified_images_root,
    )

    reason = str(row.get("reason", "")).strip()
    metadata = {
        "uid": row.get("uid"),
        "category": row.get("category"),
        "subject": row.get("subject"),
        "caption": row.get("caption"),
        "reason": reason,
        "original_image_rel": row.get("image"),
        "conversation_count": len(row.get("conversations") or []),
        "source_path": str(source_jsonl),
    }

    return {
        "sample_id": row.get("uid"),
        "source_dataset": dataset_name,
        "image": str(image_path),
        "question": str(row["question"]).strip(),
        "choices": options,
        "correct_index": correct_index,
        "target_text": build_target_text(reason, correct_index, options[correct_index]),
        "metadata": metadata,
    }


def split_records(
    records: List[Dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group_key = str(record["image"])
        grouped.setdefault(group_key, []).append(record)

    group_keys = list(grouped.keys())
    random.Random(seed).shuffle(group_keys)

    num_train_groups = int(len(group_keys) * train_ratio)
    num_train_groups = max(1, min(num_train_groups, len(group_keys) - 1))
    train_group_keys = set(group_keys[:num_train_groups])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    for group_key in group_keys:
        split = "train" if group_key in train_group_keys else "val"
        target = train_records if split == "train" else val_records
        for record in grouped[group_key]:
            copied = dict(record)
            copied["split"] = split
            target.append(copied)
    return train_records, val_records


def main() -> None:
    args = parse_args()
    source_jsonl = Path(args.source_jsonl)
    output_dir = Path(args.output_dir)
    unified_images_root = Path(args.unified_images_root)

    raw_rows = load_jsonl(source_jsonl)
    converted = [
        convert_record(
            row,
            source_jsonl=source_jsonl,
            unified_images_root=unified_images_root,
            dataset_name=args.dataset_name,
        )
        for row in raw_rows
    ]
    train_records, val_records = split_records(
        converted,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all.jsonl", converted)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    summary = {
        "source_jsonl": str(source_jsonl),
        "dataset_name": args.dataset_name,
        "unified_images_root": str(unified_images_root),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "num_total": len(converted),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "example_target_preview": train_records[0]["target_text"][:300] if train_records else "",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
