from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import DEFAULT_UNIFIED_VQA_ROOT, infer_group_key, iter_jsonl, validate_manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split existing Unified VQA train records into train/val while keeping test records untouched. "
            "The split is applied independently per dataset and grouped by image-level keys when available."
        )
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
        help="Unified VQA bundle root that contains manifests/by_dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of the current train pool kept in train; the remainder becomes val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic train/val splitting.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_train_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    grouped_records: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped_records.setdefault(infer_group_key(record), []).append(dict(record))

    group_keys = list(grouped_records.keys())
    if len(group_keys) < 2:
        raise ValueError("Need at least 2 train groups to create a train/val split.")

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    num_train_groups = int(len(group_keys) * train_ratio)
    num_train_groups = max(1, min(num_train_groups, len(group_keys) - 1))
    train_group_keys = set(group_keys[:num_train_groups])

    resplit: List[Dict[str, Any]] = []
    for group_key in group_keys:
        new_split = "train" if group_key in train_group_keys else "val"
        for record in grouped_records[group_key]:
            record["split"] = new_split
            resplit.append(record)
    return resplit


def rebuild_indexes(output_root: Path) -> Dict[str, Any]:
    by_dataset_root = output_root / "manifests" / "by_dataset"
    by_split_root = output_root / "manifests" / "by_split"
    merged_root = output_root / "manifests" / "merged"

    by_split_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    for old_file in by_split_root.glob("*.jsonl"):
        old_file.unlink()
    for old_file in merged_root.glob("*.jsonl"):
        old_file.unlink()

    split_buckets_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    merged_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"datasets": {}, "total_records": 0}

    for dataset_file in sorted(by_dataset_root.glob("*.jsonl")):
        dataset_name = dataset_file.stem
        records = list(iter_jsonl(dataset_file))
        dataset_splits: Dict[str, int] = defaultdict(int)
        for record in records:
            split = str(record.get("split", "unspecified"))
            dataset_splits[split] += 1
            split_buckets_by_dataset[f"{dataset_name}_{split}"].append(record)
            merged_by_split[split].append(record)
            all_records.append(record)
        summary["datasets"][dataset_name] = {
            "num_records": len(records),
            "splits": dict(dataset_splits),
        }
        summary["total_records"] += len(records)

    for split_name, records in split_buckets_by_dataset.items():
        write_jsonl(by_split_root / f"{split_name}.jsonl", records)
    for split_name, records in merged_by_split.items():
        write_jsonl(merged_root / f"{split_name}.jsonl", records)
    write_jsonl(merged_root / "all.jsonl", all_records)

    summary_path = output_root / "manifests" / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    validate_manifest_paths(all_records)
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    by_dataset_root = output_root / "manifests" / "by_dataset"
    if not by_dataset_root.exists():
        raise FileNotFoundError(f"Dataset manifest directory not found: {by_dataset_root}")

    applied: Dict[str, Any] = {}
    for dataset_file in sorted(by_dataset_root.glob("*.jsonl")):
        dataset_name = dataset_file.stem
        records = list(iter_jsonl(dataset_file))

        test_records: List[Dict[str, Any]] = []
        train_pool: List[Dict[str, Any]] = []
        other_records: List[Dict[str, Any]] = []
        for record in records:
            split = str(record.get("split", "unspecified"))
            if split == "test":
                test_records.append(dict(record))
            elif split in {"train", "val"}:
                train_pool.append(dict(record))
            else:
                other_records.append(dict(record))

        if not train_pool:
            updated_records = test_records + other_records
            write_jsonl(dataset_file, updated_records)
            applied[dataset_name] = {
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "num_records": len(updated_records),
                "train_records": 0,
                "val_records": 0,
                "test_records": sum(1 for record in updated_records if record.get("split") == "test"),
                "skipped": "no_train_pool",
            }
            continue

        try:
            train_val_records = split_train_records(
                train_pool,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
        except ValueError:
            train_val_records = [dict(record) for record in train_pool]
            for record in train_val_records:
                record["split"] = "train"

        updated_records = train_val_records + test_records + other_records
        write_jsonl(dataset_file, updated_records)

        train_count = sum(1 for record in updated_records if record.get("split") == "train")
        val_count = sum(1 for record in updated_records if record.get("split") == "val")
        test_count = sum(1 for record in updated_records if record.get("split") == "test")
        applied[dataset_name] = {
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "num_records": len(updated_records),
            "train_records": train_count,
            "val_records": val_count,
            "test_records": test_count,
        }

    summary = rebuild_indexes(output_root)
    print(json.dumps({"applied": applied, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
