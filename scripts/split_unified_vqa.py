from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import (
    DEFAULT_UNIFIED_VQA_ROOT,
    get_unified_manifest_paths,
    iter_jsonl,
    parse_dataset_train_ratios,
    resplit_records,
    validate_manifest_paths,
)
from scripts.finalize_unified_vqa import main as _unused  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply train/test re-splitting to one or more datasets inside an existing unified_vqa bundle. "
            "Ratios are specified as dataset=train_ratio."
        )
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
    )
    parser.add_argument(
        "--dataset-train-ratio",
        action="append",
        default=[],
        metavar="DATASET=RATIO",
        help=(
            "Example: --dataset-train-ratio microbench=0.95 "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic re-splitting.",
    )
    return parser.parse_args()

def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_indexes(output_root: Path) -> Dict[str, dict]:
    from collections import defaultdict

    paths = get_unified_manifest_paths(output_root)
    dataset_files = sorted(paths["by_dataset"].glob("*.jsonl"))
    split_buckets: Dict[str, List[dict]] = defaultdict(list)
    all_records: List[dict] = []
    summary: Dict[str, dict] = {"datasets": {}, "total_records": 0}

    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem
        records = list(iter_jsonl(dataset_file))
        dataset_splits: Dict[str, int] = defaultdict(int)
        for record in records:
            split = str(record.get("split", "unspecified"))
            dataset_splits[split] += 1
            split_buckets[f"{dataset_name}_{split}"].append(record)
            all_records.append(record)
        summary["datasets"][dataset_name] = {
            "num_records": len(records),
            "splits": dict(dataset_splits),
        }
        summary["total_records"] += len(records)

    by_split_root = paths["by_split"]
    merged_root = paths["merged"]
    by_split_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    for split_name, records in split_buckets.items():
        write_jsonl(by_split_root / f"{split_name}.jsonl", records)

    merged_by_split: Dict[str, List[dict]] = defaultdict(list)
    for record in all_records:
        merged_by_split[str(record.get("split", "unspecified"))].append(record)
    for split_name, records in merged_by_split.items():
        write_jsonl(merged_root / f"{split_name}.jsonl", records)
    write_jsonl(paths["all"], all_records)

    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    validate_manifest_paths(all_records)
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    paths = get_unified_manifest_paths(output_root)
    ratios = parse_dataset_train_ratios(args.dataset_train_ratio)
    if not ratios:
        raise ValueError("At least one --dataset-train-ratio DATASET=RATIO must be provided.")

    applied: Dict[str, dict] = {}
    for dataset_name, train_ratio in ratios.items():
        dataset_manifest = paths["by_dataset"] / f"{dataset_name}.jsonl"
        if not dataset_manifest.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest}")
        records = list(iter_jsonl(dataset_manifest))
        resplit = resplit_records(records, train_ratio=train_ratio, seed=args.seed)
        write_jsonl(dataset_manifest, resplit)
        train_count = sum(1 for record in resplit if record.get("split") == "train")
        test_count = sum(1 for record in resplit if record.get("split") == "test")
        applied[dataset_name] = {
            "train_ratio": train_ratio,
            "seed": args.seed,
            "num_records": len(resplit),
            "train_records": train_count,
            "test_records": test_count,
        }

    summary = rebuild_indexes(output_root)
    print(json.dumps({"applied": applied, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
