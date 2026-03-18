from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import (
    DEFAULT_UNIFIED_VQA_ROOT,
    iter_jsonl,
    validate_manifest_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild merged manifests and summary.json.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
    )
    parser.add_argument(
        "--skip-validate-paths",
        action="store_true",
        help="Skip post-rebuild validation of image_path/source_path/source_root values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    by_dataset_root = output_root / "manifests" / "by_dataset"
    by_split_root = output_root / "manifests" / "by_split"
    merged_root = output_root / "manifests" / "merged"

    by_split_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(by_dataset_root.glob("*.jsonl"))
    split_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary: Dict[str, Any] = {"datasets": {}, "total_records": 0}
    all_records: List[Dict[str, Any]] = []

    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem
        records = list(iter_jsonl(dataset_file))
        dataset_splits: Dict[str, int] = defaultdict(int)
        for record in records:
            split = str(record.get("split", "unspecified"))
            dataset_splits[split] += 1
            split_buckets[split].append(record)
            all_records.append(record)
        summary["datasets"][dataset_name] = {
            "num_records": len(records),
            "splits": dict(dataset_splits),
        }
        summary["total_records"] += len(records)

    for split, records in split_buckets.items():
        with open(merged_root / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(merged_root / "all.jsonl", "w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(output_root / "manifests" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if not args.skip_validate_paths:
        validate_manifest_paths(all_records)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
