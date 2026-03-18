from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import DEFAULT_MICROBENCH_ROOT, DEFAULT_UNIFIED_VQA_ROOT
from scripts.prepare_unified_vqa import (
    merge_split_manifests,
    prepare_microbench,
    write_manifest_bundle,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare only MicroBench/uBench into unified jsonl.")
    parser.add_argument(
        "--microbench-root",
        type=str,
        default=str(DEFAULT_MICROBENCH_ROOT),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = prepare_microbench(Path(args.microbench_root), output_root)
    write_manifest_bundle(output_root, "microbench", records)
    merge_split_manifests(output_root, {"microbench": records})
    write_summary(output_root, {"microbench": records})

    split_counts: dict[str, int] = {}
    for record in records:
        split = str(record.get("split", "unspecified"))
        split_counts[split] = split_counts.get(split, 0) + 1

    print(json.dumps({"microbench": split_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
