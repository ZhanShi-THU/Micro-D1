from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import (
    DEFAULT_MMSCI_ROOT,
    DEFAULT_MMS_ROOT,
    DEFAULT_UNIFIED_VQA_ROOT,
    canonical_image_path,
    canonical_source_path,
    export_image_file,
    export_source_file,
    get_unified_manifest_paths,
    iter_jsonl,
    validate_manifest_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite an existing unified_vqa bundle into the canonical self-contained layout."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
    )
    parser.add_argument(
        "--legacy-mms-root",
        type=str,
        default=str(DEFAULT_MMS_ROOT),
    )
    parser.add_argument(
        "--legacy-mmsci-root",
        type=str,
        default=str(DEFAULT_MMSCI_ROOT),
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Best-effort deletion of legacy roots after successful rewrite.",
    )
    return parser.parse_args()


def backup_manifests(output_root: Path) -> Path:
    manifests_root = output_root / "manifests"
    backup_root = output_root / "manifests_backup_before_rehome"
    if backup_root.exists():
        return backup_root
    shutil.copytree(manifests_root, backup_root)
    return backup_root


def rewrite_record(record: Dict[str, Any], output_root: Path, mms_root: Path, mmsci_root: Path) -> Dict[str, Any]:
    dataset_name = str(record.get("source_dataset"))
    split = str(record.get("split", "unspecified"))
    metadata = dict(record.get("metadata") or {})

    if dataset_name == "mms":
        source_image = Path(record["image_path"])
        if not source_image.exists():
            source_image = mms_root / "images" / source_image.name
        record["image_path"] = export_image_file(
            source_image,
            canonical_image_path(output_root, "mms", split, source_image.name),
        )
        legacy_source = mms_root / "microvqa_custom_test.json"
        metadata["source_path"] = export_source_file(
            legacy_source,
            canonical_source_path(output_root, "mms", legacy_source.name),
        )
    elif dataset_name == "mmsci++":
        source_image = Path(record["image_path"])
        if not source_image.exists():
            source_image = mmsci_root / "images" / source_image.name
        record["image_path"] = export_image_file(
            source_image,
            canonical_image_path(output_root, "mmsci++", split, source_image.name),
        )
        legacy_source = mmsci_root / "generated_mcq.jsonl"
        metadata["source_path"] = export_source_file(
            legacy_source,
            canonical_source_path(output_root, "mmsci++", legacy_source.name),
        )
    elif dataset_name == "microvqa":
        metadata["source_root"] = str((output_root / "images" / "microvqa").resolve())
    elif dataset_name == "microbench":
        metadata["source_root"] = str((output_root / "images" / "microbench").resolve())

    record["metadata"] = metadata
    return record


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_indexes(output_root: Path, dataset_records: Dict[str, List[Dict[str, Any]]]) -> None:
    paths = get_unified_manifest_paths(output_root)
    all_records: List[Dict[str, Any]] = []
    split_records: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Any] = {"datasets": {}, "total_records": 0}

    for dataset_name, records in dataset_records.items():
        split_counts: Dict[str, int] = {}
        for record in records:
            split = str(record.get("split", "unspecified"))
            split_records.setdefault(f"{dataset_name}_{split}", []).append(record)
            split_counts[split] = split_counts.get(split, 0) + 1
            all_records.append(record)

        summary["datasets"][dataset_name] = {
            "num_records": len(records),
            "splits": split_counts,
        }
        summary["total_records"] += len(records)

    for split_name, records in split_records.items():
        write_jsonl(paths["by_split"] / f"{split_name}.jsonl", records)

    merged_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for record in all_records:
        merged_by_split.setdefault(str(record.get("split", "unspecified")), []).append(record)
    for split_name, records in merged_by_split.items():
        write_jsonl(paths["merged"] / f"{split_name}.jsonl", records)

    write_jsonl(paths["all"], all_records)
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def best_effort_delete(paths: List[Path]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            results[str(path)] = "missing"
            continue
        try:
            shutil.rmtree(path)
            results[str(path)] = "deleted"
        except PermissionError:
            results[str(path)] = "permission_denied"
    return results


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    mms_root = Path(args.legacy_mms_root)
    mmsci_root = Path(args.legacy_mmsci_root)
    paths = get_unified_manifest_paths(output_root)

    backup_root = backup_manifests(output_root)
    dataset_records: Dict[str, List[Dict[str, Any]]] = {}

    for manifest_path in sorted(paths["by_dataset"].glob("*.jsonl")):
        records = [
            rewrite_record(record, output_root, mms_root, mmsci_root)
            for record in iter_jsonl(manifest_path)
        ]
        dataset_records[manifest_path.stem] = records
        write_jsonl(manifest_path, records)

    rebuild_indexes(output_root, dataset_records)
    validate_manifest_paths(
        record
        for records in dataset_records.values()
        for record in records
    )

    delete_results: Dict[str, str] = {}
    if args.delete_legacy:
        delete_results = best_effort_delete([mms_root, mmsci_root])

    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(output_root),
                "backup_root": str(backup_root),
                "datasets": {name: len(records) for name, records in dataset_records.items()},
                "legacy_cleanup": delete_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
