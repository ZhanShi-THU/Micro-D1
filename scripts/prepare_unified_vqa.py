from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_vqa import (
    DEFAULT_MICROBENCH_ROOT,
    DEFAULT_MICROVQA_ROOT,
    DEFAULT_MMSCI_ROOT,
    DEFAULT_MMS_ROOT,
    DEFAULT_UNIFIED_VQA_ROOT,
    build_unified_record,
    canonical_image_path,
    canonical_source_path,
    export_image_file,
    export_source_file,
    group_records_by_split,
    infer_answer_index,
    iter_question_items,
    parse_dataset_train_ratios,
    resplit_records,
    validate_manifest_paths,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize microscopy VQA datasets into one manifest schema."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_UNIFIED_VQA_ROOT),
    )
    parser.add_argument(
        "--mms-root",
        type=str,
        default=str(DEFAULT_MMS_ROOT),
    )
    parser.add_argument(
        "--mms-split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--mmsci-root",
        type=str,
        default=str(DEFAULT_MMSCI_ROOT),
    )
    parser.add_argument(
        "--mmsci-split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--microvqa-root",
        type=str,
        default=str(DEFAULT_MICROVQA_ROOT),
        help="Path created by datasets.save_to_disk(...) for MicroVQA.",
    )
    parser.add_argument(
        "--microbench-root",
        type=str,
        default=str(DEFAULT_MICROBENCH_ROOT),
        help="Directory containing perception/0.1.0/*.arrow for MicroBench/uBench.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mms", "mmsci++", "microvqa", "microbench"],
        choices=["mms", "mmsci++", "microvqa", "microbench"],
    )
    parser.add_argument(
        "--dataset-train-ratio",
        action="append",
        default=[],
        metavar="DATASET=RATIO",
        help=(
            "Optionally resplit prepared dataset records before writing manifests. "
            "Example: --dataset-train-ratio microbench=0.95"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic dataset re-splitting.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def export_pil_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.convert("RGB").save(path)
    return str(path.resolve())


def export_bytes_image(blob: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with Image.open(io.BytesIO(blob)) as image:
            image.convert("RGB").save(path)
    return str(path.resolve())


def repair_embedded_choices(choices: List[str]) -> List[str]:
    repaired: List[str] = []
    extra_choice_pattern = re.compile(r"^[A-Z]:\s*(.+)$")

    for choice in choices:
        lines = [line.strip() for line in str(choice).splitlines() if line.strip()]
        if not lines:
            continue
        repaired.append(lines[0])
        for line in lines[1:]:
            if line.lower().startswith("answer with"):
                continue
            match = extra_choice_pattern.match(line)
            if match:
                repaired.append(match.group(1).strip())
    return repaired or [str(choice) for choice in choices]


def prepare_mms(dataset_root: Path, output_root: Path, split: str) -> List[Dict[str, Any]]:
    source_path = dataset_root / "microvqa_custom_test.json"
    exported_source_path = export_source_file(
        source_path,
        canonical_source_path(output_root, "mms", source_path.name),
    )
    samples = load_json(source_path)
    records: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples):
        choices = repair_embedded_choices(sample["choices"])
        source_image_path = (dataset_root / sample["image_path"]).resolve()
        image_path = export_image_file(
            source_image_path,
            canonical_image_path(output_root, "mms", split, source_image_path.name),
        )
        record = build_unified_record(
            sample_id=str(sample.get("key_question", f"mms_{split}_{index:06d}")),
            source_dataset="mms",
            split=split,
            image_path=image_path,
            question=sample["question"],
            choices=choices,
            correct_index=int(sample["correct_index"]),
            metadata={
                "key_question": sample.get("key_question"),
                "key_image": sample.get("key_image"),
                "task": sample.get("task"),
                "source_path": exported_source_path,
            },
        )
        records.append(record)

    return records


def prepare_mmsci(dataset_root: Path, output_root: Path, split: str) -> List[Dict[str, Any]]:
    source_path = dataset_root / "generated_mcq.jsonl"
    exported_source_path = export_source_file(
        source_path,
        canonical_source_path(output_root, "mmsci++", source_path.name),
    )
    records: List[Dict[str, Any]] = []

    for index, sample in enumerate(iter_jsonl(source_path)):
        correct_index = infer_answer_index(sample["answer"], sample["options"])
        source_image_path = (dataset_root / sample["image"]).resolve()
        image_path = export_image_file(
            source_image_path,
            canonical_image_path(output_root, "mmsci++", split, source_image_path.name),
        )
        record = build_unified_record(
            sample_id=str(sample.get("uid", f"mmsci_pp_{split}_{index:06d}")),
            source_dataset="mmsci++",
            split=split,
            image_path=image_path,
            question=sample["question"],
            choices=sample["options"],
            correct_index=correct_index,
            metadata={
                "uid": sample.get("uid"),
                "category": sample.get("category"),
                "subject": sample.get("subject"),
                "caption": sample.get("caption"),
                "conversations": sample.get("conversations"),
                "source_path": exported_source_path,
            },
        )
        records.append(record)

    return records


def prepare_microvqa(dataset_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("MicroVQA preparation requires the 'datasets' package.") from exc

    dataset = load_from_disk(str(dataset_root))
    image_output_root = output_root / "images" / "microvqa"
    records: List[Dict[str, Any]] = []

    for split, split_dataset in dataset.items():
        for index, sample in enumerate(split_dataset):
            images_list = sample["images_list"]
            if not images_list:
                raise ValueError(f"MicroVQA sample {index} has empty images_list.")

            image_filename = f"{sample.get('key_question', f'{split}_{index:06d}')}_0.png"
            image_path = export_pil_image(
                images_list[0],
                image_output_root / split / image_filename,
            )
            record = build_unified_record(
                sample_id=str(sample.get("key_question", f"microvqa_{split}_{index:06d}")),
                source_dataset="microvqa",
                split=split,
                image_path=image_path,
                question=sample["question"],
                choices=sample["choices"],
                correct_index=int(sample["correct_index"]),
                metadata={
                    "key_question": sample.get("key_question"),
                    "key_image": sample.get("key_image"),
                    "task": sample.get("task"),
                    "task_str": sample.get("task_str"),
                    "images_source": sample.get("images_source"),
                    "image_caption": sample.get("image_caption"),
                    "context_image_generation": sample.get("context_image_generation"),
                    "context_motivation": sample.get("context_motivation"),
                    "source_root": str(image_output_root.resolve()),
                },
            )
            records.append(record)

    return records


def iter_microbench_rows(dataset_root: Path) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("MicroBench preparation requires pyarrow.") from exc

    for arrow_path in sorted(dataset_root.rglob("*.arrow")):
        with ipc.open_stream(str(arrow_path)) as reader:
            table = reader.read_all()
        for row in table.to_pylist():
            yield row


def iter_microbench_records(dataset_root: Path, output_root: Path) -> Iterator[Dict[str, Any]]:
    image_output_root = output_root / "images" / "microbench"
    image_cache: Dict[str, str] = {}

    for row in iter_microbench_rows(dataset_root):
        split = str(row.get("split", "test"))
        image_id = str(row["image_id"])
        if image_id not in image_cache:
            image_obj = row["image"]
            image_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None
            if image_bytes is None:
                raise ValueError(f"MicroBench image payload missing bytes for image_id={image_id}")
            image_cache[image_id] = export_bytes_image(
                image_bytes,
                image_output_root / split / f"{image_id}.png",
            )

        for question_key, question in iter_question_items(row["questions"]):
            sample_id = f"microbench_{image_id}_{question_key}"
            record = build_unified_record(
                sample_id=sample_id,
                source_dataset="microbench",
                split=split,
                image_path=image_cache[image_id],
                question=question["question"],
                choices=question["options"],
                correct_index=int(question["answer_idx"]),
                metadata={
                    "image_id": image_id,
                    "question_id": question.get("id"),
                    "question_name": question.get("name"),
                    "dataset": row.get("dataset"),
                    "domain": row.get("domain"),
                    "subdomain": row.get("subdomain"),
                    "modality": row.get("modality"),
                    "submodality": row.get("submodality"),
                    "stain": row.get("stain"),
                    "label_name": row.get("label_name"),
                    "license": row.get("license"),
                    "pmid": row.get("pmid"),
                    "source_root": str(image_output_root.resolve()),
                },
            )
            yield record


def prepare_microbench(dataset_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    return list(iter_microbench_records(dataset_root, output_root))


def write_manifest_bundle(output_root: Path, dataset_name: str, records: List[Dict[str, Any]]) -> None:
    manifests_root = output_root / "manifests"
    by_dataset_root = manifests_root / "by_dataset"
    by_split_root = manifests_root / "by_split"

    write_jsonl(by_dataset_root / f"{dataset_name}.jsonl", records)

    split_groups = group_records_by_split(records)
    for split, split_records in split_groups.items():
        write_jsonl(by_split_root / f"{dataset_name}_{split}.jsonl", split_records)


def merge_split_manifests(output_root: Path, dataset_records: Dict[str, List[Dict[str, Any]]]) -> None:
    manifests_root = output_root / "manifests"
    merged_root = manifests_root / "merged"
    all_records: List[Dict[str, Any]] = []
    split_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for records in dataset_records.values():
        all_records.extend(records)
        for split, split_records in group_records_by_split(records).items():
            split_groups[split].extend(split_records)

    write_jsonl(merged_root / "all.jsonl", all_records)
    for split, split_records in split_groups.items():
        write_jsonl(merged_root / f"{split}.jsonl", split_records)


def write_summary(output_root: Path, dataset_records: Dict[str, List[Dict[str, Any]]]) -> None:
    summary: Dict[str, Any] = {"datasets": {}, "total_records": 0}
    for dataset_name, records in dataset_records.items():
        split_groups = group_records_by_split(records)
        summary["datasets"][dataset_name] = {
            "num_records": len(records),
            "splits": {split: len(split_records) for split, split_records in split_groups.items()},
        }
        summary["total_records"] += len(records)

    summary_path = output_root / "manifests" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_train_ratios = parse_dataset_train_ratios(args.dataset_train_ratio)

    dataset_records: Dict[str, List[Dict[str, Any]]] = {}

    if "mms" in args.datasets:
        dataset_records["mms"] = prepare_mms(Path(args.mms_root), output_root, args.mms_split)
    if "mmsci++" in args.datasets:
        dataset_records["mmsci++"] = prepare_mmsci(Path(args.mmsci_root), output_root, args.mmsci_split)
    if "microvqa" in args.datasets:
        dataset_records["microvqa"] = prepare_microvqa(Path(args.microvqa_root), output_root)
    if "microbench" in args.datasets:
        dataset_records["microbench"] = prepare_microbench(Path(args.microbench_root), output_root)

    for dataset_name, train_ratio in dataset_train_ratios.items():
        if dataset_name not in dataset_records:
            raise ValueError(
                f"Requested resplit for dataset {dataset_name!r}, but it is not included in --datasets."
            )
        dataset_records[dataset_name] = resplit_records(
            dataset_records[dataset_name],
            train_ratio=train_ratio,
            seed=args.seed,
        )

    for dataset_name, records in dataset_records.items():
        write_manifest_bundle(output_root, dataset_name, records)

    merge_split_manifests(output_root, dataset_records)
    write_summary(output_root, dataset_records)
    validate_manifest_paths(
        record
        for records in dataset_records.values()
        for record in records
    )

    print(json.dumps(
        {
            "output_root": str(output_root),
            "datasets": {name: len(records) for name, records in dataset_records.items()},
            "dataset_train_ratios": dataset_train_ratios,
            "seed": args.seed,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
