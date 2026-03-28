from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


REASONING_MULTIPLE_CHOICE_PROMPT_TEMPLATE = """The following is a multiple choice question (with answers) related to the image below.
The option indices start from 0 and are shown explicitly as (0), (1), (2), ...
Think step by step, use the option index exactly as shown, and then output the answer in the format of \"The answer is (X)\" at the end.

{question}

Options:
{choices}
"""

ANSWER_ONLY_MULTIPLE_CHOICE_PROMPT_TEMPLATE = """The following is a multiple choice question (with answers) related to the image below.
The option indices start from 0 and are shown explicitly as (0), (1), (2), ...
Answer using the option index exactly as shown in the format of \"The answer is (X)\".

{question}

Options:
{choices}
"""

UNIFIED_MANIFEST_VERSION = 1
DEFAULT_UNIFIED_VQA_ROOT = Path("/data1/staging_datasets/unified_vqa")
DEFAULT_MMS_ROOT = Path("/data1/mms_data")
DEFAULT_MMSCI_ROOT = Path("/data1/mmsci++")
DEFAULT_MICROVQA_ROOT = Path("/data1/staging_datasets/microvqa")
DEFAULT_MICROBENCH_ROOT = Path(
    "/data1/huggingface/hub/datasets--jnirschl--uBench/snapshots/3f2c5b590bc7a208d5b60f3527ce4c76a331aa2b"
)


def get_unified_vqa_root(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_UNIFIED_VQA_ROOT
    return Path(path)


def get_unified_manifest_paths(root: str | Path | None = None) -> Dict[str, Path]:
    output_root = get_unified_vqa_root(root)
    manifests_root = output_root / "manifests"
    return {
        "root": output_root,
        "manifests": manifests_root,
        "by_dataset": manifests_root / "by_dataset",
        "by_split": manifests_root / "by_split",
        "merged": manifests_root / "merged",
        "summary": manifests_root / "summary.json",
        "train": manifests_root / "merged" / "train.jsonl",
        "test": manifests_root / "merged" / "test.jsonl",
        "all": manifests_root / "merged" / "all.jsonl",
    }


def parse_dataset_train_ratios(items: Sequence[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected DATASET=RATIO, got {item!r}")
        dataset_name, ratio_text = item.split("=", 1)
        dataset_name = dataset_name.strip()
        if not dataset_name:
            raise ValueError(f"Dataset name is empty in {item!r}")
        parsed[dataset_name] = float(ratio_text)
    return parsed


def canonical_image_path(
    output_root: str | Path,
    dataset_name: str,
    split: str,
    filename: str,
) -> Path:
    return get_unified_vqa_root(output_root) / "images" / dataset_name / split / filename


def canonical_source_path(
    output_root: str | Path,
    dataset_name: str,
    filename: str,
) -> Path:
    return get_unified_vqa_root(output_root) / "sources" / dataset_name / filename


def export_image_file(source_path: Path, destination_path: Path) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)
    return str(destination_path.resolve())


def export_source_file(source_path: Path, destination_path: Path) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)
    return str(destination_path.resolve())


def format_choices_for_prompt(choices: Sequence[str]) -> str:
    return "".join(f"  ({idx}): {choice}\n" for idx, choice in enumerate(choices))


def resolve_prompt_style(prompt_style: str | None = None) -> str:
    normalized = str(prompt_style or "reasoning").strip().lower()
    if normalized not in {"reasoning", "answer_only"}:
        raise ValueError(
            "Unsupported prompt_style. Expected one of: reasoning, answer_only. "
            f"Received: {prompt_style!r}"
        )
    return normalized


def build_multiple_choice_prompt(
    question: str,
    choices: Sequence[str],
    prompt_style: str | None = None,
) -> str:
    style = resolve_prompt_style(prompt_style)
    template = (
        ANSWER_ONLY_MULTIPLE_CHOICE_PROMPT_TEMPLATE
        if style == "answer_only"
        else REASONING_MULTIPLE_CHOICE_PROMPT_TEMPLATE
    )
    return template.format(
        question=question,
        choices=format_choices_for_prompt(choices),
    )


def build_multiple_choice_target(
    correct_index: int,
    *,
    choice_text: str | None = None,
    rationale: str | None = None,
) -> str:
    lines: List[str] = []
    if rationale:
        cleaned_rationale = " ".join(str(rationale).split()).strip()
        if cleaned_rationale:
            lines.append(cleaned_rationale)

    answer_line = f"The answer is ({correct_index})"
    if choice_text is not None:
        cleaned_choice = " ".join(str(choice_text).split()).strip()
        if cleaned_choice:
            answer_line += f": {cleaned_choice}"
    lines.append(answer_line)
    return "\n".join(lines)


def infer_answer_index(answer: str, choices: Sequence[str]) -> int:
    try:
        return list(choices).index(answer)
    except ValueError as exc:
        raise ValueError(f"Answer '{answer}' not found in choices: {choices}") from exc


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> Path:
    output_path = ensure_parent_dir(path)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_records_by_split(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        split = str(record.get("split", "unspecified"))
        grouped.setdefault(split, []).append(record)
    return grouped


def validate_manifest_paths(records: Iterable[Dict[str, Any]]) -> None:
    missing_images: List[str] = []
    missing_metadata_paths: List[str] = []

    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.exists():
            missing_images.append(str(image_path))

        metadata = record.get("metadata") or {}
        source_path = metadata.get("source_path")
        if source_path and not Path(source_path).exists():
            missing_metadata_paths.append(str(source_path))
        source_root = metadata.get("source_root")
        if source_root and not Path(source_root).exists():
            missing_metadata_paths.append(str(source_root))

    if missing_images or missing_metadata_paths:
        raise FileNotFoundError(
            "Unified manifest validation failed with "
            f"{len(set(missing_images))} missing images and "
            f"{len(set(missing_metadata_paths))} missing metadata paths."
        )


def infer_group_key(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    for key in ("image_id", "key_image", "uid"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}:{value}"

    image_path = record.get("image_path")
    if image_path:
        path = Path(str(image_path))
        return f"image_path:{path.name}"

    sample_id = record.get("sample_id")
    if sample_id is not None:
        return f"sample_id:{sample_id}"
    raise KeyError("Unable to infer split grouping key for record.")


def resplit_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    grouped_records: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group_key = infer_group_key(record)
        grouped_records.setdefault(group_key, []).append(dict(record))

    group_keys = list(grouped_records.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    num_groups = len(group_keys)
    num_train_groups = int(num_groups * train_ratio)
    if num_train_groups <= 0:
        num_train_groups = 1
    if num_train_groups >= num_groups:
        num_train_groups = num_groups - 1

    train_group_keys = set(group_keys[:num_train_groups])
    resplit: List[Dict[str, Any]] = []
    for group_key in group_keys:
        split = "train" if group_key in train_group_keys else "test"
        for record in grouped_records[group_key]:
            record["split"] = split
            resplit.append(record)
    return resplit


def build_unified_record(
    *,
    sample_id: str,
    source_dataset: str,
    split: str,
    image_path: str,
    question: str,
    choices: Sequence[str],
    correct_index: int,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    choices = [str(choice) for choice in choices]
    metadata = dict(metadata or {})
    return {
        "manifest_version": UNIFIED_MANIFEST_VERSION,
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "split": split,
        "question_type": "multiple_choice",
        "image_path": str(image_path),
        "question": str(question),
        "choices": choices,
        "correct_index": int(correct_index),
        "correct_answer": choices[int(correct_index)],
        "target_text": build_multiple_choice_target(
            int(correct_index),
            choice_text=choices[int(correct_index)],
        ),
        "metadata": metadata,
    }


def iter_question_items(question_map: Dict[str, Any]) -> Iterator[tuple[str, Dict[str, Any]]]:
    for key, value in question_map.items():
        if not value:
            continue
        if not isinstance(value, dict):
            continue
        if "question" not in value or "options" not in value:
            continue
        yield key, value
