from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, List, Mapping, Sequence

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cli import (
    BaselineQwenEvaluator,
    ModularVLMEvaluator,
    build_baseline_prompt,
    build_mcq_summary,
    build_microvqa_prompt,
    ensure_output_dir,
    load_config,
    load_jsonl,
    parse_choice_answer,
    resolve_image,
    write_csv,
    write_json,
)


DEFAULT_TASK_ALIAS_MAP = {
    "perception": "EU",
    "hypothesis_gen": "HG",
    "experiment_proposal": "EP",
}

CODE_VERSION_PATHS = (
    PROJECT_ROOT / "evaluation" / "run_microvqa_suite.py",
    PROJECT_ROOT / "evaluation" / "cli.py",
    PROJECT_ROOT / "data" / "dataset.py",
    PROJECT_ROOT / "data" / "unified_vqa.py",
    PROJECT_ROOT / "models" / "adapter.py",
    PROJECT_ROOT / "models" / "vision_encoder.py",
    PROJECT_ROOT / "models" / "modular_vlm.py",
    PROJECT_ROOT / "models" / "state_loading.py",
    PROJECT_ROOT / "models" / "generation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the detailed MicroVQA gold-standard suite with EU/HG/EP breakdowns."
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        default="evaluation/suites/microvqa_gold_detailed.yaml",
        help="YAML file describing the detailed MicroVQA suite.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional single-device override for all runs.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated device list for multi-worker evaluation, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes to use. Defaults to number of devices.",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="results",
        choices=("off", "results"),
        help="Result cache mode for completed runs.",
    )
    parser.add_argument(
        "--resume-partials",
        action="store_true",
        help="Reuse matching shard partials when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap for quick smoke evaluation.",
    )
    return parser.parse_args()


def load_suite_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(preferred: str | None) -> torch.device:
    if preferred:
        device_name = preferred
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cuda":
        device_name = "cuda:0"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def resolve_device_names(args: argparse.Namespace, suite_cfg: Dict[str, Any]) -> List[str]:
    raw_devices = args.devices or suite_cfg.get("devices")
    if raw_devices:
        if isinstance(raw_devices, str):
            parts = [part.strip() for part in raw_devices.split(",")]
        else:
            parts = [str(part).strip() for part in raw_devices]
        device_names = [part for part in parts if part]
    else:
        single = args.device or suite_cfg.get("device")
        chosen = choose_device(single)
        device_names = [str(chosen)]

    normalized: List[str] = []
    for device_name in device_names:
        if device_name == "cuda":
            normalized.append("cuda:0")
        else:
            normalized.append(device_name)
    return normalized


def apply_config_overrides(config: Dict[str, Any], run_cfg: Dict[str, Any], manifest: str) -> Dict[str, Any]:
    updated = json.loads(json.dumps(config))
    updated.setdefault("data", {})
    updated["data"]["test_manifest"] = manifest

    if run_cfg.get("image_size") is not None:
        updated["data"]["image_size"] = int(run_cfg["image_size"])
    if run_cfg.get("image_preprocessing") is not None:
        updated["data"]["image_preprocessing"] = str(run_cfg["image_preprocessing"])
    if run_cfg.get("dynamic_buckets") is not None:
        updated["data"]["dynamic_buckets"] = list(run_cfg["dynamic_buckets"])
    if run_cfg.get("patch_size") is not None:
        updated["data"]["patch_size"] = int(run_cfg["patch_size"])
    if run_cfg.get("prompt_style") is not None:
        updated["data"]["prompt_style"] = str(run_cfg["prompt_style"])
    if run_cfg.get("max_text_length") is not None:
        updated["data"]["max_text_length"] = int(run_cfg["max_text_length"])
    if run_cfg.get("image_root") is not None:
        updated["data"]["image_root"] = run_cfg["image_root"]
    return updated


def normalize_task_name(task_name: str | None) -> str:
    return str(task_name or "unknown").strip().lower()


def resolve_task_alias_map(raw_value: Any) -> Dict[str, str]:
    mapping = dict(DEFAULT_TASK_ALIAS_MAP)
    if isinstance(raw_value, Mapping):
        for key, value in raw_value.items():
            mapping[normalize_task_name(str(key))] = str(value)
    return mapping


def extract_microvqa_task(sample: Dict[str, Any], task_alias_map: Mapping[str, str]) -> tuple[str, str]:
    metadata = dict(sample.get("metadata") or {})
    raw_task = normalize_task_name(metadata.get("task_str"))
    if raw_task in {"", "none", "null", "unknown"}:
        raw_task = normalize_task_name(metadata.get("task"))
    alias = task_alias_map.get(raw_task, raw_task.upper() if raw_task != "unknown" else "UNKNOWN")
    return raw_task, alias


def build_task_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_counts: Dict[str, int] = {}
    raw_correct: Dict[str, int] = {}
    alias_counts: Dict[str, int] = {}
    alias_correct: Dict[str, int] = {}

    for row in rows:
        raw_task = str(row["task_raw"])
        alias = str(row["task_alias"])
        is_correct = bool(row["correct"])

        raw_counts[raw_task] = raw_counts.get(raw_task, 0) + 1
        raw_correct[raw_task] = raw_correct.get(raw_task, 0) + int(is_correct)
        alias_counts[alias] = alias_counts.get(alias, 0) + 1
        alias_correct[alias] = alias_correct.get(alias, 0) + int(is_correct)

    raw_accuracy = {
        task_name: raw_correct[task_name] / count
        for task_name, count in raw_counts.items()
        if count > 0
    }
    alias_accuracy = {
        alias: alias_correct[alias] / count
        for alias, count in alias_counts.items()
        if count > 0
    }
    macro_accuracy = (
        sum(alias_accuracy.values()) / len(alias_accuracy)
        if alias_accuracy
        else math.nan
    )

    return {
        "raw_task_counts": raw_counts,
        "raw_task_accuracy": raw_accuracy,
        "alias_counts": alias_counts,
        "alias_accuracy": alias_accuracy,
        "macro_accuracy_by_alias": macro_accuracy,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_code_version() -> str:
    payload = {str(path.relative_to(PROJECT_ROOT)): _hash_file(path) for path in CODE_VERSION_PATHS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_run_signature(
    *,
    kind: str,
    suite_manifest: str,
    run_name: str,
    prompt_style: str,
    max_new_tokens: int,
    checkpoint: str | None,
    model_path: str | None,
    data_cfg: Dict[str, Any],
    task_alias_map: Mapping[str, str],
    code_version: str,
) -> str:
    payload = {
        "kind": kind,
        "suite_manifest": suite_manifest,
        "run_name": run_name,
        "prompt_style": prompt_style,
        "max_new_tokens": int(max_new_tokens),
        "checkpoint": checkpoint,
        "model_path": model_path,
        "image_preprocessing": data_cfg.get("image_preprocessing"),
        "image_size": data_cfg.get("image_size"),
        "dynamic_buckets": data_cfg.get("dynamic_buckets"),
        "patch_size": data_cfg.get("patch_size"),
        "max_text_length": data_cfg.get("max_text_length"),
        "image_root": data_cfg.get("image_root"),
        "task_alias_map": dict(task_alias_map),
        "code_version": code_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def get_run_artifact_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "predictions": run_dir / "predictions.csv",
        "summary": run_dir / "summary.json",
        "metadata": run_dir / "run_metadata.json",
        "partials_dir": run_dir / ".partials",
    }


def load_cached_summary(run_dir: Path, run_signature: str) -> Dict[str, Any] | None:
    paths = get_run_artifact_paths(run_dir)
    if not (paths["predictions"].exists() and paths["summary"].exists() and paths["metadata"].exists()):
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("run_signature") != run_signature:
        return None
    return json.loads(paths["summary"].read_text(encoding="utf-8"))


def log_status(message: str) -> None:
    print(message, flush=True)


def write_run_metadata(
    *,
    run_dir: Path,
    run_signature: str,
    cache_mode: str,
    device_names: Sequence[str],
    worker_count: int,
    num_samples: int,
    max_new_tokens: int,
    prompt_style: str,
    checkpoint: str | None,
    kind: str,
) -> None:
    metadata = {
        "run_signature": run_signature,
        "cache_mode": cache_mode,
        "devices": list(device_names),
        "worker_count": int(worker_count),
        "num_samples": int(num_samples),
        "max_new_tokens": int(max_new_tokens),
        "prompt_style": prompt_style,
        "checkpoint": checkpoint,
        "kind": kind,
    }
    write_json(run_dir / "run_metadata.json", metadata)


def shard_samples(samples: Sequence[Dict[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(max(shard_count, 1))]
    for sample_index, sample in enumerate(samples):
        shard_id = sample_index % len(shards)
        shard_sample = dict(sample)
        shard_sample["_sample_order"] = sample_index
        shards[shard_id].append(shard_sample)
    return shards


def get_partial_paths(run_dir: Path, worker_id: int) -> tuple[Path, Path]:
    partials_dir = run_dir / ".partials"
    return partials_dir / f"part_{worker_id}.jsonl", partials_dir / f"meta_{worker_id}.json"


def write_jsonl_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_partial_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def reusable_partial_exists(
    *,
    part_path: Path,
    meta_path: Path,
    run_signature: str,
    worker_id: int,
) -> bool:
    if not (part_path.exists() and meta_path.exists()):
        return False
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        metadata.get("run_signature") == run_signature
        and int(metadata.get("worker_id", -1)) == int(worker_id)
    )


def strip_internal_row_fields(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if not key.startswith("_")})
    return cleaned


def build_modular_summary(
    *,
    run_cfg: Dict[str, Any],
    suite_manifest: str,
    config: Dict[str, Any],
    prompt_style: str,
    max_new_tokens: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    parse_success = sum(int(row.get("prediction_index") not in {None, ""}) for row in rows)
    mcq_summary = build_mcq_summary(rows, {"modular_vlm": "correct"})
    task_metrics = build_task_metrics(rows)
    return {
        "run_name": run_cfg["name"],
        "config": str(run_cfg["config"]),
        "checkpoint": run_cfg.get("checkpoint"),
        "manifest": suite_manifest,
        "prompt_style": prompt_style,
        "image_preprocessing": config["data"].get("image_preprocessing"),
        "image_size": config["data"].get("image_size"),
        "dynamic_buckets": config["data"].get("dynamic_buckets"),
        "patch_size": config["data"].get("patch_size"),
        "max_new_tokens": int(max_new_tokens),
        "parse_success_rate": parse_success / len(rows) if rows else math.nan,
        "mcq_metrics": mcq_summary["models"]["modular_vlm"],
        "dataset_counts": mcq_summary["dataset_counts"],
        "microvqa_task_metrics": task_metrics,
        "num_samples": len(rows),
    }


def build_baseline_summary(
    *,
    baseline_cfg: Dict[str, Any],
    suite_manifest: str,
    prompt_style: str,
    max_new_tokens: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    parse_success = sum(int(row.get("prediction_index") not in {None, ""}) for row in rows)
    mcq_summary = build_mcq_summary(rows, {"baseline_qwen3_vl": "correct"})
    task_metrics = build_task_metrics(rows)
    return {
        "run_name": baseline_cfg.get("name", "baseline_qwen3_vl"),
        "model_path": str(baseline_cfg["model_path"]),
        "manifest": suite_manifest,
        "prompt_style": prompt_style,
        "max_new_tokens": int(max_new_tokens),
        "parse_success_rate": parse_success / len(rows) if rows else math.nan,
        "mcq_metrics": mcq_summary["models"]["baseline_qwen3_vl"],
        "dataset_counts": mcq_summary["dataset_counts"],
        "microvqa_task_metrics": task_metrics,
        "num_samples": len(rows),
    }


def evaluate_modular_rows(
    *,
    config: Dict[str, Any],
    checkpoint: str | None,
    samples: Sequence[Dict[str, Any]],
    device_name: str,
    max_new_tokens: int,
    task_alias_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    image_root = config.get("data", {}).get("image_root")
    prompt_style = str(config.get("data", {}).get("prompt_style", "reasoning"))
    evaluator = ModularVLMEvaluator(
        config=config,
        device=choose_device(device_name),
        adapter_checkpoint=checkpoint,
    )

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        image = resolve_image(sample, image_root=image_root)
        prompt = build_microvqa_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=prompt_style,
        )
        response = evaluator.generate(
            image,
            prompt,
            max_new_tokens,
            stop_on_first_parsed_answer=True,
        )
        pred = parse_choice_answer(response)
        answer = int(sample["correct_index"])
        task_raw, task_alias = extract_microvqa_task(sample, task_alias_map)
        rows.append(
            {
                "_sample_order": int(sample["_sample_order"]),
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "split": sample.get("split"),
                "task_raw": task_raw,
                "task_alias": task_alias,
                "question": sample["question"],
                "correct_index": answer,
                "prediction_index": pred,
                "response": response,
                "correct": pred == answer,
            }
        )
    return rows


def evaluate_baseline_rows(
    *,
    baseline_cfg: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    device_name: str,
    max_new_tokens: int,
    task_alias_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    image_root = baseline_cfg.get("image_root")
    prompt_style = str(baseline_cfg.get("prompt_style", "answer_only"))
    evaluator = BaselineQwenEvaluator(
        model_path=str(baseline_cfg["model_path"]),
        device=choose_device(device_name),
    )

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        image = resolve_image(sample, image_root=image_root)
        prompt = build_baseline_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=prompt_style,
        )
        response = evaluator.generate(image, prompt, max_new_tokens)
        pred = parse_choice_answer(response)
        answer = int(sample["correct_index"])
        task_raw, task_alias = extract_microvqa_task(sample, task_alias_map)
        rows.append(
            {
                "_sample_order": int(sample["_sample_order"]),
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "split": sample.get("split"),
                "task_raw": task_raw,
                "task_alias": task_alias,
                "question": sample["question"],
                "correct_index": answer,
                "prediction_index": pred,
                "response": response,
                "correct": pred == answer,
            }
        )
    return rows


def evaluate_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(task["kind"])
    if kind == "modular":
        rows = evaluate_modular_rows(
            config=task["config"],
            checkpoint=task.get("checkpoint"),
            samples=task["samples"],
            device_name=task["device_name"],
            max_new_tokens=int(task["max_new_tokens"]),
            task_alias_map=task["task_alias_map"],
        )
    elif kind == "baseline":
        rows = evaluate_baseline_rows(
            baseline_cfg=task["baseline_cfg"],
            samples=task["samples"],
            device_name=task["device_name"],
            max_new_tokens=int(task["max_new_tokens"]),
            task_alias_map=task["task_alias_map"],
        )
    else:
        raise ValueError(f"Unsupported worker kind: {kind}")

    part_path = Path(task["part_path"])
    meta_path = Path(task["meta_path"])
    write_jsonl_rows(part_path, rows)
    write_json(
        meta_path,
        {
            "run_signature": task["run_signature"],
            "worker_id": int(task["worker_id"]),
            "device": task["device_name"],
            "row_count": len(rows),
            "kind": kind,
        },
    )
    return {
        "worker_id": int(task["worker_id"]),
        "device_name": task["device_name"],
        "row_count": len(rows),
    }


def execute_parallel_tasks(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    if len(tasks) == 1:
        return [evaluate_worker(tasks[0])]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        return pool.map(evaluate_worker, tasks)


def make_leaderboard_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(summary["mcq_metrics"])
    task_metrics = dict(summary.get("microvqa_task_metrics") or {})
    alias_accuracy = dict(task_metrics.get("alias_accuracy") or {})
    return {
        "run_name": summary["run_name"],
        "overall_accuracy": metrics.get("overall_accuracy"),
        "macro_accuracy_by_dataset": metrics.get("macro_accuracy_by_dataset"),
        "microvqa_macro_accuracy": task_metrics.get("macro_accuracy_by_alias"),
        "eu_accuracy": alias_accuracy.get("EU"),
        "hg_accuracy": alias_accuracy.get("HG"),
        "ep_accuracy": alias_accuracy.get("EP"),
        "num_correct": metrics.get("num_correct"),
        "num_samples": metrics.get("num_samples"),
        "parse_success_rate": summary.get("parse_success_rate"),
        "checkpoint": summary.get("checkpoint"),
        "config": summary.get("config"),
    }


def evaluate_run_with_workers(
    *,
    kind: str,
    run_name: str,
    run_dir: Path,
    samples: Sequence[Dict[str, Any]],
    device_names: Sequence[str],
    resume_partials: bool,
    run_signature: str,
    worker_task_factory,
) -> List[Dict[str, Any]]:
    active_devices = list(device_names) if device_names else ["cpu"]
    shards = shard_samples(samples, len(active_devices))
    tasks: List[Dict[str, Any]] = []
    reused_worker_ids: List[int] = []

    for worker_id, (device_name, shard_samples_list) in enumerate(zip(active_devices, shards)):
        if not shard_samples_list:
            continue
        part_path, meta_path = get_partial_paths(run_dir, worker_id)
        if resume_partials and reusable_partial_exists(
            part_path=part_path,
            meta_path=meta_path,
            run_signature=run_signature,
            worker_id=worker_id,
        ):
            reused_worker_ids.append(worker_id)
            continue
        tasks.append(
            worker_task_factory(
                worker_id=worker_id,
                device_name=device_name,
                samples=shard_samples_list,
                part_path=part_path,
                meta_path=meta_path,
            )
        )

    if reused_worker_ids:
        log_status(
            f"[resume-partials] run={run_name} reused worker shards {reused_worker_ids}"
        )
    if tasks:
        log_status(
            f"[evaluate] run={run_name} launching {len(tasks)} worker(s) across devices "
            f"{[task['device_name'] for task in tasks]}"
        )
    execute_parallel_tasks(tasks)

    merged_rows: List[Dict[str, Any]] = []
    for worker_id, shard_samples_list in enumerate(shards):
        if not shard_samples_list:
            continue
        part_path, meta_path = get_partial_paths(run_dir, worker_id)
        if not reusable_partial_exists(
            part_path=part_path,
            meta_path=meta_path,
            run_signature=run_signature,
            worker_id=worker_id,
        ):
            raise RuntimeError(
                f"Missing or invalid partial output for run={run_name}, worker_id={worker_id}, path={part_path}"
            )
        merged_rows.extend(load_partial_rows(part_path))

    merged_rows.sort(key=lambda row: int(row["_sample_order"]))
    return merged_rows


def main() -> None:
    args = parse_args()
    suite_cfg = load_suite_config(args.suite_config)
    output_root = ensure_output_dir(str(suite_cfg["output_root"]))
    manifest = str(suite_cfg["manifest"])
    suite_max_new_tokens = int(suite_cfg.get("max_new_tokens", 64))
    device_names = resolve_device_names(args, suite_cfg)
    requested_workers = args.num_workers or len(device_names)
    device_names = device_names[: max(min(requested_workers, len(device_names)), 1)]
    task_alias_map = resolve_task_alias_map(suite_cfg.get("task_alias_map"))
    code_version = compute_code_version()

    samples = load_jsonl(manifest)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    leaderboard_rows: List[Dict[str, Any]] = []
    suite_summary: Dict[str, Any] = {
        "suite_name": suite_cfg.get("suite_name", Path(args.suite_config).stem),
        "manifest": manifest,
        "num_samples": len(samples),
        "devices": list(device_names),
        "worker_count": len(device_names),
        "cache_mode": args.cache_mode,
        "task_alias_map": dict(task_alias_map),
        "runs": [],
    }

    baseline_cfg = dict(suite_cfg.get("baseline") or {})
    if baseline_cfg.get("enabled", False):
        run_name = baseline_cfg.get("name", "baseline_qwen3_vl")
        run_dir = ensure_output_dir(str(output_root / run_name))
        prompt_style = str(baseline_cfg.get("prompt_style", "answer_only"))
        baseline_max_new_tokens = int(baseline_cfg.get("max_new_tokens", suite_max_new_tokens))
        run_signature = build_run_signature(
            kind="baseline",
            suite_manifest=manifest,
            run_name=run_name,
            prompt_style=prompt_style,
            max_new_tokens=baseline_max_new_tokens,
            checkpoint=None,
            model_path=str(baseline_cfg["model_path"]),
            data_cfg={"image_root": baseline_cfg.get("image_root")},
            task_alias_map=task_alias_map,
            code_version=code_version,
        )

        summary = None
        if args.cache_mode == "results":
            summary = load_cached_summary(run_dir, run_signature)
            if summary is not None:
                log_status(f"[cache-hit] run={run_name} summary/predictions match signature; skipping execution")

        if summary is None:
            if len(device_names) > 1:
                rows = evaluate_run_with_workers(
                    kind="baseline",
                    run_name=run_name,
                    run_dir=run_dir,
                    samples=samples,
                    device_names=device_names,
                    resume_partials=args.resume_partials,
                    run_signature=run_signature,
                    worker_task_factory=lambda **worker_kwargs: {
                        "kind": "baseline",
                        "baseline_cfg": baseline_cfg,
                        "task_alias_map": dict(task_alias_map),
                        "max_new_tokens": baseline_max_new_tokens,
                        "run_signature": run_signature,
                        **worker_kwargs,
                    },
                )
            else:
                rows = evaluate_baseline_rows(
                    baseline_cfg=baseline_cfg,
                    samples=[{**sample, "_sample_order": idx} for idx, sample in enumerate(samples)],
                    device_name=device_names[0],
                    max_new_tokens=baseline_max_new_tokens,
                    task_alias_map=task_alias_map,
                )

            rows = strip_internal_row_fields(rows)
            summary = build_baseline_summary(
                baseline_cfg=baseline_cfg,
                suite_manifest=manifest,
                prompt_style=prompt_style,
                max_new_tokens=baseline_max_new_tokens,
                rows=rows,
            )
            write_csv(run_dir / "predictions.csv", rows)
            write_json(run_dir / "summary.json", summary)
            write_run_metadata(
                run_dir=run_dir,
                run_signature=run_signature,
                cache_mode=args.cache_mode,
                device_names=device_names,
                worker_count=len(device_names),
                num_samples=len(rows),
                max_new_tokens=baseline_max_new_tokens,
                prompt_style=prompt_style,
                checkpoint=None,
                kind="baseline",
            )

        suite_summary["runs"].append(summary)
        leaderboard_rows.append(make_leaderboard_row(summary))

    for run_cfg in suite_cfg.get("runs", []):
        if not run_cfg.get("enabled", True):
            continue

        run_name = str(run_cfg["name"])
        run_dir = ensure_output_dir(str(output_root / run_name))
        config = load_config(str(run_cfg["config"]))
        config = apply_config_overrides(config, run_cfg, manifest)
        prompt_style = str(config.get("data", {}).get("prompt_style", "reasoning"))
        run_max_new_tokens = int(run_cfg.get("max_new_tokens", suite_max_new_tokens))
        run_signature = build_run_signature(
            kind="modular",
            suite_manifest=manifest,
            run_name=run_name,
            prompt_style=prompt_style,
            max_new_tokens=run_max_new_tokens,
            checkpoint=run_cfg.get("checkpoint"),
            model_path=None,
            data_cfg=dict(config.get("data", {})),
            task_alias_map=task_alias_map,
            code_version=code_version,
        )

        summary = None
        if args.cache_mode == "results":
            summary = load_cached_summary(run_dir, run_signature)
            if summary is not None:
                log_status(f"[cache-hit] run={run_name} summary/predictions match signature; skipping execution")

        if summary is None:
            if len(device_names) > 1:
                rows = evaluate_run_with_workers(
                    kind="modular",
                    run_name=run_name,
                    run_dir=run_dir,
                    samples=samples,
                    device_names=device_names,
                    resume_partials=args.resume_partials,
                    run_signature=run_signature,
                    worker_task_factory=lambda **worker_kwargs: {
                        "kind": "modular",
                        "config": config,
                        "checkpoint": run_cfg.get("checkpoint"),
                        "task_alias_map": dict(task_alias_map),
                        "max_new_tokens": run_max_new_tokens,
                        "run_signature": run_signature,
                        **worker_kwargs,
                    },
                )
            else:
                rows = evaluate_modular_rows(
                    config=config,
                    checkpoint=run_cfg.get("checkpoint"),
                    samples=[{**sample, "_sample_order": idx} for idx, sample in enumerate(samples)],
                    device_name=device_names[0],
                    max_new_tokens=run_max_new_tokens,
                    task_alias_map=task_alias_map,
                )

            rows = strip_internal_row_fields(rows)
            summary = build_modular_summary(
                run_cfg=run_cfg,
                suite_manifest=manifest,
                config=config,
                prompt_style=prompt_style,
                max_new_tokens=run_max_new_tokens,
                rows=rows,
            )
            write_csv(run_dir / "predictions.csv", rows)
            write_json(run_dir / "summary.json", summary)
            write_run_metadata(
                run_dir=run_dir,
                run_signature=run_signature,
                cache_mode=args.cache_mode,
                device_names=device_names,
                worker_count=len(device_names),
                num_samples=len(rows),
                max_new_tokens=run_max_new_tokens,
                prompt_style=prompt_style,
                checkpoint=run_cfg.get("checkpoint"),
                kind="modular",
            )

        suite_summary["runs"].append(summary)
        leaderboard_rows.append(make_leaderboard_row(summary))

    leaderboard_rows = sorted(
        leaderboard_rows,
        key=lambda row: -float(row["microvqa_macro_accuracy"])
        if row["microvqa_macro_accuracy"] is not None and not math.isnan(row["microvqa_macro_accuracy"])
        else float("inf"),
    )
    write_csv(output_root / "leaderboard.csv", leaderboard_rows)
    write_json(output_root / "suite_summary.json", suite_summary)
    print(json.dumps(suite_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Suppress multiprocessing resource_tracker warnings (harmless semapore leak warnings)
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
    main()
