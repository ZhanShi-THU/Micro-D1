from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simplified unified MCQ suite with overall accuracy only."
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        default="evaluation/suites/unified_accuracy_suite.yaml",
        help="YAML file describing the simplified unified suite.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for all runs.",
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


def apply_config_overrides(config: Dict[str, Any], run_cfg: Dict[str, Any], manifest: str) -> Dict[str, Any]:
    updated = json.loads(json.dumps(config))
    updated.setdefault("data", {})
    updated["data"]["test_manifest"] = manifest

    if run_cfg.get("image_size") is not None:
        updated["data"]["image_size"] = int(run_cfg["image_size"])
    if run_cfg.get("image_preprocessing") is not None:
        updated["data"]["image_preprocessing"] = str(run_cfg["image_preprocessing"])
    if run_cfg.get("prompt_style") is not None:
        updated["data"]["prompt_style"] = str(run_cfg["prompt_style"])
    if run_cfg.get("max_text_length") is not None:
        updated["data"]["max_text_length"] = int(run_cfg["max_text_length"])
    if run_cfg.get("image_root") is not None:
        updated["data"]["image_root"] = run_cfg["image_root"]
    return updated


def evaluate_modular_run(
    *,
    run_cfg: Dict[str, Any],
    suite_manifest: str,
    samples: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    config = load_config(str(run_cfg["config"]))
    config = apply_config_overrides(config, run_cfg, suite_manifest)
    image_root = config.get("data", {}).get("image_root")
    prompt_style = str(config.get("data", {}).get("prompt_style", "reasoning"))

    evaluator = ModularVLMEvaluator(
        config=config,
        device=device,
        adapter_checkpoint=run_cfg.get("checkpoint"),
    )

    rows: List[Dict[str, Any]] = []
    parse_success = 0
    for sample in samples:
        image = resolve_image(sample, image_root=image_root)
        prompt = build_microvqa_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=prompt_style,
        )
        response = evaluator.generate(image, prompt, max_new_tokens)
        pred = parse_choice_answer(response)
        answer = int(sample["correct_index"])
        parse_success += int(pred is not None)
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "split": sample.get("split"),
                "question": sample["question"],
                "correct_index": answer,
                "prediction_index": pred,
                "response": response,
                "correct": pred == answer,
            }
        )

    mcq_summary = build_mcq_summary(rows, {"modular_vlm": "correct"})
    summary = {
        "run_name": run_cfg["name"],
        "config": str(run_cfg["config"]),
        "checkpoint": run_cfg.get("checkpoint"),
        "manifest": suite_manifest,
        "prompt_style": prompt_style,
        "image_preprocessing": config["data"].get("image_preprocessing"),
        "image_size": config["data"].get("image_size"),
        "parse_success_rate": parse_success / len(rows) if rows else math.nan,
        "mcq_metrics": mcq_summary["models"]["modular_vlm"],
        "num_samples": len(rows),
    }
    return {"rows": rows, "summary": summary}


def evaluate_baseline_run(
    *,
    baseline_cfg: Dict[str, Any],
    suite_manifest: str,
    samples: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    image_root = baseline_cfg.get("image_root")
    prompt_style = str(baseline_cfg.get("prompt_style", "answer_only"))
    baseline_max_new_tokens = int(baseline_cfg.get("max_new_tokens", max_new_tokens))
    evaluator = BaselineQwenEvaluator(
        model_path=str(baseline_cfg["model_path"]),
        device=device,
    )

    rows: List[Dict[str, Any]] = []
    parse_success = 0
    for sample in samples:
        image = resolve_image(sample, image_root=image_root)
        prompt = build_baseline_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=prompt_style,
        )
        response = evaluator.generate(image, prompt, baseline_max_new_tokens)
        pred = parse_choice_answer(response)
        answer = int(sample["correct_index"])
        parse_success += int(pred is not None)
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "split": sample.get("split"),
                "question": sample["question"],
                "correct_index": answer,
                "prediction_index": pred,
                "response": response,
                "correct": pred == answer,
            }
        )

    mcq_summary = build_mcq_summary(rows, {"baseline_qwen3_vl": "correct"})
    summary = {
        "run_name": baseline_cfg.get("name", "baseline_qwen3_vl"),
        "model_path": str(baseline_cfg["model_path"]),
        "manifest": suite_manifest,
        "prompt_style": prompt_style,
        "max_new_tokens": baseline_max_new_tokens,
        "parse_success_rate": parse_success / len(rows) if rows else math.nan,
        "mcq_metrics": mcq_summary["models"]["baseline_qwen3_vl"],
        "num_samples": len(rows),
    }
    return {"rows": rows, "summary": summary}


def make_leaderboard_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(summary["mcq_metrics"])
    return {
        "run_name": summary["run_name"],
        "overall_accuracy": metrics.get("overall_accuracy"),
        "num_correct": metrics.get("num_correct"),
        "num_samples": metrics.get("num_samples"),
        "parse_success_rate": summary.get("parse_success_rate"),
        "checkpoint": summary.get("checkpoint"),
        "config": summary.get("config"),
    }


def main() -> None:
    args = parse_args()
    suite_cfg = load_suite_config(args.suite_config)
    output_root = ensure_output_dir(str(suite_cfg["output_root"]))
    manifest = str(suite_cfg["manifest"])
    max_new_tokens = int(suite_cfg.get("max_new_tokens", 64))
    device = choose_device(args.device or suite_cfg.get("device"))

    samples = load_jsonl(manifest)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    leaderboard_rows: List[Dict[str, Any]] = []
    suite_summary: Dict[str, Any] = {
        "suite_name": suite_cfg.get("suite_name", Path(args.suite_config).stem),
        "manifest": manifest,
        "num_samples": len(samples),
        "device": str(device),
        "runs": [],
    }

    baseline_cfg = dict(suite_cfg.get("baseline") or {})
    if baseline_cfg.get("enabled", False):
        baseline_result = evaluate_baseline_run(
            baseline_cfg=baseline_cfg,
            suite_manifest=manifest,
            samples=samples,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        baseline_dir = ensure_output_dir(str(output_root / baseline_cfg.get("name", "baseline_qwen3_vl")))
        write_csv(baseline_dir / "predictions.csv", baseline_result["rows"])
        write_json(baseline_dir / "summary.json", baseline_result["summary"])
        suite_summary["runs"].append(baseline_result["summary"])
        leaderboard_rows.append(make_leaderboard_row(baseline_result["summary"]))

    for run_cfg in suite_cfg.get("runs", []):
        if not run_cfg.get("enabled", True):
            continue
        run_result = evaluate_modular_run(
            run_cfg=run_cfg,
            suite_manifest=manifest,
            samples=samples,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        run_dir = ensure_output_dir(str(output_root / str(run_cfg["name"])))
        write_csv(run_dir / "predictions.csv", run_result["rows"])
        write_json(run_dir / "summary.json", run_result["summary"])
        suite_summary["runs"].append(run_result["summary"])
        leaderboard_rows.append(make_leaderboard_row(run_result["summary"]))

    leaderboard_rows = sorted(
        leaderboard_rows,
        key=lambda row: -float(row["overall_accuracy"])
        if row["overall_accuracy"] is not None and not math.isnan(row["overall_accuracy"])
        else float("inf"),
    )
    write_csv(output_root / "leaderboard.csv", leaderboard_rows)
    write_json(output_root / "suite_summary.json", suite_summary)
    print(json.dumps(suite_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
