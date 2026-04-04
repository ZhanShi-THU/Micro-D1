#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VARIANTS: List[Dict[str, Any]] = [
    {
        "key": "full",
        "display_name": "Full Model (Micro-D1)",
        "description": "Reference full model with all components enabled.",
        "model_updates": {
            "use_deepstack_injection": True,
            "alignment_head_type": "dinotxt",
            "adapter_type": "mlp",
        },
    },
    {
        "key": "no_deepstack",
        "display_name": "w/o DeepStack",
        "description": "Disable DeepStack reinjection while keeping alignment and MLP adapter.",
        "model_updates": {
            "use_deepstack_injection": False,
            "alignment_head_type": "dinotxt",
            "adapter_type": "mlp",
        },
    },
    {
        "key": "no_alignment",
        "display_name": "w/o alignment head (dino-txt)",
        "description": "Replace the learned dinotxt alignment head with identity passthrough.",
        "model_updates": {
            "use_deepstack_injection": True,
            "alignment_head_type": "identity",
            "adapter_type": "mlp",
            "alignment_head_weights": None,
        },
    },
    {
        "key": "linear_adapter",
        "display_name": "w/o adapter (linear)",
        "description": "Replace the nonlinear MLP adapter with a single linear projection.",
        "model_updates": {
            "use_deepstack_injection": True,
            "alignment_head_type": "dinotxt",
            "adapter_type": "linear",
        },
    },
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def resolve_variant_run_name(prefix: str, variant_key: str) -> str:
    return f"{prefix}_{variant_key}"


def resolve_variant_config_name(base_name: str, variant_key: str) -> str:
    stem = Path(base_name).stem
    return f"{stem}_{variant_key}.yaml"


def build_variant_config(
    *,
    base_config: Dict[str, Any],
    variant: Dict[str, Any],
    run_name: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["project_name"] = run_name
    cfg.setdefault("model", {})
    cfg["model"].update(variant["model_updates"])
    cfg["model"]["ablation_variant"] = variant["key"]

    cfg.setdefault("training", {})
    cfg["training"]["run_name"] = run_name
    wandb_cfg = dict(cfg["training"].get("wandb") or {})
    if wandb_cfg.get("enabled", False):
        wandb_cfg["name"] = run_name
    cfg["training"]["wandb"] = wandb_cfg
    return cfg


def build_suite_run_entry(
    *,
    suite_template: Dict[str, Any],
    variant: Dict[str, Any],
    config_relpath: str,
    checkpoint_abspath: str,
) -> Dict[str, Any]:
    base_run = {
        "name": variant["key"],
        "enabled": True,
        "config": config_relpath,
        "checkpoint": checkpoint_abspath,
        "prompt_style": str(suite_template.get("prompt_style") or "reasoning"),
        "image_preprocessing": suite_template.get("image_preprocessing"),
        "dynamic_buckets": suite_template.get("dynamic_buckets"),
        "patch_size": suite_template.get("patch_size"),
        "image_size": suite_template.get("image_size"),
        "max_text_length": suite_template.get("max_text_length"),
    }
    return {key: value for key, value in base_run.items() if value is not None}


def choose_phase3_run_template(base_suite: Dict[str, Any]) -> Dict[str, Any]:
    runs = list(base_suite.get("runs") or [])
    for run in runs:
        name = str(run.get("name", "")).lower()
        config = str(run.get("config", "")).lower()
        if "phase3" in name or "phase3" in config:
            return dict(run)
    return dict(runs[-1]) if runs else {}


def prepare(args: argparse.Namespace) -> None:
    base_config_path = (PROJECT_ROOT / args.base_config).resolve()
    base_suite_path = (PROJECT_ROOT / args.base_suite).resolve()
    config_output_dir = (PROJECT_ROOT / args.config_output_dir).resolve()
    suite_output_path = (PROJECT_ROOT / args.suite_output).resolve()
    commands_output_path = (PROJECT_ROOT / args.commands_output).resolve()

    base_config = load_yaml(base_config_path)
    base_suite = load_yaml(base_suite_path)
    base_training_output_dir = Path(base_config["training"]["output_dir"])

    generated_runs: List[Dict[str, Any]] = []
    training_commands: List[str] = []

    for variant in VARIANTS:
        run_name = resolve_variant_run_name(args.run_prefix, variant["key"])
        cfg = build_variant_config(
            base_config=base_config,
            variant=variant,
            run_name=run_name,
        )
        config_name = resolve_variant_config_name(base_config_path.name, variant["key"])
        config_path = config_output_dir / config_name
        dump_yaml(config_path, cfg)

        if variant["key"] == "full" and args.full_checkpoint:
            checkpoint_path = Path(args.full_checkpoint).resolve()
            config_relpath = args.full_config or args.base_config
        else:
            checkpoint_path = (PROJECT_ROOT / base_training_output_dir / run_name / args.checkpoint_name).resolve()
            config_relpath = str(config_path.relative_to(PROJECT_ROOT))
        run_entry = build_suite_run_entry(
            suite_template=choose_phase3_run_template(base_suite),
            variant=variant,
            config_relpath=config_relpath,
            checkpoint_abspath=str(checkpoint_path),
        )
        generated_runs.append(run_entry)

        if variant["key"] == "full" and args.full_checkpoint:
            training_commands.append(
                "\n".join(
                    [
                        f"# {variant['display_name']}",
                        f"# Reusing existing full-model checkpoint: {args.full_checkpoint}",
                    ]
                )
            )
        else:
            training_commands.append(
                "\n".join(
                    [
                        f"# {variant['display_name']}",
                        "accelerate launch "
                        f"--num_processes {args.num_processes} "
                        "train_phase3.py "
                        f"--config {config_path.relative_to(PROJECT_ROOT)} "
                        f"--phase2-checkpoint {args.phase2_checkpoint}",
                    ]
                )
            )

    suite_payload = {
        "suite_name": args.suite_name,
        "output_root": args.eval_output_root,
        "manifest": base_suite["manifest"],
        "device": args.device or base_suite.get("device", "cuda"),
        "max_new_tokens": int(args.max_new_tokens or base_suite.get("max_new_tokens", 500)),
        "task_alias_map": base_suite.get("task_alias_map", {}),
        "baseline": {
            "enabled": False,
        },
        "runs": generated_runs,
    }
    dump_yaml(suite_output_path, suite_payload)

    commands_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        "# Evaluate all ablation runs",
        "python3 evaluation/run_microvqa_suite.py",
        f"--suite-config {suite_output_path.relative_to(PROJECT_ROOT)}",
    ]
    if args.eval_devices:
        eval_cmd.append(f"--devices {args.eval_devices}")
    if args.eval_num_workers is not None:
        eval_cmd.append(f"--num-workers {args.eval_num_workers}")
    eval_cmd.append("--cache-mode off")

    commands_text = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated Phase 3 component ablation commands.",
        "",
    ]
    for command in training_commands:
        commands_text.append(command)
        commands_text.append("")
    commands_text.append(eval_cmd[0])
    commands_text.append(" ".join(eval_cmd[1:]))
    commands_output_path.write_text("\n".join(commands_text) + "\n", encoding="utf-8")
    commands_output_path.chmod(0o755)

    print(json.dumps(
        {
            "generated_configs": [
                str((config_output_dir / resolve_variant_config_name(base_config_path.name, variant["key"])).relative_to(PROJECT_ROOT))
                for variant in VARIANTS
            ],
            "suite_config": str(suite_output_path.relative_to(PROJECT_ROOT)),
            "commands_script": str(commands_output_path.relative_to(PROJECT_ROOT)),
        },
        ensure_ascii=False,
        indent=2,
    ))


def load_summary_metrics(summary_path: Path) -> Dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alias_accuracy = summary["microvqa_task_metrics"]["alias_accuracy"]
    return {
        "EU": float(alias_accuracy["EU"]) * 100.0,
        "HG": float(alias_accuracy["HG"]) * 100.0,
        "EP": float(alias_accuracy["EP"]) * 100.0,
        "Avg": float(summary["mcq_metrics"]["overall_accuracy"]) * 100.0,
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def summarize(args: argparse.Namespace) -> None:
    suite_path = (PROJECT_ROOT / args.suite_config).resolve()
    suite_cfg = load_yaml(suite_path)
    output_root = (PROJECT_ROOT / suite_cfg["output_root"]).resolve()

    rows: List[Dict[str, Any]] = []
    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"  \hline",
        r"  Model Variant & EU & HG & EP & Avg \\",
        r"  \hline",
    ]

    for variant in VARIANTS:
        run_name = variant["key"]
        summary_path = output_root / run_name / "summary.json"
        metrics: Dict[str, float | None]
        if summary_path.exists():
            metrics = load_summary_metrics(summary_path)
        else:
            metrics = {"EU": None, "HG": None, "EP": None, "Avg": None}

        row = {
            "variant_key": variant["key"],
            "model_variant": variant["display_name"],
            "summary_path": str(summary_path),
            "EU": metrics["EU"],
            "HG": metrics["HG"],
            "EP": metrics["EP"],
            "Avg": metrics["Avg"],
        }
        rows.append(row)
        latex_lines.append(
            "  "
            + f"{variant['display_name']} & "
            + f"{format_metric(metrics['EU'])} & "
            + f"{format_metric(metrics['HG'])} & "
            + f"{format_metric(metrics['EP'])} & "
            + f"{format_metric(metrics['Avg'])} "
            + r"\\"
        )

    latex_lines.extend(
        [
            r"  \hline",
            r"\end{tabular}",
        ]
    )

    csv_output_path = (PROJECT_ROOT / args.csv_output).resolve()
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    latex_output_path = (PROJECT_ROOT / args.latex_output).resolve()
    latex_output_path.parent.mkdir(parents=True, exist_ok=True)
    latex_output_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "csv": str(csv_output_path.relative_to(PROJECT_ROOT)),
            "latex": str(latex_output_path.relative_to(PROJECT_ROOT)),
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and summarize Phase 3 component ablation studies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--base-config",
        type=str,
        default="configs/phase3_qwen3_dinov3_qwen_hybrid_lora128.yaml",
    )
    prepare_parser.add_argument(
        "--base-suite",
        type=str,
        default="evaluation/suites/microvqa_gold_detailed.yaml",
    )
    prepare_parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        required=True,
    )
    prepare_parser.add_argument(
        "--config-output-dir",
        type=str,
        default="configs/ablations/phase3_component_ablation",
    )
    prepare_parser.add_argument(
        "--suite-output",
        type=str,
        default="evaluation/suites/phase3_component_ablation.yaml",
    )
    prepare_parser.add_argument(
        "--commands-output",
        type=str,
        default="scripts/generated/phase3_component_ablation.sh",
    )
    prepare_parser.add_argument(
        "--run-prefix",
        type=str,
        default="phase3_component_ablation",
    )
    prepare_parser.add_argument(
        "--suite-name",
        type=str,
        default="phase3_component_ablation",
    )
    prepare_parser.add_argument(
        "--eval-output-root",
        type=str,
        default="./eval_outputs/phase3_component_ablation",
    )
    prepare_parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="phase3_best_loss.pt",
    )
    prepare_parser.add_argument(
        "--full-checkpoint",
        type=str,
        default=None,
        help="Optional existing full-model checkpoint to reuse instead of retraining the full variant.",
    )
    prepare_parser.add_argument(
        "--full-config",
        type=str,
        default=None,
        help="Config path paired with --full-checkpoint. Defaults to --base-config.",
    )
    prepare_parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
    )
    prepare_parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    prepare_parser.add_argument(
        "--eval-devices",
        type=str,
        default=None,
    )
    prepare_parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
    )
    prepare_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument(
        "--suite-config",
        type=str,
        default="evaluation/suites/phase3_component_ablation.yaml",
    )
    summarize_parser.add_argument(
        "--csv-output",
        type=str,
        default="paper_drafts/generated/phase3_component_ablation.csv",
    )
    summarize_parser.add_argument(
        "--latex-output",
        type=str,
        default="paper_drafts/generated/phase3_component_ablation_table.tex",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "summarize":
        summarize(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
