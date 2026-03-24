from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import torch
import yaml
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import build_microvqa_prompt
from data.image_transforms import build_image_transform as build_configurable_image_transform
from data.unified_vqa import (
    build_multiple_choice_prompt,
    format_choices_for_prompt,
    get_unified_manifest_paths,
)
from models.modular_vlm import ModularVLM


ANSWER_REGEX = re.compile(
    r"(?:^|\b)(?:the\s+)?answer(?:\s+is|:)?\s*\*?\*?\(?([0-9]+)\)?|^\s*\(([0-9]+)\)",
    re.IGNORECASE,
)

CAPTION_PROMPT = "Describe the image in one sentence with fine-grained details."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_dinov3.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device in config.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    mcq_parser = subparsers.add_parser("mcq")
    mcq_parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Defaults to data.test_manifest in the config.",
    )
    mcq_parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default=None,
        help="Checkpoint created by train.py. If omitted, the adapter stays at current weights.",
    )
    mcq_parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="/home/user/Project_files/microvqa/models/qwen3-vl-2b",
    )
    mcq_parser.add_argument("--output-dir", type=str, default="./eval_outputs/mcq")
    mcq_parser.add_argument("--max-new-tokens", type=int, default=64)
    mcq_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of evaluation samples.",
    )

    fine_parser = subparsers.add_parser("finegrained")
    fine_parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Defaults to data.test_manifest in the config.",
    )
    fine_parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default=None,
        help="Checkpoint created by train.py. If omitted, the adapter stays at current weights.",
    )
    fine_parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="/home/user/Project_files/microvqa/models/qwen3-vl-2b",
    )
    fine_parser.add_argument("--output-dir", type=str, default="./eval_outputs/finegrained")
    fine_parser.add_argument("--max-new-tokens", type=int, default=64)

    efficiency_parser = subparsers.add_parser("efficiency")
    efficiency_parser.add_argument("--manifest", type=str, required=True)
    efficiency_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing phase1/phase2 checkpoints from train.py.",
    )
    efficiency_parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="/home/user/Project_files/microvqa/models/qwen3-vl-2b",
    )
    efficiency_parser.add_argument("--output-dir", type=str, default="./eval_outputs/efficiency")
    efficiency_parser.add_argument("--max-new-tokens", type=int, default=64)
    efficiency_parser.add_argument(
        "--target-token-f1",
        type=float,
        default=0.50,
        help="First checkpoint reaching this score is reported as convergence step.",
    )
    efficiency_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of evaluation samples.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_eval_device(device_name: str | None) -> torch.device:
    resolved = device_name or "cuda"
    if resolved == "cuda" and not torch.cuda.is_available():
        resolved = "cpu"
    if resolved == "cuda":
        resolved = "cuda:0"
    device = torch.device(resolved)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def build_image_transform(image_size: int, preprocessing: str = "resize"):
    return build_configurable_image_transform(image_size, preprocessing=preprocessing)


def resolve_prompt_style(config: Dict[str, Any]) -> str:
    return str(config.get("data", {}).get("prompt_style", "reasoning"))


def maybe_apply_configured_lora(model: ModularVLM, config: Dict[str, Any]) -> None:
    lora_cfg = None
    for section_name in ("phase3", "phase2"):
        section_cfg = config.get(section_name, {})
        candidate = section_cfg.get("lora")
        if candidate and candidate.get("enabled", True):
            lora_cfg = dict(candidate)
            break

    if lora_cfg is None:
        return

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Evaluating Phase 2/3 checkpoints with LoRA weights requires 'peft'."
        ) from exc

    model.llm = get_peft_model(
        model.llm,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 128)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(lora_cfg.get("bias", "none")),
        ),
    )
    model.refresh_llm_references()


def load_jsonl(manifest_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_mcq_manifest(config: Dict[str, Any], manifest: str | None) -> str:
    if manifest:
        return manifest
    data_cfg = config.get("data", {})
    configured = data_cfg.get("test_manifest")
    if configured:
        return str(configured)
    return str(get_unified_manifest_paths(data_cfg.get("unified_root"))["test"])


def resolve_image(sample: Dict[str, Any], image_root: str | None = None) -> Image.Image:
    if "images_list" in sample:
        images_list = sample["images_list"]
        if not images_list:
            raise ValueError("images_list cannot be empty.")
        first_image = images_list[0]
        if isinstance(first_image, str):
            return open_image(first_image, image_root=image_root)
        if isinstance(first_image, dict) and "path" in first_image:
            return open_image(first_image["path"], image_root=image_root)
        raise TypeError("Only path-based images_list entries are supported.")

    image_path = sample.get("image") or sample.get("image_path")
    if image_path is None:
        raise KeyError("Sample is missing image information.")
    return open_image(image_path, image_root=image_root)


def open_image(image_path: str, image_root: str | None = None) -> Image.Image:
    path = Path(image_path)
    if not path.is_absolute() and image_root is not None:
        path = Path(image_root) / path
    return Image.open(path).convert("RGB")


def format_choices(choices: Sequence[str]) -> str:
    return format_choices_for_prompt(choices)


def build_baseline_prompt(
    question: str,
    choices: Sequence[str],
    prompt_style: str = "answer_only",
) -> str:
    # The baseline evaluator already passes the image through the processor,
    # so we avoid embedding a literal <image> token in the text prompt.
    return build_multiple_choice_prompt(question, choices, prompt_style=prompt_style)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_choice_answer(text: str) -> int | None:
    match = ANSWER_REGEX.search(text)
    if match is None:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def infer_finegrained_bucket(sample: Dict[str, Any]) -> str:
    task = str(sample.get("task", "")).lower()
    question = str(sample.get("question", "")).lower()
    if "count" in task or "how many" in question or "几只" in question:
        return "object_count"
    if "attribute" in task or any(
        phrase in question
        for phrase in ("what color", "what pattern", "what texture", "什么颜色", "什么纹理")
    ):
        return "attribute"
    return "other"


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class ModularVLMEvaluator:
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        adapter_checkpoint: str | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.model = ModularVLM(config)
        self.model = self.model.prepare_for_training_device(device)
        maybe_apply_configured_lora(self.model, config)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["llm_base"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        image_size = config["data"]["image_size"]
        preprocessing = str(config["data"].get("image_preprocessing", "resize"))
        self.image_transform = build_image_transform(image_size, preprocessing=preprocessing)

        if adapter_checkpoint:
            self.load_checkpoint(adapter_checkpoint)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "adapter" in checkpoint:
            self.model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
        elif "model" in checkpoint:
            self.model.adapter.load_state_dict(checkpoint["model"], strict=True)
        else:
            raise KeyError("Checkpoint is missing 'adapter' or 'model' state.")

        if "vision_alignment_head" in checkpoint:
            self.model.vision_encoder.alignment_head.load_state_dict(
                checkpoint["vision_alignment_head"],
                strict=True,
            )

        backbone_state = checkpoint.get("vision_backbone_top_blocks")
        if backbone_state:
            missing_keys, unexpected_keys = self.model.vision_encoder.backbone.load_state_dict(
                backbone_state,
                strict=False,
            )
            if unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys when loading backbone state from {checkpoint_path}: {unexpected_keys}"
                )
            _ = missing_keys

        llm_state = checkpoint.get("llm_trainable_state")
        if llm_state:
            self.model.llm.load_state_dict(llm_state, strict=False)

        lora_state = checkpoint.get("lora_state")
        if lora_state:
            try:
                from peft import set_peft_model_state_dict
            except ImportError as exc:
                raise ImportError(
                    "Loading a checkpoint with LoRA weights requires 'peft'."
                ) from exc
            set_peft_model_state_dict(self.model.llm, lora_state)

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        pixel_values = self.image_transform(image).unsqueeze(0).to(self.device)
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        model_inputs = self.model.build_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        inputs_embeds = model_inputs["inputs_embeds"]
        merged_attention_mask = model_inputs["attention_mask"]
        visual_pos_masks = model_inputs.get("visual_pos_masks")
        deepstack_visual_embeds = model_inputs.get("deepstack_visual_embeds")

        generated_ids: List[int] = []
        for _ in range(max_new_tokens):
            decoder_outputs = self.model.llm_body.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=merged_attention_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            hidden_states = decoder_outputs[0]
            logits = self.model.lm_head(hidden_states)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            token_id = int(next_token_id.item())

            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            next_embed = self.model.get_text_embeds(next_token_id.unsqueeze(0))
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            next_mask = torch.ones((1, 1), dtype=merged_attention_mask.dtype, device=self.device)
            merged_attention_mask = torch.cat([merged_attention_mask, next_mask], dim=1)
            if visual_pos_masks is not None:
                next_visual_mask = torch.zeros((visual_pos_masks.size(0), 1), dtype=torch.bool, device=self.device)
                visual_pos_masks = torch.cat([visual_pos_masks, next_visual_mask], dim=1)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class BaselineQwenEvaluator:
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=None,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        generated = outputs[:, inputs.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_mcq_summary(
    rows: Sequence[Dict[str, Any]],
    model_key_to_correct_column: Dict[str, str],
) -> Dict[str, Any]:
    dataset_counts: Dict[str, int] = {}
    for row in rows:
        dataset_name = str(row["source_dataset"])
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    summary_models: Dict[str, Any] = {}
    for model_name, correct_column in model_key_to_correct_column.items():
        total_correct = sum(int(bool(row[correct_column])) for row in rows)
        per_dataset_accuracy: Dict[str, float] = {}
        for dataset_name, count in dataset_counts.items():
            dataset_rows = [row for row in rows if row["source_dataset"] == dataset_name]
            correct = sum(int(bool(row[correct_column])) for row in dataset_rows)
            per_dataset_accuracy[dataset_name] = correct / count if count else math.nan

        macro_accuracy = (
            sum(per_dataset_accuracy.values()) / len(per_dataset_accuracy)
            if per_dataset_accuracy
            else math.nan
        )
        summary_models[model_name] = {
            "overall_accuracy": total_correct / len(rows) if rows else math.nan,
            "num_correct": total_correct,
            "num_samples": len(rows),
            "macro_accuracy_by_dataset": macro_accuracy,
            "per_dataset_accuracy": per_dataset_accuracy,
        }

    return {
        "num_samples": len(rows),
        "dataset_counts": dataset_counts,
        "models": summary_models,
    }


def evaluate_finegrained(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    manifest_path = resolve_mcq_manifest(config, args.manifest)
    samples = load_jsonl(manifest_path)
    image_root = config["data"].get("image_root")

    modular_model = ModularVLMEvaluator(
        config=config,
        device=device,
        adapter_checkpoint=args.adapter_checkpoint,
    )
    baseline_model = BaselineQwenEvaluator(args.baseline_model_path, device=device)

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, float]] = {
        "modular_vlm": {"overall": 0.0, "object_count": 0.0, "attribute": 0.0},
        "baseline_qwen3_vl": {"overall": 0.0, "object_count": 0.0, "attribute": 0.0},
    }
    counts = {
        "overall": 0,
        "object_count": 0,
        "attribute": 0,
    }
    corrects = {
        "modular_vlm": {"overall": 0, "object_count": 0, "attribute": 0},
        "baseline_qwen3_vl": {"overall": 0, "object_count": 0, "attribute": 0},
    }

    for sample in samples:
        if "question" not in sample or "choices" not in sample or "correct_index" not in sample:
            raise KeyError(
                "Fine-grained evaluation manifest must contain question, choices, correct_index, and image fields."
            )

        image = resolve_image(sample, image_root=image_root)
        modular_prompt = build_microvqa_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=resolve_prompt_style(config),
        )
        baseline_prompt = build_baseline_prompt(sample["question"], sample["choices"])

        modular_text = modular_model.generate(image, modular_prompt, args.max_new_tokens)
        baseline_text = baseline_model.generate(image, baseline_prompt, args.max_new_tokens)

        modular_pred = parse_choice_answer(modular_text)
        baseline_pred = parse_choice_answer(baseline_text)
        answer = int(sample["correct_index"])
        bucket = infer_finegrained_bucket(sample)

        row = {
            "key_question": sample.get("key_question"),
            "key_image": sample.get("key_image"),
            "task": sample.get("task"),
            "bucket": bucket,
            "question": sample["question"],
            "correct_index": answer,
            "modular_pred": modular_pred,
            "baseline_pred": baseline_pred,
            "modular_response": modular_text,
            "baseline_response": baseline_text,
            "modular_correct": modular_pred == answer,
            "baseline_correct": baseline_pred == answer,
        }
        rows.append(row)

        counts["overall"] += 1
        if bucket in counts:
            counts[bucket] += 1
        if modular_pred == answer:
            corrects["modular_vlm"]["overall"] += 1
            if bucket in corrects["modular_vlm"]:
                corrects["modular_vlm"][bucket] += 1
        if baseline_pred == answer:
            corrects["baseline_qwen3_vl"]["overall"] += 1
            if bucket in corrects["baseline_qwen3_vl"]:
                corrects["baseline_qwen3_vl"][bucket] += 1

    for model_name, stats in corrects.items():
        for bucket, correct in stats.items():
            denom = counts.get(bucket, 0)
            summary[model_name][bucket] = correct / denom if denom > 0 else math.nan

    output_dir = ensure_output_dir(args.output_dir)
    write_csv(output_dir / "finegrained_predictions.csv", rows)
    write_json(
        output_dir / "finegrained_summary.json",
        {
            "manifest": manifest_path,
            "bucket_metrics": summary,
            "mcq_metrics": build_mcq_summary(
                rows,
                {
                    "modular_vlm": "modular_correct",
                    "baseline_qwen3_vl": "baseline_correct",
                },
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_mcq(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    manifest_path = resolve_mcq_manifest(config, args.manifest)
    samples = load_jsonl(manifest_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    image_root = config["data"].get("image_root")
    modular_model = ModularVLMEvaluator(
        config=config,
        device=device,
        adapter_checkpoint=args.adapter_checkpoint,
    )
    baseline_model = BaselineQwenEvaluator(args.baseline_model_path, device=device)

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        if "question" not in sample or "choices" not in sample or "correct_index" not in sample:
            raise KeyError(
                "MCQ evaluation manifest must contain question, choices, correct_index, and image fields."
            )

        image = resolve_image(sample, image_root=image_root)
        modular_prompt = build_microvqa_prompt(
            sample["question"],
            sample["choices"],
            prompt_style=resolve_prompt_style(config),
        )
        baseline_prompt = build_baseline_prompt(sample["question"], sample["choices"])

        modular_text = modular_model.generate(image, modular_prompt, args.max_new_tokens)
        baseline_text = baseline_model.generate(image, baseline_prompt, args.max_new_tokens)

        modular_pred = parse_choice_answer(modular_text)
        baseline_pred = parse_choice_answer(baseline_text)
        answer = int(sample["correct_index"])

        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "split": sample.get("split"),
                "question": sample["question"],
                "correct_index": answer,
                "modular_pred": modular_pred,
                "baseline_pred": baseline_pred,
                "modular_response": modular_text,
                "baseline_response": baseline_text,
                "modular_correct": modular_pred == answer,
                "baseline_correct": baseline_pred == answer,
            }
        )

    summary = {
        "manifest": manifest_path,
        **build_mcq_summary(
            rows,
            {
                "modular_vlm": "modular_correct",
                "baseline_qwen3_vl": "baseline_correct",
            },
        ),
    }

    output_dir = ensure_output_dir(args.output_dir)
    write_csv(output_dir / "mcq_predictions.csv", rows)
    write_json(output_dir / "mcq_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_step_from_checkpoint(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"Unable to parse step from checkpoint name: {path.name}")
    return int(match.group(1))


def iter_checkpoints(checkpoint_dir: str) -> List[Path]:
    paths = sorted(Path(checkpoint_dir).glob("*.pt"), key=parse_step_from_checkpoint)
    if not paths:
        raise FileNotFoundError(f"No checkpoints were found in {checkpoint_dir}")
    return paths


def evaluate_caption_set(
    evaluator,
    samples: Sequence[Dict[str, Any]],
    image_root: str | None,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_f1 = 0.0
    total_exact = 0

    for sample in samples:
        image = resolve_image(sample, image_root=image_root)
        reference = sample.get("target_text") or sample.get("caption") or sample.get("text")
        if reference is None:
            raise KeyError("Caption evaluation sample is missing target_text/caption/text.")

        prediction = evaluator.generate(image, CAPTION_PROMPT, max_new_tokens)
        score = token_f1(prediction, reference)
        exact = normalize_text(prediction) == normalize_text(reference)

        rows.append(
            {
                "image": sample.get("image") or sample.get("image_path"),
                "reference": reference,
                "prediction": prediction,
                "token_f1": score,
                "exact_match": exact,
            }
        )
        total_f1 += score
        total_exact += int(exact)

    count = len(rows)
    return {
        "rows": rows,
        "metrics": {
            "token_f1": total_f1 / count if count else math.nan,
            "exact_match": total_exact / count if count else math.nan,
            "num_samples": count,
        },
    }


def read_training_log_steps(checkpoint_dir: str) -> List[Dict[str, Any]]:
    log_path = Path(checkpoint_dir) / "train_log.jsonl"
    if not log_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def evaluate_efficiency(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    samples = load_jsonl(args.manifest)
    if args.limit > 0:
        samples = samples[: args.limit]

    image_root = config["data"].get("image_root")
    output_dir = ensure_output_dir(args.output_dir)

    baseline_model = BaselineQwenEvaluator(args.baseline_model_path, device=device)
    baseline_eval = evaluate_caption_set(
        evaluator=baseline_model,
        samples=samples,
        image_root=image_root,
        max_new_tokens=args.max_new_tokens,
    )
    write_csv(output_dir / "baseline_caption_predictions.csv", baseline_eval["rows"])

    checkpoints = iter_checkpoints(args.checkpoint_dir)
    checkpoint_metrics: List[Dict[str, Any]] = []
    convergence_step: int | None = None

    for checkpoint_path in checkpoints:
        step = parse_step_from_checkpoint(checkpoint_path)
        modular_model = ModularVLMEvaluator(
            config=config,
            device=device,
            adapter_checkpoint=str(checkpoint_path),
        )
        result = evaluate_caption_set(
            evaluator=modular_model,
            samples=samples,
            image_root=image_root,
            max_new_tokens=args.max_new_tokens,
        )

        metrics = result["metrics"]
        metrics_row = {
            "checkpoint": str(checkpoint_path),
            "step": step,
            "token_f1": metrics["token_f1"],
            "exact_match": metrics["exact_match"],
            "num_samples": metrics["num_samples"],
        }
        checkpoint_metrics.append(metrics_row)
        write_csv(output_dir / f"caption_predictions_step_{step}.csv", result["rows"])

        if convergence_step is None and metrics["token_f1"] >= args.target_token_f1:
            convergence_step = step

    baseline_token_f1 = baseline_eval["metrics"]["token_f1"]
    step_to_match_baseline = None
    for row in checkpoint_metrics:
        if row["token_f1"] >= baseline_token_f1:
            step_to_match_baseline = row["step"]
            break

    summary = {
        "baseline_qwen3_vl": baseline_eval["metrics"],
        "modular_vlm_checkpoints": checkpoint_metrics,
        "convergence": {
            "target_token_f1": args.target_token_f1,
            "first_step_reaching_target": convergence_step,
            "first_step_matching_or_exceeding_baseline": step_to_match_baseline,
        },
        "training_log_records": read_training_log_steps(args.checkpoint_dir),
    }

    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint_metrics)
    write_json(output_dir / "efficiency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device or config["training"]["device"]
    device = resolve_eval_device(device_name)

    if args.command == "mcq":
        evaluate_mcq(args, config, device)
        return

    if args.command == "finegrained":
        evaluate_finegrained(args, config, device)
        return

    if args.command == "efficiency":
        evaluate_efficiency(args, config, device)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
