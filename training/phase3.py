"""
Phase 3 microscopy Unified VQA training.

This stage trains:
  - top DINOv3 backbone blocks
  - dinotxt alignment head
  - visual adapter
  - Qwen LoRA parameters

It expects a prior Phase 2 checkpoint as initialization and optimizes
domain-specific multiple-choice microscopy performance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.dataset import ImageTextDataset
from data.image_transforms import build_image_transform
from models.modular_vlm import ModularVLM

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


ANSWER_REGEX = re.compile(
    r"(?:the\s+)?answer(?:\s+is|:)?\s*\*?\*?\(?([0-9]+)\)?",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 microscopy Unified VQA training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase3_qwen3_dinov3.yaml",
        help="Path to the Phase 3 config file.",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default=None,
        help="Optional Phase 2 checkpoint override for initialization.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume full Phase 3 training state from a prior checkpoint.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override training.num_epochs from config.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override training.max_steps from config.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_main(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_main_process:
        print(message, flush=True)


def parse_int_list(raw_value: Any) -> List[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    else:
        items = list(raw_value)

    parsed: List[int] = []
    for item in items:
        if item in {"", None}:
            continue
        value = int(item)
        if value > 0:
            parsed.append(value)
    return sorted(set(parsed))


def parse_optional_int(raw_value: Any) -> int | None:
    if raw_value in {None, "", "null", "none", "None"}:
        return None
    return int(raw_value)


def resolve_run_output_dir(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> tuple[Path, str]:
    training_cfg = config["training"]
    base_output_dir = Path(training_cfg["output_dir"])
    use_run_subdir = bool(training_cfg.get("use_run_subdir", True))
    configured_run_name = training_cfg.get("run_name")

    if args.resume:
        output_dir = Path(args.resume).resolve().parent
        return output_dir, output_dir.name

    if not use_run_subdir:
        run_name = configured_run_name or f"{base_output_dir.name}_phase3"
        return base_output_dir, run_name

    run_name = configured_run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.get('project_name', 'phase3')}_{timestamp}"
    return base_output_dir / run_name, run_name


def write_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    run_name: str,
    args: argparse.Namespace,
) -> None:
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    run_info = {
        "run_name": run_name,
        "stage": "phase3",
        "config_path": args.config,
        "phase2_checkpoint": args.phase2_checkpoint,
        "resume": args.resume,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)


def maybe_init_wandb(
    accelerator: Accelerator,
    config: Dict[str, Any],
    run_name: str,
):
    if not accelerator.is_main_process:
        return None

    wandb_cfg = dict(config.get("training", {}).get("wandb", {}))
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        print("wandb logging was requested but wandb is not installed. Continuing without wandb.")
        return None

    return wandb.init(
        project=wandb_cfg.get("project", "microvqa"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or run_name,
        tags=(wandb_cfg.get("tags") or []) + ["phase3", "unified_vqa"],
        mode=wandb_cfg.get("mode"),
        config=config,
    )


def disable_llm_cache_for_training(model: ModularVLM) -> None:
    llm_modules = [model.llm]
    base_model = getattr(model.llm, "base_model", None)
    if base_model is not None and base_model is not model.llm:
        llm_modules.append(base_model)

    for module in llm_modules:
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

        generation_config = getattr(module, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False


def build_collate_fn(
    tokenizer,
    image_transform,
    max_text_length: int,
):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([image_transform(item["image"]) for item in batch])
        prompts = [item["text"] for item in batch]
        targets = [item["target_text"] for item in batch]

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        eos_token_id = tokenizer.eos_token_id
        input_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        attention_rows: List[List[int]] = []

        for prompt, target in zip(prompts, targets):
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            if eos_token_id is not None:
                target_ids = target_ids + [eos_token_id]
            if not target_ids:
                raise ValueError("Phase 3 target_text must produce at least one target token.")

            max_prompt_len = max_text_length - 1
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

            available_target_len = max_text_length - len(prompt_ids)
            if available_target_len <= 0:
                raise RuntimeError("Unable to preserve target tokens after prompt truncation.")

            target_ids = target_ids[:available_target_len]
            if not target_ids:
                raise RuntimeError("Prompt truncation removed all target tokens.")

            combined_ids = prompt_ids + target_ids
            combined_labels = ([-100] * len(prompt_ids)) + target_ids
            combined_attention = [1] * len(combined_ids)
            pad_len = max_text_length - len(combined_ids)
            if pad_len < 0:
                raise RuntimeError("Combined prompt/target sequence exceeded max_text_length unexpectedly.")

            input_rows.append(combined_ids + ([pad_token_id] * pad_len))
            label_rows.append(combined_labels + ([-100] * pad_len))
            attention_rows.append(combined_attention + ([0] * pad_len))

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }

    return collate_fn


def build_dataset(
    manifest_path: str,
    image_root: str | None,
    prompt_style: str,
) -> ImageTextDataset:
    return ImageTextDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        prompt_style=prompt_style,
    )


def build_dataloader(
    dataset: Dataset,
    data_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    tokenizer,
    *,
    shuffle: bool,
    batch_size: int | None = None,
):
    image_transform = build_image_transform(
        image_size=int(data_cfg["image_size"]),
        preprocessing=str(data_cfg.get("image_preprocessing", "resize")),
    )
    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_text_length=int(data_cfg["max_text_length"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size or training_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
    )


def resolve_phase3_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    if data_cfg.get("train_manifest") is None:
        raise ValueError("Phase 3 requires data.train_manifest.")
    if data_cfg.get("val_manifest") is None:
        raise ValueError("Phase 3 requires data.val_manifest for validation accuracy.")
    data_cfg.setdefault("prompt_style", "answer_only")
    data_cfg.setdefault("image_preprocessing", "pad_preserve")
    return data_cfg


def apply_phase3_lora(model: ModularVLM, config: Dict[str, Any]) -> None:
    phase3_cfg = config["phase3"]
    lora_cfg = dict(phase3_cfg.get("lora", {}))
    if not lora_cfg.get("enabled", True):
        raise ValueError("Phase 3 expects phase3.lora.enabled=true for the QLoRA path.")
    if str(config["model"].get("llm_quantization", "")).lower() != "4bit":
        raise ValueError("Phase 3 QLoRA expects model.llm_quantization='4bit'.")

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "Phase 3 LoRA training requires 'peft'. Install it in the active environment "
            "or add it to your environment before launching train_phase3.py."
        ) from exc

    gradient_checkpointing = bool(config["training"].get("gradient_checkpointing", True))
    disable_llm_cache_for_training(model)
    model.llm = prepare_model_for_kbit_training(
        model.llm,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )
    model.llm = get_peft_model(
        model.llm,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 128)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 256)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(lora_cfg.get("bias", "none")),
        ),
    )
    disable_llm_cache_for_training(model)
    model.refresh_llm_references()


def get_backbone_block_range(model: ModularVLM, phase3_cfg: Dict[str, Any]) -> tuple[int, int]:
    blocks = getattr(model.vision_encoder.backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Phase 3 expects vision_encoder.backbone.blocks to exist.")
    start_block = int(phase3_cfg.get("backbone_train_start_block", 20))
    end_block = len(blocks)
    if start_block < 0 or start_block >= end_block:
        raise ValueError(
            f"Invalid backbone_train_start_block={start_block} for {end_block} backbone blocks."
        )
    return start_block, end_block


def apply_phase3_trainable_state(model: ModularVLM, phase3_cfg: Dict[str, Any]) -> tuple[int, int]:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.vision_encoder.alignment_head.parameters():
        param.requires_grad = True
    for param in model.adapter.parameters():
        param.requires_grad = True

    start_block, end_block = get_backbone_block_range(model, phase3_cfg)
    for block_index in range(start_block, end_block):
        for param in model.vision_encoder.backbone.blocks[block_index].parameters():
            param.requires_grad = True

    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    return start_block, end_block


def load_phase3_checkpoint(
    model: ModularVLM,
    checkpoint_path: str,
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
    resume_training_state: bool = False,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "adapter" in checkpoint:
        model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
    if "vision_alignment_head" in checkpoint:
        model.vision_encoder.alignment_head.load_state_dict(
            checkpoint["vision_alignment_head"],
            strict=True,
        )

    backbone_state = checkpoint.get("vision_backbone_top_blocks")
    if backbone_state:
        missing_keys, unexpected_keys = model.vision_encoder.backbone.load_state_dict(
            backbone_state,
            strict=False,
        )
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading Phase 3 backbone state: {unexpected_keys}"
            )
        if not backbone_state:
            raise RuntimeError("Phase 3 backbone state dict was unexpectedly empty.")
        _ = missing_keys

    lora_state = checkpoint.get("lora_state")
    if lora_state:
        try:
            from peft import set_peft_model_state_dict
        except ImportError as exc:
            raise ImportError("Loading a Phase 3 checkpoint with LoRA weights requires 'peft'.") from exc
        set_peft_model_state_dict(model.llm, lora_state)

    if resume_training_state:
        if optimizer is None or scheduler is None:
            raise ValueError("optimizer and scheduler are required when resume_training_state=True.")
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def split_param_groups_by_module(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    learning_rates: Dict[str, float],
    backbone_start_block: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {
        "vision_backbone_top_blocks_decay": {
            "params": [],
            "lr": float(learning_rates["vision_backbone_top_blocks"]),
            "weight_decay": None,
        },
        "vision_backbone_top_blocks_no_decay": {
            "params": [],
            "lr": float(learning_rates["vision_backbone_top_blocks"]),
            "weight_decay": 0.0,
        },
        "vision_alignment_head_decay": {
            "params": [],
            "lr": float(learning_rates["vision_alignment_head"]),
            "weight_decay": None,
        },
        "vision_alignment_head_no_decay": {
            "params": [],
            "lr": float(learning_rates["vision_alignment_head"]),
            "weight_decay": 0.0,
        },
        "adapter_decay": {
            "params": [],
            "lr": float(learning_rates["adapter"]),
            "weight_decay": None,
        },
        "adapter_no_decay": {
            "params": [],
            "lr": float(learning_rates["adapter"]),
            "weight_decay": 0.0,
        },
        "llm_lora_decay": {
            "params": [],
            "lr": float(learning_rates["llm_lora"]),
            "weight_decay": None,
        },
        "llm_lora_no_decay": {
            "params": [],
            "lr": float(learning_rates["llm_lora"]),
            "weight_decay": 0.0,
        },
    }
    no_decay_terms = ("bias", "norm", "ln", "layernorm")

    for name, param in named_params:
        if not param.requires_grad:
            continue

        if name.startswith("vision_encoder.backbone.blocks."):
            parts = name.split(".")
            block_index = int(parts[3])
            if block_index < backbone_start_block:
                raise RuntimeError(
                    f"Found unexpectedly trainable frozen backbone block parameter: {name}"
                )
            module_key = "vision_backbone_top_blocks"
        elif name.startswith("vision_encoder.alignment_head."):
            module_key = "vision_alignment_head"
        elif name.startswith("adapter."):
            module_key = "adapter"
        elif name.startswith("llm.") and "lora_" in name:
            module_key = "llm_lora"
        else:
            raise RuntimeError(f"Unexpected trainable parameter outside Phase 3 groups: {name}")

        group_suffix = "_no_decay" if (
            param.ndim == 1 or any(term in name.lower() for term in no_decay_terms)
        ) else "_decay"
        grouped[module_key + group_suffix]["params"].append(param)

    return [group for group in grouped.values() if group["params"]]


def build_optimizer(
    model: ModularVLM,
    phase3_cfg: Dict[str, Any],
    backbone_start_block: int,
) -> AdamW:
    learning_rates = dict(phase3_cfg.get("learning_rates", {}))
    required_keys = {
        "vision_backbone_top_blocks",
        "vision_alignment_head",
        "adapter",
        "llm_lora",
    }
    missing = required_keys.difference(learning_rates.keys())
    if missing:
        raise KeyError(f"phase3.learning_rates is missing keys: {sorted(missing)}")

    param_groups = split_param_groups_by_module(
        model.named_parameters(),
        learning_rates,
        backbone_start_block=backbone_start_block,
    )
    if not param_groups:
        raise RuntimeError("No trainable parameters were found for Phase 3.")

    weight_decay = float(phase3_cfg.get("weight_decay", 0.01))
    for group in param_groups:
        if group["weight_decay"] is None:
            group["weight_decay"] = weight_decay

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def extract_trainable_backbone_state(
    backbone: torch.nn.Module,
    start_block: int,
) -> Dict[str, torch.Tensor]:
    selected: Dict[str, torch.Tensor] = {}
    for key, value in backbone.state_dict().items():
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        block_index = int(parts[1])
        if block_index >= start_block:
            selected[key] = value.detach().cpu()
    if not selected:
        raise RuntimeError(
            "No trainable backbone block tensors were found when saving Phase 3 checkpoint."
        )
    return selected


def save_phase3_checkpoint(
    accelerator: Accelerator,
    model: ModularVLM,
    optimizer: AdamW,
    scheduler: Any,
    output_dir: Path,
    optimizer_step: int,
    global_step: int,
    checkpoint_name: str,
    backbone_start_block: int,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Path:
    unwrap_model = accelerator.unwrap_model(model)
    checkpoint = {
        "stage": "phase3",
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "adapter": unwrap_model.adapter.state_dict(),
        "vision_alignment_head": unwrap_model.vision_encoder.alignment_head.state_dict(),
        "vision_backbone_top_blocks": extract_trainable_backbone_state(
            unwrap_model.vision_encoder.backbone,
            start_block=backbone_start_block,
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError("Saving Phase 3 LoRA checkpoints requires 'peft'.") from exc

    checkpoint["lora_state"] = get_peft_model_state_dict(unwrap_model.llm)
    if checkpoint_metadata is not None:
        checkpoint["checkpoint_metadata"] = checkpoint_metadata

    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def append_train_log(log_path: Path, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_validation_loss(
    accelerator: Accelerator,
    model: ModularVLM,
    dataloader: DataLoader,
    *,
    max_batches: int | None = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Validation forward pass did not return a loss tensor.")

            reduced_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            total_loss += reduced_loss
            total_batches += 1

            if max_batches is not None and batch_idx >= max_batches:
                break

    if was_training:
        model.train()

    if total_batches == 0:
        raise RuntimeError("Validation dataloader produced zero batches.")

    return {
        "loss": total_loss / total_batches,
        "num_batches": float(total_batches),
    }


def parse_choice_answer(text: str) -> int | None:
    match = ANSWER_REGEX.search(text)
    if match is None:
        return None
    return int(match.group(1))


def sample_eval_indices(dataset_size: int, max_samples: int | None, seed: int) -> List[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= dataset_size:
        return list(range(dataset_size))
    indices = random.Random(seed).sample(range(dataset_size), max_samples)
    return sorted(indices)


def sanitize_metric_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


@torch.inference_mode()
def generate_answer_text(
    model: ModularVLM,
    tokenizer,
    image_transform,
    image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    device = model.get_llm_device()
    pixel_values = image_transform(image).unsqueeze(0).to(device)
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    model_inputs = model.build_multimodal_inputs(
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
        decoder_outputs = model.llm_body.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        hidden_states = decoder_outputs[0]
        logits = model.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        token_id = int(next_token_id.item())

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        next_embed = model.get_text_embeds(next_token_id.unsqueeze(0))
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
        next_mask = torch.ones((1, 1), dtype=merged_attention_mask.dtype, device=device)
        merged_attention_mask = torch.cat([merged_attention_mask, next_mask], dim=1)
        if visual_pos_masks is not None:
            next_visual_mask = torch.zeros(
                (visual_pos_masks.size(0), 1),
                dtype=torch.bool,
                device=device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, next_visual_mask], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_validation_accuracy(
    accelerator: Accelerator,
    model: ModularVLM,
    tokenizer,
    dataset: Sequence[Dict[str, Any]],
    data_cfg: Dict[str, Any],
    *,
    max_samples: int | None,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, Any] | None:
    accelerator.wait_for_everyone()

    metrics: Dict[str, Any] | None = None
    if accelerator.is_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        was_training = unwrap_model.training
        unwrap_model.eval()

        image_transform = build_image_transform(
            image_size=int(data_cfg["image_size"]),
            preprocessing=str(data_cfg.get("image_preprocessing", "resize")),
        )
        selected_indices = sample_eval_indices(
            dataset_size=len(dataset),
            max_samples=max_samples,
            seed=seed,
        )

        total_correct = 0
        per_dataset_total: Dict[str, int] = defaultdict(int)
        per_dataset_correct: Dict[str, int] = defaultdict(int)

        for index in selected_indices:
            sample = dataset[index]
            metadata = dict(sample.get("metadata") or {})
            correct_index = int(metadata["correct_index"])
            source_dataset = str(metadata.get("source_dataset") or "unknown")
            prediction_text = generate_answer_text(
                model=unwrap_model,
                tokenizer=tokenizer,
                image_transform=image_transform,
                image=sample["image"],
                prompt=sample["text"],
                max_new_tokens=max_new_tokens,
            )
            prediction_index = parse_choice_answer(prediction_text)
            is_correct = prediction_index == correct_index

            total_correct += int(is_correct)
            per_dataset_total[source_dataset] += 1
            per_dataset_correct[source_dataset] += int(is_correct)

        per_dataset_accuracy = {
            dataset_name: per_dataset_correct[dataset_name] / count
            for dataset_name, count in per_dataset_total.items()
            if count > 0
        }
        metrics = {
            "overall_accuracy": total_correct / len(selected_indices) if selected_indices else 0.0,
            "num_samples": len(selected_indices),
            "per_dataset_accuracy": per_dataset_accuracy,
        }

        if was_training:
            unwrap_model.train()

    accelerator.wait_for_everyone()
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    phase3_cfg = dict(config["phase3"])
    data_cfg = resolve_phase3_data_config(config)
    training_cfg = dict(config["training"])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    mixed_precision = str(training_cfg.get("mixed_precision", "bf16")).lower()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=None if mixed_precision == "none" else mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(int(config.get("seed", 42)))
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.local_process_index)

    output_dir, run_name = resolve_run_output_dir(args, config)
    config["training"]["resolved_output_dir"] = str(output_dir)
    config["training"]["resolved_run_name"] = run_name
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(output_dir, config, run_name, args)
    accelerator.wait_for_everyone()

    log_path = output_dir / "train_log.jsonl"
    phase2_init_checkpoint = args.phase2_checkpoint or phase3_cfg.get("resume_from_phase2_checkpoint")
    if not args.resume and not phase2_init_checkpoint:
        raise ValueError(
            "Phase 3 requires a Phase 2 checkpoint. "
            "Set phase3.resume_from_phase2_checkpoint or pass --phase2-checkpoint."
        )

    print_main(accelerator, f"[startup] config={args.config}")
    print_main(accelerator, "[startup] stage=phase3")
    print_main(accelerator, f"[startup] run_name={run_name}")
    print_main(accelerator, f"[startup] output_dir={output_dir}")
    print_main(
        accelerator,
        f"[startup] device={accelerator.device} process_index={accelerator.process_index} "
        f"num_processes={accelerator.num_processes}",
    )
    print_main(
        accelerator,
        "[startup] data_settings="
        f"train_manifest={data_cfg.get('train_manifest')} "
        f"val_manifest={data_cfg.get('val_manifest')} "
        f"test_manifest={data_cfg.get('test_manifest')} "
        f"image_size={data_cfg.get('image_size')} "
        f"image_preprocessing={data_cfg.get('image_preprocessing')} "
        f"max_text_length={data_cfg.get('max_text_length')} "
        f"prompt_style={data_cfg.get('prompt_style')} "
        f"num_workers={data_cfg.get('num_workers', 4)}",
    )
    print_main(
        accelerator,
        "[startup] model_settings="
        f"llm_quantization={config['model'].get('llm_quantization')} "
        f"deepstack={config['model'].get('use_deepstack_injection', True)}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["llm_base"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ModularVLM(config)
    model = model.prepare_for_training_device(accelerator.device)
    apply_phase3_lora(model, config)
    backbone_start_block, backbone_end_block = apply_phase3_trainable_state(model, phase3_cfg)

    train_dataset = build_dataset(
        manifest_path=str(data_cfg["train_manifest"]),
        image_root=data_cfg.get("image_root"),
        prompt_style=str(data_cfg.get("prompt_style", "answer_only")),
    )
    val_dataset = build_dataset(
        manifest_path=str(data_cfg["val_manifest"]),
        image_root=data_cfg.get("val_image_root") or data_cfg.get("image_root"),
        prompt_style=str(data_cfg.get("prompt_style", "answer_only")),
    )

    train_dataloader = build_dataloader(
        train_dataset,
        data_cfg,
        training_cfg,
        tokenizer,
        shuffle=True,
    )
    val_loss_dataloader = build_dataloader(
        val_dataset,
        data_cfg,
        training_cfg,
        tokenizer,
        shuffle=False,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
    )
    print_main(
        accelerator,
        f"[startup] train dataloader ready with {len(train_dataset)} samples and "
        f"{len(train_dataloader)} local batches/epoch",
    )
    print_main(
        accelerator,
        f"[startup] val dataloader ready with {len(val_dataset)} samples and "
        f"{len(val_loss_dataloader)} local batches/eval",
    )

    optimizer = build_optimizer(
        model=model,
        phase3_cfg=phase3_cfg,
        backbone_start_block=backbone_start_block,
    )
    num_epochs = args.num_epochs if args.num_epochs is not None else int(training_cfg.get("num_epochs", 1))
    max_steps = args.max_steps if args.max_steps is not None else training_cfg.get("max_steps")
    max_steps = None if max_steps in {None, 0} else int(max_steps)

    total_update_steps = max_steps or math.ceil(
        len(train_dataloader) * num_epochs / accelerator.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        num_training_steps=max(total_update_steps, 1),
    )

    if phase2_init_checkpoint and not args.resume:
        init_state = load_phase3_checkpoint(model, phase2_init_checkpoint)
        print_main(
            accelerator,
            f"[startup] initialized from prior checkpoint: {phase2_init_checkpoint} "
            f"(stage={init_state.get('stage', 'unknown')})",
        )

    model, optimizer, train_dataloader, val_loss_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_loss_dataloader,
        scheduler,
    )

    global_step = 0
    optimizer_step = 0
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    if args.resume:
        resumed = load_phase3_checkpoint(
            accelerator.unwrap_model(model),
            args.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training_state=True,
        )
        global_step = int(resumed.get("global_step", 0))
        optimizer_step = int(resumed.get("optimizer_step", resumed.get("global_step", 0)))
        checkpoint_metadata = resumed.get("checkpoint_metadata") or {}
        best_val_accuracy = float(checkpoint_metadata.get("best_val_accuracy", best_val_accuracy))
        best_val_loss = float(checkpoint_metadata.get("best_val_loss", best_val_loss))
        print_main(accelerator, f"[startup] resumed from {args.resume} at optimizer_step={optimizer_step}")

    print_main(accelerator, "Trainable parameters:")
    for name, param in accelerator.unwrap_model(model).named_parameters():
        if param.requires_grad:
            print_main(accelerator, f"  - {name}")
    print_main(
        accelerator,
        f"[startup] trainable backbone blocks={list(range(backbone_start_block, backbone_end_block))}",
    )

    wandb_run = maybe_init_wandb(accelerator, config, run_name)
    if wandb_run is not None:
        wandb_run.summary["train_dataset_size"] = len(train_dataset)
        wandb_run.summary["val_dataset_size"] = len(val_dataset)
        wandb_run.summary["batches_per_epoch"] = len(train_dataloader)
        wandb_run.summary["total_update_steps"] = total_update_steps

    save_every = int(training_cfg.get("save_every", 0))
    save_steps = set(parse_int_list(training_cfg.get("save_steps")))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    eval_every = int(training_cfg.get("eval_every", 0))
    eval_accuracy_every = int(training_cfg.get("eval_accuracy_every", eval_every or 0))
    eval_loss_max_batches = parse_optional_int(training_cfg.get("eval_max_batches"))
    eval_accuracy_max_samples = parse_optional_int(training_cfg.get("eval_accuracy_max_samples"))
    eval_max_new_tokens = int(training_cfg.get("eval_max_new_tokens", 16))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    print_main(accelerator, "[startup] starting Phase 3 training loop...")
    print_main(
        accelerator,
        f"[startup] validation loss every {eval_every} optimizer steps "
        f"(max_batches={eval_loss_max_batches or 'all'})",
    )
    print_main(
        accelerator,
        f"[startup] validation accuracy every {eval_accuracy_every} optimizer steps "
        f"(max_samples={eval_accuracy_max_samples or 'all'}, max_new_tokens={eval_max_new_tokens})",
    )

    last_train_loss: float | None = None
    last_val_loss: float | None = None
    last_val_accuracy: float | None = None
    stop_training = False

    for epoch in range(num_epochs):
        print_main(accelerator, f"[epoch] starting epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model forward pass did not return a loss tensor.")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if not accelerator.sync_gradients:
                continue

            optimizer_step += 1
            reduced_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            last_train_loss = reduced_loss
            lr = scheduler.get_last_lr()[0]

            if wandb_run is not None:
                wandb_payload = {
                    "train/loss": reduced_loss,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/optimizer_step": optimizer_step,
                }
                if last_val_loss is not None:
                    wandb_payload["train/val_loss_gap"] = reduced_loss - last_val_loss
                wandb.log(wandb_payload, step=optimizer_step)

            if optimizer_step % log_every == 0:
                log_record = {
                    "stage": "phase3",
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": reduced_loss,
                    "lr": lr,
                }
                if accelerator.is_main_process:
                    append_train_log(log_path, log_record)
                print_main(
                    accelerator,
                    f"stage=phase3 epoch={epoch} step={optimizer_step} "
                    f"loss={reduced_loss:.4f} lr={lr:.6e}",
                )

            if eval_every > 0 and optimizer_step % eval_every == 0:
                val_loss_metrics = run_validation_loss(
                    accelerator=accelerator,
                    model=model,
                    dataloader=val_loss_dataloader,
                    max_batches=eval_loss_max_batches,
                )
                last_val_loss = float(val_loss_metrics["loss"])
                best_val_loss = min(best_val_loss, last_val_loss)

                if accelerator.is_main_process:
                    append_train_log(
                        log_path,
                        {
                            "stage": "phase3",
                            "event": "validation_loss",
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "train_loss": reduced_loss,
                            "val_loss": last_val_loss,
                            "val_batches": int(val_loss_metrics["num_batches"]),
                        },
                    )
                print_main(
                    accelerator,
                    f"[validation_loss] step={optimizer_step} val_loss={last_val_loss:.4f}",
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "val/loss": last_val_loss,
                            "val/num_batches": val_loss_metrics["num_batches"],
                        },
                        step=optimizer_step,
                    )

            accuracy_metrics = None
            previous_best_val_accuracy = best_val_accuracy
            if eval_accuracy_every > 0 and optimizer_step % eval_accuracy_every == 0:
                accuracy_metrics = run_validation_accuracy(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    data_cfg=data_cfg,
                    max_samples=eval_accuracy_max_samples,
                    max_new_tokens=eval_max_new_tokens,
                    seed=int(config.get("seed", 42)),
                )
                if accelerator.is_main_process and accuracy_metrics is not None:
                    last_val_accuracy = float(accuracy_metrics["overall_accuracy"])
                    best_val_accuracy = max(best_val_accuracy, last_val_accuracy)
                    log_payload = {
                        "stage": "phase3",
                        "event": "validation_accuracy",
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "val_accuracy": last_val_accuracy,
                        "num_samples": int(accuracy_metrics["num_samples"]),
                        "per_dataset_accuracy": accuracy_metrics["per_dataset_accuracy"],
                    }
                    append_train_log(log_path, log_payload)
                    print_main(
                        accelerator,
                        f"[validation_accuracy] step={optimizer_step} "
                        f"val_accuracy={last_val_accuracy:.4f} "
                        f"samples={int(accuracy_metrics['num_samples'])}",
                    )
                    if wandb_run is not None:
                        metrics_payload = {
                            "val/accuracy": last_val_accuracy,
                            "val/accuracy_num_samples": accuracy_metrics["num_samples"],
                        }
                        for dataset_name, accuracy in accuracy_metrics["per_dataset_accuracy"].items():
                            metrics_payload[
                                f"val_accuracy/by_dataset/{sanitize_metric_key(dataset_name)}"
                            ] = accuracy
                        wandb.log(metrics_payload, step=optimizer_step)

            if accelerator.is_main_process:
                checkpoint_metadata = {
                    "stage": "phase3",
                    "loss": reduced_loss,
                    "lr": lr,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                }
                if last_val_loss is not None:
                    checkpoint_metadata["val_loss"] = last_val_loss
                if last_val_accuracy is not None:
                    checkpoint_metadata["val_accuracy"] = last_val_accuracy

                if save_every > 0 and optimizer_step % save_every == 0:
                    checkpoint_path = save_phase3_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name=f"phase3_step_{optimizer_step}.pt",
                        backbone_start_block=backbone_start_block,
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Saved periodic checkpoint to {checkpoint_path}")

                if optimizer_step in save_steps:
                    checkpoint_path = save_phase3_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name=f"phase3_milestone_step_{optimizer_step}.pt",
                        backbone_start_block=backbone_start_block,
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Saved milestone checkpoint to {checkpoint_path}")

                if (
                    accuracy_metrics is not None
                    and last_val_accuracy is not None
                    and last_val_accuracy > previous_best_val_accuracy
                ):
                    best_path = save_phase3_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name="phase3_best_accuracy.pt",
                        backbone_start_block=backbone_start_block,
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Updated best-accuracy checkpoint at {best_path}")

            if max_steps is not None and optimizer_step >= max_steps:
                stop_training = True
                break

        if stop_training:
            break

    accelerator.wait_for_everyone()
    final_checkpoint = None
    if accelerator.is_main_process:
        final_checkpoint = save_phase3_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            optimizer_step=optimizer_step,
            global_step=global_step,
            checkpoint_name="phase3_final.pt",
            backbone_start_block=backbone_start_block,
            checkpoint_metadata={
                "stage": "phase3",
                "final_loss": last_train_loss,
                "final_val_loss": last_val_loss,
                "final_val_accuracy": last_val_accuracy,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
            },
        )
        print_main(accelerator, f"Saved final checkpoint to {final_checkpoint}")
        print_main(accelerator, "Phase 3 training complete.")

    if wandb_run is not None:
        if final_checkpoint is not None:
            wandb_run.summary["final_checkpoint"] = str(final_checkpoint)
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_val_accuracy"] = best_val_accuracy
        wandb_run.finish()


if __name__ == "__main__":
    main()
