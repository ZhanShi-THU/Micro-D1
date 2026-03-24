"""
Phase 2 deep semantic fusion training.

This stage trains:
  - dinotxt alignment head
  - visual adapter
  - Qwen LoRA parameters

It expects Phase 1 adapter weights as initialization for the instruct stage,
and a prior Phase 2 checkpoint as initialization for the VQA stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


SUPPORTED_STAGES = {"instruct", "vqa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 deep semantic fusion training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_qwen3_dinov3.yaml",
        help="Path to the Phase 2 config file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_STAGES),
        help="Override phase2.stage from the config.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=str,
        default=None,
        help="Optional Phase 1 adapter checkpoint override.",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default=None,
        help="Optional prior Phase 2 checkpoint override for model initialization.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume full Phase 2 training state from a prior checkpoint.",
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


def build_collate_fn(tokenizer, image_transform, max_text_length: int):
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
                raise ValueError("Phase 2 target_text must produce at least one target token.")

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


def parse_float_list(raw_value: Any) -> List[float]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    else:
        items = list(raw_value)

    parsed: List[float] = []
    for item in items:
        if item in {"", None}:
            continue
        parsed.append(float(item))
    return sorted(set(parsed), reverse=True)


def parse_optional_float(raw_value: Any) -> float | None:
    if raw_value in {None, "", "null", "none", "None"}:
        return None
    return float(raw_value)


def format_metric_for_filename(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")


def resolve_stage(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    stage = (args.stage or config.get("phase2", {}).get("stage") or "instruct").strip().lower()
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported phase2.stage: {stage}")
    return stage


def resolve_stage_data_config(config: Dict[str, Any], stage: str) -> Dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    stage_cfg = dict(config.get("phase2", {}).get("stages", {}).get(stage, {}))
    merged = {**data_cfg, **stage_cfg}
    if merged.get("train_manifest") is None:
        raise ValueError(
            f"No train_manifest configured for Phase 2 stage '{stage}'. "
            "Set data.train_manifest or phase2.stages.<stage>.train_manifest."
        )
    merged.setdefault("image_preprocessing", "resize")
    return merged


def has_validation_data(data_cfg: Dict[str, Any]) -> bool:
    return data_cfg.get("val_manifest") is not None


class MixedImageTextDataset(Dataset):
    """
    Mix a primary dataset with an auxiliary dataset while keeping the primary
    dataset fully represented in each logical epoch.

    auxiliary_fraction is interpreted as the desired fraction of auxiliary
    samples in the combined dataset.
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        auxiliary_dataset: Dataset,
        auxiliary_fraction: float,
    ) -> None:
        if not 0.0 < auxiliary_fraction < 0.5:
            raise ValueError(
                "auxiliary_fraction must be in (0.0, 0.5) for mixed Phase 2 VQA training, "
                f"got {auxiliary_fraction}."
            )
        if len(primary_dataset) == 0:
            raise ValueError("primary_dataset must not be empty.")
        if len(auxiliary_dataset) == 0:
            raise ValueError("auxiliary_dataset must not be empty.")

        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.auxiliary_fraction = auxiliary_fraction
        self.primary_size = len(primary_dataset)
        self.auxiliary_size = len(auxiliary_dataset)
        self.auxiliary_count = max(
            1,
            round(self.primary_size * auxiliary_fraction / (1.0 - auxiliary_fraction)),
        )
        self.total_size = self.primary_size + self.auxiliary_count

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < self.primary_size:
            sample = dict(self.primary_dataset[index])
            metadata = dict(sample.get("metadata") or {})
            metadata["phase2_mix_role"] = "primary"
            sample["metadata"] = metadata
            return sample

        auxiliary_index = (index - self.primary_size) % self.auxiliary_size
        sample = dict(self.auxiliary_dataset[auxiliary_index])
        metadata = dict(sample.get("metadata") or {})
        metadata["phase2_mix_role"] = "auxiliary"
        sample["metadata"] = metadata
        return sample


def build_training_dataset(data_cfg: Dict[str, Any], stage: str) -> Dataset:
    primary_dataset = ImageTextDataset(
        manifest_path=str(data_cfg["train_manifest"]),
        image_root=data_cfg.get("image_root"),
    )

    auxiliary_manifest = data_cfg.get("auxiliary_train_manifest")
    auxiliary_fraction = data_cfg.get("auxiliary_fraction")
    if stage != "vqa" or auxiliary_manifest is None or auxiliary_fraction in {None, 0, 0.0}:
        return primary_dataset

    auxiliary_dataset = ImageTextDataset(
        manifest_path=str(auxiliary_manifest),
        image_root=data_cfg.get("auxiliary_image_root") or data_cfg.get("image_root"),
    )
    return MixedImageTextDataset(
        primary_dataset=primary_dataset,
        auxiliary_dataset=auxiliary_dataset,
        auxiliary_fraction=float(auxiliary_fraction),
    )


def build_validation_dataset(data_cfg: Dict[str, Any]) -> Dataset:
    val_manifest = data_cfg.get("val_manifest")
    if val_manifest is None:
        raise ValueError("Validation dataset requested, but val_manifest is not configured.")
    return ImageTextDataset(
        manifest_path=str(val_manifest),
        image_root=data_cfg.get("val_image_root") or data_cfg.get("image_root"),
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
        int(data_cfg["image_size"]),
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


def resolve_run_output_dir(
    args: argparse.Namespace,
    config: Dict[str, Any],
    stage: str,
) -> tuple[Path, str]:
    training_cfg = config["training"]
    base_output_dir = Path(training_cfg["output_dir"])
    use_run_subdir = bool(training_cfg.get("use_run_subdir", True))
    configured_run_name = training_cfg.get("run_name")

    if args.resume:
        output_dir = Path(args.resume).resolve().parent
        return output_dir, output_dir.name

    if not use_run_subdir:
        run_name = configured_run_name or f"{base_output_dir.name}_{stage}"
        return base_output_dir, run_name

    run_name = configured_run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.get('project_name', 'phase2')}_{stage}_{timestamp}"
    return base_output_dir / run_name, run_name


def write_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    run_name: str,
    stage: str,
    args: argparse.Namespace,
) -> None:
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    run_info = {
        "run_name": run_name,
        "stage": stage,
        "config_path": args.config,
        "adapter_checkpoint": args.adapter_checkpoint,
        "phase2_checkpoint": args.phase2_checkpoint,
        "resume": args.resume,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)


def maybe_init_wandb(
    accelerator: Accelerator,
    config: Dict[str, Any],
    stage: str,
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
        tags=(wandb_cfg.get("tags") or []) + [f"stage:{stage}", "phase2"],
        mode=wandb_cfg.get("mode"),
        config=config,
    )


def refresh_model_llm_references(model: ModularVLM) -> None:
    model.refresh_llm_references()


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


def apply_phase2_lora(model: ModularVLM, config: Dict[str, Any]) -> None:
    phase2_cfg = config["phase2"]
    lora_cfg = dict(phase2_cfg.get("lora", {}))
    if not lora_cfg.get("enabled", True):
        raise ValueError("Phase 2 expects phase2.lora.enabled=true for the QLoRA path.")
    if str(config["model"].get("llm_quantization", "")).lower() != "4bit":
        raise ValueError("Phase 2 QLoRA expects model.llm_quantization='4bit'.")

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "Phase 2 LoRA training requires 'peft'. Install it in the active environment "
            "or add it to your environment before launching train_phase2.py."
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
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 128)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(lora_cfg.get("bias", "none")),
        ),
    )
    disable_llm_cache_for_training(model)
    refresh_model_llm_references(model)


def apply_phase2_trainable_state(model: ModularVLM) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.vision_encoder.alignment_head.parameters():
        param.requires_grad = True
    for param in model.adapter.parameters():
        param.requires_grad = True
    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def load_phase1_adapter_checkpoint(model: ModularVLM, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "adapter" in checkpoint:
        model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
        return
    if "model" in checkpoint:
        model.adapter.load_state_dict(checkpoint["model"], strict=True)
        return
    raise KeyError("Adapter checkpoint is missing 'adapter' or 'model' state.")


def load_phase2_checkpoint(
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

    lora_state = checkpoint.get("lora_state")
    if lora_state:
        try:
            from peft import set_peft_model_state_dict
        except ImportError as exc:
            raise ImportError("Loading a Phase 2 checkpoint with LoRA weights requires 'peft'.") from exc
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
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {
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

        if name.startswith("vision_encoder.alignment_head."):
            module_key = "vision_alignment_head"
        elif name.startswith("adapter."):
            module_key = "adapter"
        elif name.startswith("llm.") and "lora_" in name:
            module_key = "llm_lora"
        else:
            raise RuntimeError(f"Unexpected trainable parameter outside Phase 2 groups: {name}")

        group_suffix = "_no_decay" if (
            param.ndim == 1 or any(term in name.lower() for term in no_decay_terms)
        ) else "_decay"
        grouped[module_key + group_suffix]["params"].append(param)

    return [group for group in grouped.values() if group["params"]]


def build_optimizer(model: ModularVLM, phase2_cfg: Dict[str, Any]) -> AdamW:
    learning_rates = dict(phase2_cfg.get("learning_rates", {}))
    required_keys = {"vision_alignment_head", "adapter", "llm_lora"}
    missing = required_keys.difference(learning_rates.keys())
    if missing:
        raise KeyError(f"phase2.learning_rates is missing keys: {sorted(missing)}")

    param_groups = split_param_groups_by_module(model.named_parameters(), learning_rates)
    if not param_groups:
        raise RuntimeError("No trainable parameters were found for Phase 2.")

    weight_decay = float(phase2_cfg.get("weight_decay", 0.01))
    for group in param_groups:
        if group["weight_decay"] is None:
            group["weight_decay"] = weight_decay

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def save_phase2_checkpoint(
    accelerator: Accelerator,
    model: ModularVLM,
    optimizer: AdamW,
    scheduler: Any,
    output_dir: Path,
    stage: str,
    optimizer_step: int,
    global_step: int,
    checkpoint_name: str,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Path:
    unwrap_model = accelerator.unwrap_model(model)
    checkpoint = {
        "stage": stage,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "adapter": unwrap_model.adapter.state_dict(),
        "vision_alignment_head": unwrap_model.vision_encoder.alignment_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError("Saving Phase 2 LoRA checkpoints requires 'peft'.") from exc

    checkpoint["lora_state"] = get_peft_model_state_dict(unwrap_model.llm)
    if checkpoint_metadata is not None:
        checkpoint["checkpoint_metadata"] = checkpoint_metadata

    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def append_train_log(log_path: Path, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_validation(
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage = resolve_stage(args, config)
    training_cfg = dict(config["training"])
    phase2_cfg = dict(config["phase2"])
    data_cfg = resolve_stage_data_config(config, stage)
    phase2_cfg["stage"] = stage
    data_cfg["stage"] = stage

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

    output_dir, run_name = resolve_run_output_dir(args, config, stage)
    config["training"]["resolved_output_dir"] = str(output_dir)
    config["training"]["resolved_run_name"] = run_name
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(output_dir, config, run_name, stage, args)
    accelerator.wait_for_everyone()

    log_path = output_dir / "train_log.jsonl"
    adapter_checkpoint = args.adapter_checkpoint or config["model"].get("adapter_init_checkpoint")
    phase2_init_checkpoint = args.phase2_checkpoint or phase2_cfg.get("resume_from_phase2_checkpoint")

    if stage == "vqa" and not args.resume and not phase2_init_checkpoint:
        raise ValueError(
            "Phase 2 stage 'vqa' requires a prior Phase 2 checkpoint. "
            "Set phase2.resume_from_phase2_checkpoint or pass --phase2-checkpoint."
        )

    print_main(accelerator, f"[startup] config={args.config}")
    print_main(accelerator, f"[startup] stage={stage}")
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
        f"manifest={data_cfg.get('train_manifest')} "
        f"val_manifest={data_cfg.get('val_manifest')} "
        f"aux_manifest={data_cfg.get('auxiliary_train_manifest')} "
        f"aux_fraction={data_cfg.get('auxiliary_fraction')} "
        f"image_size={data_cfg.get('image_size')} "
        f"image_preprocessing={data_cfg.get('image_preprocessing')} "
        f"max_text_length={data_cfg.get('max_text_length')} "
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
    if adapter_checkpoint:
        load_phase1_adapter_checkpoint(model, adapter_checkpoint)
    apply_phase2_lora(model, config)
    apply_phase2_trainable_state(model)

    train_dataset = build_training_dataset(
        data_cfg=data_cfg,
        stage=str(data_cfg.get("stage", "instruct")),
    )
    train_dataloader = build_dataloader(
        train_dataset,
        data_cfg,
        training_cfg,
        tokenizer,
        shuffle=True,
    )
    dataset_size = len(train_dataset)
    print_main(
        accelerator,
        f"[startup] train dataloader ready with {dataset_size} samples and "
        f"{len(train_dataloader)} local batches/epoch",
    )

    val_dataloader = None
    val_dataset_size = 0
    if has_validation_data(data_cfg):
        val_dataset = build_validation_dataset(data_cfg)
        val_dataloader = build_dataloader(
            val_dataset,
            data_cfg,
            training_cfg,
            tokenizer,
            shuffle=False,
            batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        )
        val_dataset_size = len(val_dataset)
        print_main(
            accelerator,
            f"[startup] val dataloader ready with {val_dataset_size} samples and "
            f"{len(val_dataloader)} local batches/eval",
        )

    optimizer = build_optimizer(model, phase2_cfg)
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
        init_state = load_phase2_checkpoint(model, phase2_init_checkpoint)
        print_main(
            accelerator,
            f"[startup] initialized from prior Phase 2 checkpoint: {phase2_init_checkpoint} "
            f"(stage={init_state.get('stage', 'unknown')})",
        )

    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        )
    else:
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            scheduler,
        )

    global_step = 0
    optimizer_step = 0
    best_loss = float("inf")
    if args.resume:
        resumed = load_phase2_checkpoint(
            accelerator.unwrap_model(model),
            args.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training_state=True,
        )
        global_step = int(resumed.get("global_step", 0))
        optimizer_step = int(resumed.get("optimizer_step", resumed.get("global_step", 0)))
        checkpoint_metadata = resumed.get("checkpoint_metadata") or {}
        best_loss = float(checkpoint_metadata.get("best_loss", best_loss))
        print_main(accelerator, f"[startup] resumed from {args.resume} at optimizer_step={optimizer_step}")

    print_main(accelerator, "Trainable parameters:")
    for name, param in accelerator.unwrap_model(model).named_parameters():
        if param.requires_grad:
            print_main(accelerator, f"  - {name}")

    wandb_run = maybe_init_wandb(accelerator, config, stage, run_name)
    if wandb_run is not None:
        wandb_run.summary["dataset_size"] = dataset_size
        wandb_run.summary["batches_per_epoch"] = len(train_dataloader)
        wandb_run.summary["total_update_steps"] = total_update_steps
        if val_dataloader is not None:
            wandb_run.summary["val_dataset_size"] = val_dataset_size
            wandb_run.summary["val_batches_per_eval"] = len(val_dataloader)

    save_every = int(training_cfg.get("save_every", 0))
    save_steps = set(parse_int_list(training_cfg.get("save_steps")))
    save_loss_thresholds = parse_float_list(training_cfg.get("save_loss_thresholds"))
    best_checkpoint_min_loss = parse_optional_float(training_cfg.get("best_checkpoint_min_loss"))
    eval_every = int(training_cfg.get("eval_every", 0))
    eval_max_batches = int(training_cfg.get("eval_max_batches", 0))
    eval_max_batches = None if eval_max_batches <= 0 else eval_max_batches
    triggered_loss_thresholds: set[float] = set()
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = int(training_cfg.get("log_every", 10))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    print_main(accelerator, "[startup] starting Phase 2 training loop...")
    if best_checkpoint_min_loss is not None:
        print_main(
            accelerator,
            f"[startup] best checkpoint saving is gated until loss <= {best_checkpoint_min_loss:.4f}",
        )
    if val_dataloader is not None and eval_every > 0:
        print_main(
            accelerator,
            f"[startup] validation is enabled every {eval_every} optimizer steps "
            f"(max_batches={eval_max_batches or 'all'})",
        )

    last_reduced_loss: float | None = None
    last_val_loss: float | None = None
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
            previous_best_loss = best_loss
            best_loss = min(best_loss, reduced_loss)
            last_reduced_loss = reduced_loss
            lr = scheduler.get_last_lr()[0]

            if wandb_run is not None:
                wandb_payload = {
                    "train/loss": reduced_loss,
                    "train/best_loss": best_loss,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/optimizer_step": optimizer_step,
                }
                if last_val_loss is not None:
                    wandb_payload["train/val_gap"] = reduced_loss - last_val_loss
                wandb.log(wandb_payload, step=optimizer_step)

            if optimizer_step % log_every == 0:
                log_record = {
                    "stage": stage,
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": reduced_loss,
                    "best_loss": best_loss,
                    "lr": lr,
                }
                if accelerator.is_main_process:
                    append_train_log(log_path, log_record)
                print_main(
                    accelerator,
                    f"stage={stage} epoch={epoch} step={optimizer_step} "
                    f"loss={reduced_loss:.4f} best={best_loss:.4f} lr={lr:.6e}",
                )

            if val_dataloader is not None and eval_every > 0 and optimizer_step % eval_every == 0:
                val_metrics = run_validation(
                    accelerator=accelerator,
                    model=model,
                    dataloader=val_dataloader,
                    max_batches=eval_max_batches,
                )
                last_val_loss = float(val_metrics["loss"])
                val_gap = reduced_loss - last_val_loss

                if accelerator.is_main_process:
                    append_train_log(
                        log_path,
                        {
                            "stage": stage,
                            "event": "validation",
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "train_loss": reduced_loss,
                            "val_loss": last_val_loss,
                            "train_val_gap": val_gap,
                            "val_batches": int(val_metrics["num_batches"]),
                        },
                    )
                print_main(
                    accelerator,
                    f"[validation] stage={stage} step={optimizer_step} "
                    f"val_loss={last_val_loss:.4f} gap={val_gap:.4f}",
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "val/loss": last_val_loss,
                            "val/train_gap": val_gap,
                            "val/num_batches": val_metrics["num_batches"],
                        },
                        step=optimizer_step,
                    )

            if accelerator.is_main_process:
                checkpoint_metadata = {
                    "stage": stage,
                    "loss": reduced_loss,
                    "best_loss": best_loss,
                    "lr": lr,
                }
                if last_val_loss is not None:
                    checkpoint_metadata["val_loss"] = last_val_loss

                if save_every > 0 and optimizer_step % save_every == 0:
                    checkpoint_path = save_phase2_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        stage=stage,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name=f"phase2_{stage}_step_{optimizer_step}.pt",
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Saved periodic checkpoint to {checkpoint_path}")

                if optimizer_step in save_steps:
                    checkpoint_path = save_phase2_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        stage=stage,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name=f"phase2_{stage}_milestone_step_{optimizer_step}.pt",
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Saved milestone checkpoint to {checkpoint_path}")

                if (
                    reduced_loss < previous_best_loss
                    and (
                        best_checkpoint_min_loss is None
                        or reduced_loss <= best_checkpoint_min_loss
                    )
                ):
                    best_path = save_phase2_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        stage=stage,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        checkpoint_name=f"phase2_{stage}_best.pt",
                        checkpoint_metadata=checkpoint_metadata,
                    )
                    print_main(accelerator, f"Updated best checkpoint at {best_path}")

                for threshold in save_loss_thresholds:
                    if threshold in triggered_loss_thresholds:
                        continue
                    if reduced_loss <= threshold:
                        checkpoint_path = save_phase2_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            output_dir=output_dir,
                            stage=stage,
                            optimizer_step=optimizer_step,
                            global_step=global_step,
                            checkpoint_name=(
                                f"phase2_{stage}_loss_le_{format_metric_for_filename(threshold)}"
                                f"_step_{optimizer_step}.pt"
                            ),
                            checkpoint_metadata={**checkpoint_metadata, "loss_threshold": threshold},
                        )
                        triggered_loss_thresholds.add(threshold)
                        print_main(
                            accelerator,
                            f"Saved loss-threshold checkpoint ({threshold:.4f}) to {checkpoint_path}",
                        )

            if max_steps is not None and optimizer_step >= max_steps:
                stop_training = True
                break

        if stop_training:
            break

    accelerator.wait_for_everyone()
    final_checkpoint = None
    if accelerator.is_main_process:
        final_checkpoint = save_phase2_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            stage=stage,
            optimizer_step=optimizer_step,
            global_step=global_step,
            checkpoint_name=f"phase2_{stage}_final.pt",
            checkpoint_metadata={
                "stage": stage,
                "best_loss": best_loss,
                "final_loss": last_reduced_loss,
            },
        )
        print_main(accelerator, f"Saved final checkpoint to {final_checkpoint}")
        print_main(accelerator, "Phase 2 training complete.")

    if wandb_run is not None:
        if final_checkpoint is not None:
            wandb_run.summary["final_checkpoint"] = str(final_checkpoint)
        wandb_run.summary["best_loss"] = best_loss
        wandb_run.finish()


if __name__ == "__main__":
    main()
