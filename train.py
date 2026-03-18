from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.dataset import ImageTextDataset
from data.unified_vqa import get_unified_manifest_paths
from models.modular_vlm import ModularVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_dinov3.yaml",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["phase1", "phase2"],
        help="Override training.active_phase in the config.",
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


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_collate_fn(tokenizer, image_transform, max_text_length: int):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([image_transform(item["image"]) for item in batch])
        prompts = [item["text"] for item in batch]
        targets = [item["target_text"] for item in batch]

        prompt_tokenized = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=max_text_length,
            return_attention_mask=False,
        )
        target_tokenized = tokenizer(
            targets,
            padding=False,
            truncation=True,
            max_length=max_text_length,
            return_attention_mask=False,
        )

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id before training.")

        input_id_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        attention_rows: List[List[int]] = []

        for prompt_ids, target_ids in zip(
            prompt_tokenized["input_ids"],
            target_tokenized["input_ids"],
        ):
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]

            available_target_len = max(max_text_length - len(prompt_ids), 0)
            target_ids = target_ids[:available_target_len]

            combined_ids = prompt_ids + target_ids
            combined_labels = ([-100] * len(prompt_ids)) + target_ids
            combined_attention = [1] * len(combined_ids)

            pad_len = max_text_length - len(combined_ids)
            if pad_len < 0:
                combined_ids = combined_ids[:max_text_length]
                combined_labels = combined_labels[:max_text_length]
                combined_attention = combined_attention[:max_text_length]
                pad_len = 0

            input_id_rows.append(combined_ids + ([pad_token_id] * pad_len))
            label_rows.append(combined_labels + ([-100] * pad_len))
            attention_rows.append(combined_attention + ([0] * pad_len))

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_id_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }

    return collate_fn


def build_dataloader(config: Dict[str, Any], tokenizer):
    data_cfg = config["data"]
    manifest_defaults = get_unified_manifest_paths(data_cfg.get("unified_root"))
    train_manifest = data_cfg.get("train_manifest") or str(manifest_defaults["train"])
    if not train_manifest:
        return None

    dataset = ImageTextDataset(
        manifest_path=train_manifest,
        image_root=data_cfg.get("image_root"),
    )
    image_transform = build_image_transform(data_cfg["image_size"])
    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_text_length=data_cfg["max_text_length"],
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )


def resolve_active_phase(config: Dict[str, Any], phase_override: str | None) -> tuple[str, Dict[str, Any]]:
    training_cfg = config["training"]
    phase_name = phase_override or training_cfg.get("active_phase", "phase1")
    phases = training_cfg.get("phases", {})
    if phase_name not in phases:
        raise ValueError(f"Unknown training phase: {phase_name}")
    return phase_name, phases[phase_name]


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def get_decoder_layers(model: ModularVLM) -> List[torch.nn.Module]:
    llm_body = model.llm_body

    candidate_attr_paths = (
        ("layers",),
        ("model", "layers"),
        ("decoder", "layers"),
        ("language_model", "model", "layers"),
    )

    for attr_path in candidate_attr_paths:
        current = llm_body
        for attr in attr_path:
            if not hasattr(current, attr):
                current = None
                break
            current = getattr(current, attr)
        if current is not None:
            return list(current)

    raise AttributeError("Unable to locate decoder layers in the loaded Qwen model.")


def apply_training_phase(model: ModularVLM, phase_cfg: Dict[str, Any]) -> None:
    freeze_all_parameters(model)

    if "adapter" in phase_cfg.get("trainable_modules", []):
        for param in model.adapter.parameters():
            param.requires_grad = True

    llm_last_n = int(phase_cfg.get("llm_tune_last_n_layers", 0))
    if llm_last_n > 0:
        decoder_layers = get_decoder_layers(model)
        if llm_last_n > len(decoder_layers):
            raise ValueError(
                f"Requested {llm_last_n} trainable LLM layers, but only "
                f"{len(decoder_layers)} layers were found."
            )
        for layer in decoder_layers[-llm_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(model.llm_body, "norm"):
            for param in model.llm_body.norm.parameters():
                param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True


def split_weight_decay_params(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    no_decay_terms = ("bias", "norm", "ln", "layernorm")

    for name, param in named_params:
        if not param.requires_grad:
            continue
        lowered_name = name.lower()
        if param.ndim == 1 or any(term in lowered_name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


def build_optimizer(model: ModularVLM, phase_cfg: Dict[str, Any]) -> AdamW:
    decay_params, no_decay_params = split_weight_decay_params(model.named_parameters())
    learning_rate = float(phase_cfg["learning_rate"])
    weight_decay = float(phase_cfg.get("weight_decay", 0.0))

    param_groups = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": learning_rate,
                "weight_decay": 0.0,
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters were found for the active phase.")

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def resolve_autocast_dtype(mixed_precision: str) -> torch.dtype | None:
    precision = mixed_precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def save_checkpoint(
    model: ModularVLM,
    optimizer: AdamW,
    scheduler,
    output_dir: Path,
    phase_name: str,
    step: int,
) -> Path:
    checkpoint = {
        "phase": phase_name,
        "global_step": step,
        "adapter": model.adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    llm_trainable = [
        name for name, param in model.llm.named_parameters() if param.requires_grad
    ]
    if llm_trainable:
        checkpoint["llm_trainable_state"] = {
            name: tensor.detach().cpu()
            for name, tensor in model.llm.state_dict().items()
            if name in llm_trainable
        }

    checkpoint_path = output_dir / f"{phase_name}_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def append_train_log(log_path: Path, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    phase_name, phase_cfg = resolve_active_phase(config, args.phase)
    training_cfg = config["training"]
    device = torch.device(training_cfg["device"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["llm_base"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ModularVLM(config).to(device)
    apply_training_phase(model, phase_cfg)
    dataloader = build_dataloader(config, tokenizer)
    manifest_defaults = get_unified_manifest_paths(config["data"].get("unified_root"))

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    print(f"Active phase: {phase_name} ({phase_cfg.get('name', phase_name)})")
    print(f"Train manifest: {config['data'].get('train_manifest') or manifest_defaults['train']}")
    print("Trainable parameters:")
    for name in model.trainable_parameter_names():
        print(f"  - {name}")

    if dataloader is None:
        print("No train_manifest configured. Training scaffold is ready.")
        return

    optimizer = build_optimizer(model, phase_cfg)
    total_update_steps = math.ceil(
        len(dataloader)
        * int(training_cfg["num_epochs"])
        / int(training_cfg["gradient_accumulation_steps"])
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg["warmup_steps"]),
        num_training_steps=max(total_update_steps, 1),
    )

    autocast_dtype = resolve_autocast_dtype(training_cfg.get("mixed_precision", "none"))
    use_amp = device.type == "cuda" and autocast_dtype is not None
    use_grad_scaler = use_amp and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    max_grad_norm = float(training_cfg["max_grad_norm"])
    log_every = int(training_cfg["log_every"])
    save_every = int(training_cfg["save_every"])

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optimizer_step = 0

    for epoch in range(int(training_cfg["num_epochs"])):
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model forward pass did not return a loss tensor.")
                loss = loss / grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1
            should_step = batch_idx % grad_accum_steps == 0 or batch_idx == len(dataloader)
            if not should_step:
                continue

            if use_grad_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_grad_norm,
            )

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % log_every == 0:
                effective_loss = loss.detach().item() * grad_accum_steps
                lr = scheduler.get_last_lr()[0]
                log_record = {
                    "phase": phase_name,
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": effective_loss,
                    "lr": lr,
                }
                append_train_log(log_path, log_record)
                print(
                    f"epoch={epoch} step={optimizer_step} "
                    f"loss={effective_loss:.4f} lr={lr:.6e}"
                )

            if save_every > 0 and optimizer_step % save_every == 0:
                checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    phase_name=phase_name,
                    step=optimizer_step,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

    final_checkpoint = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        phase_name=phase_name,
        step=optimizer_step,
    )
    print(f"Saved final checkpoint to {final_checkpoint}")


if __name__ == "__main__":
    main()
