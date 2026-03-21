"""
LLaVA-Pretrain adapter warm-up training.

This script trains only the visual adapter using caption-style data,
separate from the VQA-focused train.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.dataset import ImageTextDataset
from models.modular_vlm import ModularVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-Pretrain adapter warm-up training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_llava.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override num_epochs from config",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs if set)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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
    """Build collate function for caption-style data.

    Format: <BOS> caption <EOS> as both input and labels (with shift).
    Visual tokens get -100 labels (no direct supervision).
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([image_transform(item["image"]) for item in batch])
        captions = [item["text"] for item in batch]

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        input_id_rows: List[List[int]] = []
        label_rows: List[List[int]] = []

        for caption in captions:
            # Tokenize caption
            caption_tokens = tokenizer(
                caption,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            # Build input: [BOS] + caption + [EOS]
            # Using BOS and EOS as special tokens for sequence boundary
            input_ids = []
            if bos_token_id is not None:
                input_ids.append(bos_token_id)
            input_ids.extend(caption_tokens)
            if eos_token_id is not None:
                input_ids.append(eos_token_id)

            # Labels: [-100 for input tokens] + [caption tokens + EOS]
            # This is teacher-forcing where visual tokens don't contribute to loss
            labels = [-100] * len(input_ids)
            for i, token_id in enumerate(caption_tokens):
                labels[i + (1 if bos_token_id is not None else 0)] = token_id
            if eos_token_id is not None:
                labels[-1] = eos_token_id

            # Truncate or pad
            if len(input_ids) > max_text_length:
                input_ids = input_ids[:max_text_length]
                labels = labels[:max_text_length]
            else:
                pad_len = max_text_length - len(input_ids)
                input_ids.extend([pad_token_id] * pad_len)
                labels.extend([-100] * pad_len)

            input_id_rows.append(input_ids)
            label_rows.append(labels)

        attention_mask = [[1 if tid != pad_token_id else 0 for tid in row] for row in input_id_rows]

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_id_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }

    return collate_fn


def build_dataloader(config: Dict[str, Any], tokenizer):
    data_cfg = config["data"]
    manifest_path = data_cfg.get("train_manifest")
    if manifest_path is None:
        print("No train_manifest configured. Exiting.")
        return None

    dataset = ImageTextDataset(
        manifest_path=manifest_path,
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
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def apply_adapter_training(model: ModularVLM) -> None:
    """Freeze everything except the adapter."""
    freeze_all_parameters(model)
    for param in model.adapter.parameters():
        param.requires_grad = True


def split_weight_decay_params(
    named_params,
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


def build_optimizer(model: ModularVLM, learning_rate: float, weight_decay: float) -> AdamW:
    decay_params, no_decay_params = split_weight_decay_params(model.named_parameters())

    param_groups = []
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": learning_rate,
            "weight_decay": 0.0,
        })

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def resolve_autocast_dtype(mixed_precision: str):
    precision = mixed_precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def save_adapter_checkpoint(
    model: ModularVLM,
    optimizer: AdamW,
    scheduler,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint = {
        "global_step": step,
        "adapter": model.adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    checkpoint_path = output_dir / f"adapter_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def append_train_log(log_path: Path, payload: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = torch.device(config["training"]["device"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["llm_base"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model
    model = ModularVLM(config).to(device)
    apply_adapter_training(model)

    # Build dataloader
    dataloader = build_dataloader(config, tokenizer)
    if dataloader is None:
        return

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.adapter.load_state_dict(ckpt["adapter"])
        if "global_step" in ckpt:
            start_step = ckpt["global_step"]
        print(f"Resumed from step {start_step}")

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    # Build optimizer and scheduler
    training_cfg = config["training"]
    optimizer = build_optimizer(
        model,
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = args.num_epochs if args.num_epochs is not None else int(training_cfg["num_epochs"])
    max_steps = args.max_steps
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])

    if max_steps is not None:
        total_update_steps = max_steps
    else:
        total_update_steps = math.ceil(
            len(dataloader) * num_epochs / grad_accum_steps
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg["warmup_steps"]),
        num_training_steps=max(total_update_steps, 1),
    )

    # Mixed precision
    autocast_dtype = resolve_autocast_dtype(training_cfg.get("mixed_precision", "none"))
    use_amp = device.type == "cuda" and autocast_dtype is not None
    use_grad_scaler = use_amp and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    max_grad_norm = float(training_cfg["max_grad_norm"])
    log_every = int(training_cfg["log_every"])
    save_every = int(training_cfg["save_every"])

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    optimizer_step = 0

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if max_steps is not None and optimizer_step >= max_steps:
                break

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
            should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader)
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
                checkpoint_path = save_adapter_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    step=optimizer_step,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        if max_steps is not None and optimizer_step >= max_steps:
            break

    # Save final checkpoint
    final_checkpoint = save_adapter_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        step=optimizer_step,
    )
    print(f"Saved final checkpoint to {final_checkpoint}")
    print("Training complete.")


if __name__ == "__main__":
    main()
