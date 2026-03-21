"""
LLaVA-Pretrain adapter warm-up training.

This script trains only the visual adapter using prompt-conditioned
caption/instruction data, separate from the VQA-focused train.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
import time
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.dataset import ImageTextDataset
from models.modular_vlm import ModularVLM

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


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
    parser.add_argument(
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
        help="Local rank provided by torchrun.",
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


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def print_main(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def setup_distributed(args: argparse.Namespace, config: Dict[str, Any]) -> torch.device:
    requested_device = str(config["training"].get("device", "cuda")).lower()
    local_rank = int(args.local_rank)

    if local_rank >= 0:
        if requested_device != "cuda":
            raise ValueError("Distributed phase-1 training currently expects training.device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun provided LOCAL_RANK but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank)

    return torch.device(config["training"]["device"])


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def maybe_reduce_scalar(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())


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
    """Build collate function for prompt-conditioned phase-1 data.

    Each sample is trained as:
      input_ids = prompt_ids + target_ids (+ eos)
      labels    = [-100] * len(prompt_ids) + target_ids (+ eos)

    Visual prefix tokens are supervised implicitly through the same LM loss path
    already implemented in the model. Prompt tokens never contribute to loss.
    """

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

        input_id_rows: List[List[int]] = []
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
                raise ValueError("Phase-1 target_text must produce at least one target token.")
            if max_text_length <= 0:
                raise ValueError(f"max_text_length must be positive, got {max_text_length}")

            # Preserve at least one supervised target token when possible.
            max_prompt_len = max_text_length - 1
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

            available_target_len = max_text_length - len(prompt_ids)
            if available_target_len <= 0:
                raise RuntimeError("Unable to allocate space for target tokens in phase-1 collation.")

            target_ids = target_ids[:available_target_len]
            if not target_ids:
                raise RuntimeError("Prompt truncation failed to preserve any target tokens.")

            combined_ids = prompt_ids + target_ids
            combined_labels = ([-100] * len(prompt_ids)) + target_ids
            combined_attention = [1] * len(combined_ids)

            pad_len = max_text_length - len(combined_ids)
            if pad_len < 0:
                raise RuntimeError("Combined prompt/target sequence exceeded max_text_length unexpectedly.")

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
    manifest_path = data_cfg.get("train_manifest")
    if manifest_path is None:
        print_main("No train_manifest configured. Exiting.")
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

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
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


def maybe_init_wandb(config: Dict[str, Any], args: argparse.Namespace):
    if not is_main_process():
        return None

    wandb_cfg = dict(config.get("training", {}).get("wandb", {}))
    if not wandb_cfg.get("enabled", False):
        return None

    if wandb is None:
        print("wandb logging was requested but wandb is not installed. Continuing without wandb.")
        return None

    run_name = wandb_cfg.get("name") or config.get("project_name", "llava_pretrain_adapter_warmup")
    if args.max_steps is not None:
        run_name = f"{run_name}-maxsteps{args.max_steps}"

    return wandb.init(
        project=wandb_cfg.get("project", "microvqa"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("resolved_run_name", run_name),
        tags=wandb_cfg.get("tags"),
        mode=wandb_cfg.get("mode"),
        config=config,
    )


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
        run_name = output_dir.name
    elif not use_run_subdir:
        output_dir = base_output_dir
        run_name = configured_run_name or base_output_dir.name
    else:
        run_name = configured_run_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.get('project_name', 'run')}_{timestamp}"
        output_dir = base_output_dir / run_name

    resolved = [str(output_dir), run_name]
    if is_distributed():
        dist.broadcast_object_list(resolved, src=0)

    return Path(resolved[0]), str(resolved[1])


def write_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    run_name: str,
    args: argparse.Namespace,
) -> None:
    resolved_config_path = output_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    run_info = {
        "run_name": run_name,
        "config_path": args.config,
        "resume": args.resume,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)


def save_adapter_checkpoint(
    model: ModularVLM,
    optimizer: AdamW,
    scheduler,
    output_dir: Path,
    step: int,
) -> Path:
    unwrap_model = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "global_step": step,
        "adapter": unwrap_model.adapter.state_dict(),
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

    device = setup_distributed(args, config)
    output_dir, run_name = resolve_run_output_dir(args, config)
    config["training"]["resolved_output_dir"] = str(output_dir)
    config["training"]["resolved_run_name"] = run_name
    wandb_cfg = dict(config.get("training", {}).get("wandb", {}))
    wandb_cfg["resolved_run_name"] = run_name
    config["training"]["wandb"] = wandb_cfg
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(output_dir=output_dir, config=config, run_name=run_name, args=args)
    if is_distributed():
        dist.barrier()
    log_path = output_dir / "train_log.jsonl"
    train_cfg = config["training"]
    data_cfg = config["data"]

    print_main(f"[startup] config={args.config}")
    print_main(f"[startup] output_dir={output_dir}")
    print_main(f"[startup] run_name={run_name}")
    print_main(
        f"[startup] device={device} rank={get_rank()} world_size={get_world_size()} "
        f"local_rank={args.local_rank}"
    )
    print_main(
        "[startup] data_settings="
        f"manifest={data_cfg.get('train_manifest')} "
        f"image_size={data_cfg.get('image_size')} "
        f"max_text_length={data_cfg.get('max_text_length')} "
        f"num_workers={data_cfg.get('num_workers', 4)}"
    )
    print_main(
        "[startup] training_settings="
        f"per_device_batch_size={train_cfg.get('batch_size')} "
        f"grad_accum={train_cfg.get('gradient_accumulation_steps')} "
        f"epochs={args.num_epochs if args.num_epochs is not None else train_cfg.get('num_epochs')} "
        f"lr={train_cfg.get('learning_rate')} "
        f"mixed_precision={train_cfg.get('mixed_precision', 'none')}"
    )
    print_main(
        "[startup] llm_loading="
        f"quantization={config['model'].get('llm_quantization', 'none')} "
        f"deepstack={config['model'].get('use_deepstack_injection', True)}"
    )
    print_main(
        "[startup] effective_global_batch="
        f"{int(train_cfg.get('batch_size', 1)) * int(train_cfg.get('gradient_accumulation_steps', 1)) * get_world_size()}"
    )

    print_main("[startup] loading tokenizer...")
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["llm_base"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_main(f"[startup] tokenizer ready in {time.perf_counter() - start_time:.1f}s")

    print_main("[startup] building model...")
    start_time = time.perf_counter()
    model = ModularVLM(config)
    model = model.prepare_for_training_device(device)
    apply_adapter_training(model)
    if is_distributed():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    print_main(f"[startup] model ready in {time.perf_counter() - start_time:.1f}s")

    print_main("[startup] building dataloader...")
    start_time = time.perf_counter()
    dataloader = build_dataloader(config, tokenizer)
    if dataloader is None:
        cleanup_distributed()
        return
    dataset_size = len(dataloader.dataset)
    print_main(
        f"[startup] dataloader ready in {time.perf_counter() - start_time:.1f}s "
        f"with {dataset_size} samples, {len(dataloader)} local batches/epoch, "
        f"world_size={get_world_size()}"
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        unwrap_model = model.module if isinstance(model, DDP) else model
        unwrap_model.adapter.load_state_dict(ckpt["adapter"])
        if "global_step" in ckpt:
            start_step = ckpt["global_step"]
        print_main(f"Resumed from step {start_step}")

    print_main("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_main(f"  - {name}")

    optimizer = build_optimizer(
        model,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = args.num_epochs if args.num_epochs is not None else int(train_cfg["num_epochs"])
    max_steps = args.max_steps
    grad_accum_steps = int(train_cfg["gradient_accumulation_steps"])

    if max_steps is not None:
        total_update_steps = max_steps
    else:
        total_update_steps = math.ceil(
            len(dataloader) * num_epochs / grad_accum_steps
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg["warmup_steps"]),
        num_training_steps=max(total_update_steps, 1),
    )
    print_main(
        "[startup] optimization="
        f"total_update_steps={total_update_steps} "
        f"warmup_steps={train_cfg.get('warmup_steps')} "
        f"max_steps={max_steps}"
    )

    autocast_dtype = resolve_autocast_dtype(train_cfg.get("mixed_precision", "none"))
    use_amp = device.type == "cuda" and autocast_dtype is not None
    use_grad_scaler = use_amp and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    max_grad_norm = float(train_cfg["max_grad_norm"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    wandb_run = maybe_init_wandb(config, args)
    if wandb_run is not None:
        wandb_run.summary["dataset_size"] = dataset_size
        wandb_run.summary["batches_per_epoch"] = len(dataloader)
        wandb_run.summary["total_update_steps"] = total_update_steps

    model.train()
    optimizer.zero_grad(set_to_none=True)
    print_main("[startup] starting training loop...")

    global_step = start_step
    optimizer_step = 0

    for epoch in range(num_epochs):
        if is_distributed() and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        print_main(f"[epoch] starting epoch {epoch + 1}/{num_epochs}")
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

            effective_loss = loss.detach().item() * grad_accum_steps
            reduced_loss = maybe_reduce_scalar(effective_loss, device)
            lr = scheduler.get_last_lr()[0]
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": reduced_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/optimizer_step": optimizer_step,
                    },
                    step=optimizer_step,
                )

            if optimizer_step % log_every == 0:
                log_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": reduced_loss,
                    "lr": lr,
                }
                if is_main_process():
                    append_train_log(log_path, log_record)
                print_main(
                    f"epoch={epoch} step={optimizer_step} "
                    f"loss={reduced_loss:.4f} lr={lr:.6e}"
                )

            if save_every > 0 and optimizer_step % save_every == 0 and is_main_process():
                checkpoint_path = save_adapter_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    step=optimizer_step,
                )
                print_main(f"Saved checkpoint to {checkpoint_path}")

        if max_steps is not None and optimizer_step >= max_steps:
            break

    final_checkpoint = None
    if is_main_process():
        final_checkpoint = save_adapter_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            step=optimizer_step,
        )
        print_main(f"Saved final checkpoint to {final_checkpoint}")
        print_main("Training complete.")
    if wandb_run is not None:
        if final_checkpoint is not None:
            wandb_run.summary["final_checkpoint"] = str(final_checkpoint)
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
