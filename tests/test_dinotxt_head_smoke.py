from __future__ import annotations

import argparse

import torch
import yaml

from models.vision_encoder import DINOTextAlignmentHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/user/Project_files/project/configs/qwen3_dinov3.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_cfg = config["model"]

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    head = DINOTextAlignmentHead(
        input_dim=model_cfg["embed_dim_dino"],
        output_dim=model_cfg["alignment_dim"],
    )
    head.load_pretrained(model_cfg["alignment_head_weights"])
    head = head.to(device).eval()

    visual_tokens = torch.randn(1, 32, model_cfg["embed_dim_dino"], device=device)
    with torch.inference_mode():
        aligned_tokens = head(visual_tokens)

    assert aligned_tokens.shape == visual_tokens.shape
    print(
        {
            "status": "ok",
            "shape": list(aligned_tokens.shape),
            "dtype": str(aligned_tokens.dtype),
            "device": str(aligned_tokens.device),
        }
    )


if __name__ == "__main__":
    main()
