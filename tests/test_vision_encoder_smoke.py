from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from models.vision_encoder import VisionEncoder


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

    encoder = VisionEncoder(
        backbone_name=model_cfg["vision_backbone"],
        embed_dim_dino=model_cfg["embed_dim_dino"],
        alignment_dim=model_cfg["alignment_dim"],
        alignment_head_weights=model_cfg["alignment_head_weights"],
        vision_source=model_cfg.get("vision_source", "torch_hub"),
        vision_repo=model_cfg.get("vision_repo", "facebookresearch/dinov3"),
        vision_model_name=model_cfg.get("vision_model_name", "dinov3_vitl16"),
    ).to(device)
    encoder.eval()

    pixel_values = torch.randn(
        1,
        3,
        config["data"]["image_size"],
        config["data"]["image_size"],
        device=device,
    )
    with torch.inference_mode():
        aligned_tokens = encoder(pixel_values)

    expected_last_dim = model_cfg["alignment_dim"]
    assert aligned_tokens.ndim == 3, aligned_tokens.shape
    assert aligned_tokens.shape[0] == 1, aligned_tokens.shape
    assert aligned_tokens.shape[-1] == expected_last_dim, aligned_tokens.shape
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
