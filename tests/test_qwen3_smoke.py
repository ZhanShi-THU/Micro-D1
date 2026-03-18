from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/user/Project_files/project/local_models/qwen3-vl-4b",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image briefly."},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = outputs[:, inputs.shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(
        {
            "status": "ok",
            "model_path": str(model_path),
            "generated_text": text,
        }
    )


if __name__ == "__main__":
    main()
