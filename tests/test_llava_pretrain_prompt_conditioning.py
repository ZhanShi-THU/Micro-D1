from __future__ import annotations

import torch
from PIL import Image

from scripts.prepare_llava_pretrain import FALLBACK_PROMPT, extract_prompt_and_target
from train_pretrain import build_collate_fn


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        tokens = text.split()
        return {"input_ids": [index + 1 for index, _ in enumerate(tokens)]}


def test_extract_prompt_and_target_uses_cleaned_human_prompt() -> None:
    record = {
        "image": "000/000.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe the image in detail."},
            {"from": "gpt", "value": "A cat is sitting on a couch."},
        ],
    }
    prompt_text, target_text = extract_prompt_and_target(record)
    assert prompt_text == "Describe the image in detail."
    assert target_text == "A cat is sitting on a couch."


def test_extract_prompt_and_target_falls_back_when_human_prompt_is_empty() -> None:
    record = {
        "image": "000/000.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n<image>"},
            {"from": "gpt", "value": "A microscope image with blue staining."},
        ],
    }
    prompt_text, target_text = extract_prompt_and_target(record)
    assert prompt_text == FALLBACK_PROMPT
    assert target_text == "A microscope image with blue staining."


def test_phase1_collate_masks_prompt_and_keeps_target_supervision() -> None:
    tokenizer = DummyTokenizer()
    image_transform = lambda image: torch.zeros(3, 2, 2)
    collate_fn = build_collate_fn(
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_text_length=4,
    )

    batch = [
        {
            "image": Image.new("RGB", (2, 2)),
            "text": "one two three",
            "target_text": "alpha beta",
        }
    ]

    outputs = collate_fn(batch)
    assert outputs["input_ids"].shape == (1, 4)
    assert outputs["attention_mask"].shape == (1, 4)
    assert outputs["labels"].shape == (1, 4)

    labels = outputs["labels"][0].tolist()
    assert labels[:3] == [-100, -100, -100]
    assert labels[3] != -100


if __name__ == "__main__":
    test_extract_prompt_and_target_uses_cleaned_human_prompt()
    test_extract_prompt_and_target_falls_back_when_human_prompt_is_empty()
    test_phase1_collate_masks_prompt_and_keeps_target_supervision()
    print({"status": "ok"})
