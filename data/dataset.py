from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset

from data.unified_vqa import (
    build_multiple_choice_prompt,
    build_multiple_choice_target,
)


def format_microvqa_choices(choices: Sequence[str]) -> str:
    from data.unified_vqa import format_choices_for_prompt

    return format_choices_for_prompt(choices)


def build_microvqa_prompt(
    question: str,
    choices: Sequence[str],
    prompt_style: str = "reasoning",
) -> str:
    return build_multiple_choice_prompt(question, choices, prompt_style=prompt_style)


def build_microvqa_target(correct_index: int) -> str:
    return build_multiple_choice_target(correct_index)


class ImageTextDataset(Dataset):
    """
    Supports two manifest styles:

    1. Caption-style jsonl
       Required keys:
         - image
         - text
       Optional keys:
         - target_text

    2. microvqa-style jsonl
       Required keys:
         - question
         - choices
         - correct_index
       Image keys:
         - images_list, or
         - image, or
         - image_path
       Optional keys:
         - key_question
         - key_image
         - task

    Each line must be a JSON object.
    """

    def __init__(
        self,
        manifest_path: str,
        image_root: str | None = None,
        prompt_style: str = "reasoning",
    ) -> None:
        self.image_root = Path(image_root) if image_root else None
        self.prompt_style = prompt_style
        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                samples.append(self._normalize_sample(sample))
        return samples

    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "question" in sample and "choices" in sample:
            return self._normalize_microvqa_sample(sample)
        if "text" in sample:
            return self._normalize_caption_sample(sample)
        raise KeyError(
            "Unsupported manifest record. Expected caption keys "
            "('image', 'text') or microvqa keys ('question', 'choices', 'correct_index')."
        )

    def _normalize_caption_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = sample.get("image") or sample.get("image_path")
        if image_path is None:
            raise KeyError("Caption sample is missing 'image' or 'image_path'.")
        return {
            "sample_type": "caption",
            "image_path": str(image_path),
            "prompt_text": sample["text"],
            "target_text": sample.get("target_text", sample["text"]),
            "metadata": {
                **(sample.get("metadata") or {}),
                "image_path": str(image_path),
            },
        }

    def _normalize_microvqa_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "correct_index" not in sample:
            raise KeyError("microvqa sample is missing 'correct_index'.")

        choices = sample["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("microvqa sample 'choices' must be a non-empty list.")

        prompt_text = build_microvqa_prompt(
            sample["question"],
            choices,
            prompt_style=self.prompt_style,
        )
        target_text = sample.get(
            "target_text",
            build_microvqa_target(int(sample["correct_index"])),
        )

        image_path = self._resolve_microvqa_image_path(sample)
        return {
            "sample_type": "microvqa",
            "image_path": image_path,
            "prompt_text": prompt_text,
            "target_text": target_text,
            "metadata": {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset"),
                "split": sample.get("split"),
                "image_path": sample.get("image") or sample.get("image_path"),
                "key_question": sample.get("key_question"),
                "key_image": sample.get("key_image"),
                "task": sample.get("task"),
                "question": sample["question"],
                "choices": choices,
                "correct_index": int(sample["correct_index"]),
            },
        }

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.image_root is None:
            return path
        return self.image_root / path

    def _load_image(self, image_path: str) -> Image.Image:
        resolved = self._resolve_image_path(image_path)
        return Image.open(resolved).convert("RGB")

    def _resolve_microvqa_image_path(self, sample: Dict[str, Any]) -> str:
        if "images_list" in sample:
            images_list = sample["images_list"]
            if not isinstance(images_list, list) or not images_list:
                raise ValueError("'images_list' must be a non-empty list.")
            first_image = images_list[0]
            if isinstance(first_image, str):
                return first_image
            if isinstance(first_image, dict) and "path" in first_image:
                return str(first_image["path"])
            raise TypeError(
                "Only path-based microvqa images are supported in manifest files."
            )

        image_path = sample.get("image") or sample.get("image_path")
        if image_path is None:
            raise KeyError(
                "microvqa sample is missing 'images_list', 'image', or 'image_path'."
            )
        return str(image_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "image": self._load_image(sample["image_path"]),
            "text": sample["prompt_text"],
            "target_text": sample["target_text"],
            "sample_type": sample["sample_type"],
            "metadata": sample["metadata"],
        }
