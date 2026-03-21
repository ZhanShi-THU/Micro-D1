from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from data.dataset import ImageTextDataset


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_caption_dataset_loads_images_lazily() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "sample.jpg"
        Image.new("RGB", (4, 4), color="white").save(image_path)

        manifest_path = root / "caption.jsonl"
        write_jsonl(
            manifest_path,
            [
                {
                    "image": str(image_path),
                    "text": "Describe the image.",
                    "target_text": "A white square.",
                }
            ],
        )

        dataset = ImageTextDataset(str(manifest_path))
        assert dataset.samples[0]["image_path"] == str(image_path)
        assert "image" not in dataset.samples[0]

        sample = dataset[0]
        assert sample["image"].size == (4, 4)
        assert sample["text"] == "Describe the image."
        assert sample["target_text"] == "A white square."


def test_microvqa_dataset_keeps_paths_until_getitem() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "micro.jpg"
        Image.new("RGB", (6, 6), color="black").save(image_path)

        manifest_path = root / "micro.jsonl"
        write_jsonl(
            manifest_path,
            [
                {
                    "image_path": str(image_path),
                    "question": "What color is the image?",
                    "choices": ["white", "black"],
                    "correct_index": 1,
                }
            ],
        )

        dataset = ImageTextDataset(str(manifest_path))
        assert dataset.samples[0]["image_path"] == str(image_path)
        assert "image" not in dataset.samples[0]

        sample = dataset[0]
        assert sample["image"].size == (6, 6)
        assert sample["target_text"] == "The answer is (1)"
        assert "What color is the image?" in sample["text"]


if __name__ == "__main__":
    test_caption_dataset_loads_images_lazily()
    test_microvqa_dataset_keeps_paths_until_getitem()
    print({"status": "ok"})
