from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.dataset import build_microvqa_prompt, build_microvqa_target
from data.unified_vqa import build_unified_record, infer_answer_index, resplit_records
from scripts.prepare_unified_vqa import prepare_mms, prepare_mmsci, repair_embedded_choices


class UnifiedVQAPrepareTests(unittest.TestCase):
    def test_prompt_and_target_use_zero_based_indices(self) -> None:
        prompt = build_microvqa_prompt("What is shown?", ["cat", "dog"])
        self.assertIn("(0): cat", prompt)
        self.assertIn("(1): dog", prompt)
        self.assertEqual(build_microvqa_target(1, ["cat", "dog"]), "The answer is (1): dog")

    def test_answer_only_prompt_style_skips_step_by_step_instruction(self) -> None:
        prompt = build_microvqa_prompt(
            "What is shown?",
            ["cat", "dog"],
            prompt_style="answer_only",
        )
        self.assertNotIn("Think step by step", prompt)
        self.assertIn('The answer is (X)', prompt)

    def test_infer_answer_index(self) -> None:
        self.assertEqual(infer_answer_index("dog", ["cat", "dog", "bird"]), 1)

    def test_prepare_mms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "sample.png").write_bytes(b"fake")
            payload = [
                {
                    "key_question": "mms_q_1",
                    "key_image": "mms_img_1",
                    "question": "What is this?",
                    "choices": ["A", "B", "C"],
                    "correct_index": 2,
                    "image_path": "images/sample.png",
                    "task": "biology",
                }
            ]
            with open(root / "microvqa_custom_test.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            records = prepare_mms(root, root / "out", "test")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_dataset"], "mms")
            self.assertEqual(records[0]["correct_answer"], "C")
            self.assertTrue(records[0]["image_path"].endswith("images/mms/test/sample.png"))
            self.assertTrue(
                records[0]["metadata"]["source_path"].endswith(
                    "sources/mms/microvqa_custom_test.json"
                )
            )

    def test_repair_embedded_choices(self) -> None:
        repaired = repair_embedded_choices(
            [
                "sub-figure (a)",
                "sub-figure (b)\nC: sub-figure (c)\nD: sub-figure (d)\nAnswer with the option letter directly.",
            ]
        )
        self.assertEqual(
            repaired,
            ["sub-figure (a)", "sub-figure (b)", "sub-figure (c)", "sub-figure (d)"],
        )

    def test_prepare_mmsci(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "sample.png").write_bytes(b"fake")
            sample = {
                "uid": "uid-1",
                "image": "images/sample.png",
                "question": "Choose the right option",
                "options": ["x", "y", "z"],
                "answer": "y",
                "caption": "caption",
                "conversations": [],
                "category": "cat",
                "subject": "subj",
            }
            with open(root / "generated_mcq.jsonl", "w", encoding="utf-8") as handle:
                handle.write(json.dumps(sample) + "\n")

            records = prepare_mmsci(root, root / "out", "train")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_dataset"], "mmsci++")
            self.assertEqual(records[0]["correct_index"], 1)
            self.assertEqual(records[0]["correct_answer"], "y")
            self.assertTrue(
                records[0]["image_path"].endswith("images/mmsci++/train/sample.png")
            )
            self.assertTrue(
                records[0]["metadata"]["source_path"].endswith(
                    "sources/mmsci++/generated_mcq.jsonl"
                )
            )

    def test_build_unified_record(self) -> None:
        record = build_unified_record(
            sample_id="sample-1",
            source_dataset="demo",
            split="test",
            image_path="/tmp/image.png",
            question="Question?",
            choices=["left", "right"],
            correct_index=0,
            metadata={"a": 1},
        )
        self.assertEqual(record["target_text"], "The answer is (0): left")
        self.assertEqual(record["metadata"]["a"], 1)

    def test_resplit_records_groups_by_image_identifier(self) -> None:
        records = [
            build_unified_record(
                sample_id="a1",
                source_dataset="microbench",
                split="test",
                image_path="/tmp/a.png",
                question="Q1",
                choices=["x", "y"],
                correct_index=0,
                metadata={"image_id": "img_a"},
            ),
            build_unified_record(
                sample_id="a2",
                source_dataset="microbench",
                split="test",
                image_path="/tmp/a.png",
                question="Q2",
                choices=["x", "y"],
                correct_index=1,
                metadata={"image_id": "img_a"},
            ),
            build_unified_record(
                sample_id="b1",
                source_dataset="microbench",
                split="test",
                image_path="/tmp/b.png",
                question="Q3",
                choices=["x", "y"],
                correct_index=0,
                metadata={"image_id": "img_b"},
            ),
        ]

        resplit = resplit_records(records, train_ratio=0.5, seed=123)
        split_by_image = {}
        for record in resplit:
            split_by_image.setdefault(record["metadata"]["image_id"], set()).add(record["split"])

        self.assertEqual(len(split_by_image["img_a"]), 1)
        self.assertEqual(len(split_by_image["img_b"]), 1)


if __name__ == "__main__":
    unittest.main()
