from __future__ import annotations

from scripts.prepare_llava_instruct import FALLBACK_PROMPT, extract_prompt_and_target
from scripts.prepare_vqav2 import build_image_filename, choose_target_answer


def test_llava_instruct_extracts_first_human_and_gpt_turn() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is happening here?"},
            {"from": "gpt", "value": "A dog is jumping over a log."},
            {"from": "human", "value": "ignored"},
        ]
    }
    prompt_text, target_text = extract_prompt_and_target(record)  # type: ignore[misc]
    assert prompt_text == "What is happening here?"
    assert target_text == "A dog is jumping over a log."


def test_llava_instruct_uses_fallback_prompt_when_human_turn_is_empty() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "<image>"},
            {"from": "gpt", "value": "Cells are densely packed."},
        ]
    }
    prompt_text, target_text = extract_prompt_and_target(record)  # type: ignore[misc]
    assert prompt_text == FALLBACK_PROMPT
    assert target_text == "Cells are densely packed."


def test_vqav2_prefers_multiple_choice_answer() -> None:
    annotation = {
        "multiple_choice_answer": "blue",
        "answers": [
            {"answer": "red"},
            {"answer": "blue"},
        ],
    }
    assert choose_target_answer(annotation) == "blue"


def test_vqav2_falls_back_to_majority_vote() -> None:
    annotation = {
        "answers": [
            {"answer": "cat"},
            {"answer": "cat"},
            {"answer": "dog"},
        ]
    }
    assert choose_target_answer(annotation) == "cat"


def test_vqav2_builds_expected_coco_filename() -> None:
    assert build_image_filename(42, "train2014") == "COCO_train2014_000000000042.jpg"


if __name__ == "__main__":
    test_llava_instruct_extracts_first_human_and_gpt_turn()
    test_llava_instruct_uses_fallback_prompt_when_human_turn_is_empty()
    test_vqav2_prefers_multiple_choice_answer()
    test_vqav2_falls_back_to_majority_vote()
    test_vqav2_builds_expected_coco_filename()
    print({"status": "ok"})
