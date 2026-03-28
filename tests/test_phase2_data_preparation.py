from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_scienceqa import (
    build_question_text,
    build_scienceqa_target,
    is_retained_scienceqa_record,
    normalize_choices,
)
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


def test_scienceqa_normalizes_numpy_like_choices() -> None:
    class _FakeArray:
        def tolist(self):
            return ["A", " B ", ""]

    assert normalize_choices(_FakeArray()) == ["A", "B"]


def test_scienceqa_appends_hint_to_question() -> None:
    assert build_question_text("What is shown?", "Look at the nucleus.") == (
        "What is shown?\nContext: Look at the nucleus."
    )
    assert build_question_text("What is shown?", "") == "What is shown?"


def test_scienceqa_builds_reasoning_target_with_choice_text() -> None:
    target = build_scienceqa_target(
        answer_index=1,
        choices=["mitosis", "meiosis"],
        solution="The chromosome count is halved, which matches meiosis.",
    )
    assert "The chromosome count is halved" in target
    assert target.endswith("The answer is (1): meiosis")


def test_scienceqa_filter_keeps_relevant_natural_science_topics() -> None:
    keep = is_retained_scienceqa_record(
        {"subject": "natural science", "topic": "biology"},
        allowed_subjects={"natural science"},
        allowed_topics={"biology", "physics"},
        disable_default_filter=False,
    )
    drop = is_retained_scienceqa_record(
        {"subject": "social science", "topic": "geography"},
        allowed_subjects={"natural science"},
        allowed_topics={"biology", "physics"},
        disable_default_filter=False,
    )
    assert keep is True
    assert drop is False


if __name__ == "__main__":
    test_llava_instruct_extracts_first_human_and_gpt_turn()
    test_llava_instruct_uses_fallback_prompt_when_human_turn_is_empty()
    test_vqav2_prefers_multiple_choice_answer()
    test_vqav2_falls_back_to_majority_vote()
    test_vqav2_builds_expected_coco_filename()
    test_scienceqa_normalizes_numpy_like_choices()
    test_scienceqa_appends_hint_to_question()
    test_scienceqa_builds_reasoning_target_with_choice_text()
    test_scienceqa_filter_keeps_relevant_natural_science_topics()
    print({"status": "ok"})
