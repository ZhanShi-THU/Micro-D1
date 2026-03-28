from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.phase3 import parse_choice_answer, sample_eval_indices


def test_parse_choice_answer_reads_expected_format() -> None:
    assert parse_choice_answer("The answer is (2)") == 2
    assert parse_choice_answer("the answer is 3") == 3
    assert parse_choice_answer("(4)") == 4
    assert parse_choice_answer("(1) because of the observed morphology") == 1
    assert parse_choice_answer("I think it is option two") is None


def test_sample_eval_indices_is_deterministic_and_bounded() -> None:
    first = sample_eval_indices(dataset_size=100, max_samples=8, seed=42)
    second = sample_eval_indices(dataset_size=100, max_samples=8, seed=42)
    assert first == second
    assert len(first) == 8
    assert all(0 <= index < 100 for index in first)


if __name__ == "__main__":
    test_parse_choice_answer_reads_expected_format()
    test_sample_eval_indices_is_deterministic_and_bounded()
    print({"status": "ok"})
