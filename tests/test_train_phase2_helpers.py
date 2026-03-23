from __future__ import annotations

import torch.nn as nn

from models.modular_vlm import ModularVLM
from train_phase2 import (
    MixedImageTextDataset,
    format_metric_for_filename,
    has_validation_data,
    parse_float_list,
    parse_int_list,
    resolve_stage_data_config,
)


def test_parse_int_list_deduplicates_and_sorts() -> None:
    assert parse_int_list([3000, 2000, 3000, 0]) == [2000, 3000]


def test_parse_float_list_sorts_descending() -> None:
    assert parse_float_list([2.0, 3.0, 2.0, 1.8]) == [3.0, 2.0, 1.8]


def test_format_metric_for_filename_is_path_safe() -> None:
    assert format_metric_for_filename(2.5) == "2p5"


class ToyDataset:
    def __init__(self, prefix: str, size: int) -> None:
        self.prefix = prefix
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {
            "image": None,
            "text": f"{self.prefix}-{index}",
            "target_text": f"target-{index}",
            "metadata": {"source": self.prefix},
        }


def test_mixed_image_text_dataset_keeps_primary_full_and_adds_auxiliary_fraction() -> None:
    primary = ToyDataset("primary", 8)
    auxiliary = ToyDataset("aux", 3)
    mixed = MixedImageTextDataset(primary, auxiliary, auxiliary_fraction=0.2)

    assert len(mixed) == 10
    assert mixed[0]["metadata"]["phase2_mix_role"] == "primary"
    assert mixed[7]["metadata"]["phase2_mix_role"] == "primary"
    assert mixed[8]["metadata"]["phase2_mix_role"] == "auxiliary"
    assert mixed[9]["metadata"]["phase2_mix_role"] == "auxiliary"


def test_resolve_stage_data_config_prefers_stage_specific_values() -> None:
    config = {
        "data": {
            "train_manifest": "/base/train.jsonl",
            "image_size": 448,
            "max_text_length": 512,
        },
        "phase2": {
            "stages": {
                "instruct": {
                    "image_size": 336,
                    "train_manifest": "/stage/train.jsonl",
                }
            }
        },
    }
    resolved = resolve_stage_data_config(config, "instruct")
    assert resolved["train_manifest"] == "/stage/train.jsonl"
    assert resolved["image_size"] == 336
    assert resolved["max_text_length"] == 512


def test_has_validation_data_detects_val_manifest() -> None:
    assert has_validation_data({"val_manifest": "/tmp/val.jsonl"}) is True
    assert has_validation_data({"val_manifest": None}) is False
    assert has_validation_data({}) is False


class _LeafModule(nn.Module):
    pass


class _BodyWithLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LeafModule()


def test_resolve_llm_body_handles_direct_language_model_attribute() -> None:
    stub = ModularVLM.__new__(ModularVLM)
    body = _BodyWithLanguageModel()

    resolved = ModularVLM._resolve_llm_body(stub, body)

    assert resolved is body.language_model


if __name__ == "__main__":
    test_parse_int_list_deduplicates_and_sorts()
    test_parse_float_list_sorts_descending()
    test_format_metric_for_filename_is_path_safe()
    test_mixed_image_text_dataset_keeps_primary_full_and_adds_auxiliary_fraction()
    test_resolve_stage_data_config_prefers_stage_specific_values()
    test_has_validation_data_detects_val_manifest()
    test_resolve_llm_body_handles_direct_language_model_attribute()
    print({"status": "ok"})
