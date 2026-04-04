from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import nn


def _format_keys(keys: list[str], limit: int = 8) -> str:
    if not keys:
        return "[]"
    preview = keys[:limit]
    suffix = "" if len(keys) <= limit else f", ... (+{len(keys) - limit} more)"
    return "[" + ", ".join(preview) + suffix + "]"


def load_matching_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    module_name: str,
    strict: bool,
) -> Dict[str, Any]:
    current_state = module.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    mismatched_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        if current_state[key].shape != value.shape:
            mismatched_keys.append(
                f"{key}: checkpoint{tuple(value.shape)} != model{tuple(current_state[key].shape)}"
            )
            continue
        filtered_state[key] = value

    missing_keys = [key for key in current_state.keys() if key not in filtered_state]

    if strict and (unexpected_keys or mismatched_keys or missing_keys):
        raise RuntimeError(
            f"Failed to strictly load {module_name} state dict. "
            f"missing={_format_keys(missing_keys)} "
            f"unexpected={_format_keys(unexpected_keys)} "
            f"mismatched={_format_keys(mismatched_keys)}"
        )

    module.load_state_dict(filtered_state, strict=False)
    return {
        "loaded_keys": sorted(filtered_state.keys()),
        "missing_keys": sorted(missing_keys),
        "unexpected_keys": sorted(unexpected_keys),
        "mismatched_keys": sorted(mismatched_keys),
    }
