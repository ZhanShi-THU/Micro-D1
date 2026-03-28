from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from models.modular_vlm import ModularVLM


ANSWER_REGEX = re.compile(
    r"(?:^|\b)(?:the\s+)?answer(?:\s+is|:)?\s*\*?\*?\(?([0-9]+)\)?|^\s*\(([0-9]+)\)",
    re.IGNORECASE,
)


@dataclass
class GreedyGenerationResult:
    text: str
    generated_token_count: int
    stopped_by_eos: bool


def parse_choice_answer(text: str) -> int | None:
    match = ANSWER_REGEX.search(text)
    if match is None:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def _decode_text(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def greedy_generate(
    *,
    model: ModularVLM,
    tokenizer,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_on_first_parsed_answer: bool = False,
) -> GreedyGenerationResult:
    model_inputs = model.build_multimodal_inputs(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
    )
    inputs_embeds = model_inputs["inputs_embeds"]
    merged_attention_mask = model_inputs["attention_mask"]
    visual_pos_masks = model_inputs.get("visual_pos_masks")
    deepstack_visual_embeds = model_inputs.get("deepstack_visual_embeds")
    device = inputs_embeds.device

    generated_ids: list[int] = []
    stopped_by_eos = False

    decoder_outputs = model.llm_body.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=merged_attention_mask,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        use_cache=True,
    )
    hidden_states = decoder_outputs[0]
    past_key_values = getattr(decoder_outputs, "past_key_values", None)

    for _ in range(max_new_tokens):
        logits = model.lm_head(hidden_states)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        token_id = int(next_token_id.item())

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            stopped_by_eos = True
            break

        generated_ids.append(token_id)
        current_text = _decode_text(tokenizer, generated_ids)

        next_embed = model.get_text_embeds(next_token_id.unsqueeze(0))
        next_attention = torch.ones((1, 1), dtype=merged_attention_mask.dtype, device=device)
        merged_attention_mask = torch.cat([merged_attention_mask, next_attention], dim=1)
        next_visual_pos_mask = None
        if visual_pos_masks is not None:
            next_visual_pos_mask = torch.zeros((visual_pos_masks.size(0), 1), dtype=torch.bool, device=device)

        decoder_outputs = model.llm_body.forward(
            inputs_embeds=next_embed,
            attention_mask=merged_attention_mask,
            visual_pos_masks=next_visual_pos_mask,
            deepstack_visual_embeds=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = decoder_outputs[0]
        past_key_values = getattr(decoder_outputs, "past_key_values", None)

    return GreedyGenerationResult(
        text=_decode_text(tokenizer, generated_ids),
        generated_token_count=len(generated_ids),
        stopped_by_eos=stopped_by_eos
    )
