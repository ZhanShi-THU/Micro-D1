from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.adapter import VisualAdapter
from models.vision_encoder import VisionEncoder


class ModularVLM(nn.Module):
    """
    Core glue layer for the modular VLM.

    Flow:
      1. frozen vision encoder: image -> aligned visual tokens
      2. trainable adapter: aligned visual tokens -> Qwen embedding space
      3. frozen Qwen embedding layer: input_ids -> text embeddings
      4. concat visual/text embeddings along sequence dimension
      5. call the LLM decoder with a merged attention mask

    Visual tokens are treated as a prefix sequence so they can interact with
    text tokens through the LLM self-attention stack.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        self.model_cfg = model_cfg
        self.config = config

        vision_backbone = model_cfg.get("vision_backbone") or model_cfg.get(
            "vision_model_name",
            "dinov3_vitl16",
        )
        model_cfg["vision_backbone"] = vision_backbone

        llm_local_only = Path(model_cfg["llm_base"]).exists()
        llm_config = AutoConfig.from_pretrained(
            model_cfg["llm_base"],
            trust_remote_code=True,
            local_files_only=llm_local_only,
        )
        self.hidden_size_qwen = self._resolve_llm_hidden_size(llm_config)
        configured_hidden_size = model_cfg.get("hidden_size_qwen")
        if configured_hidden_size is None:
            model_cfg["hidden_size_qwen"] = self.hidden_size_qwen
        elif int(configured_hidden_size) != self.hidden_size_qwen:
            raise ValueError(
                "hidden_size_qwen does not match the loaded Qwen3-VL text hidden size: "
                f"config={configured_hidden_size}, model={self.hidden_size_qwen}"
            )

        embed_dim_dino = int(model_cfg["embed_dim_dino"])
        alignment_dim = int(model_cfg["alignment_dim"])
        if embed_dim_dino != alignment_dim:
            raise ValueError(
                "DINO/dinotxt dimensions must match for the current alignment head: "
                f"embed_dim_dino={embed_dim_dino}, alignment_dim={alignment_dim}"
            )

        self.vision_encoder = VisionEncoder(
            backbone_name=vision_backbone,
            embed_dim_dino=embed_dim_dino,
            alignment_dim=alignment_dim,
            alignment_head_weights=model_cfg.get("alignment_head_weights"),
            vision_source=model_cfg.get("vision_source", "torch_hub"),
            vision_repo=model_cfg.get("vision_repo", "facebookresearch/dinov3"),
            vision_model_name=model_cfg.get("vision_model_name", "dinov3_vitl16"),
            vision_pretrained=model_cfg.get("vision_pretrained", True),
            vision_checkpoint_path=model_cfg.get("vision_checkpoint_path"),
        )
        self.adapter = VisualAdapter(
            input_dim=alignment_dim,
            output_dim=self.hidden_size_qwen,
            dropout=model_cfg.get("adapter_dropout", 0.0),
        )
        llm_dtype = self._resolve_llm_dtype(config)
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_cfg["llm_base"],
            trust_remote_code=True,
            local_files_only=llm_local_only,
            dtype=llm_dtype,
        )
        self.llm_body = self._resolve_llm_body(self.llm)
        self.lm_head = self._resolve_lm_head(self.llm)

        self.freeze_modules(
            freeze_strategy=model_cfg.get("freeze_strategy", []),
            trainable_modules=model_cfg.get("trainable_modules", ["adapter"]),
        )

    def _resolve_llm_hidden_size(self, llm_config: Any) -> int:
        text_config = getattr(llm_config, "text_config", None)
        if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
            return int(text_config.hidden_size)
        if getattr(llm_config, "hidden_size", None) is not None:
            return int(llm_config.hidden_size)
        raise ValueError("Unable to determine the Qwen3-VL text hidden size from config.")

    def _resolve_llm_dtype(self, config: Dict[str, Any]) -> torch.dtype | None:
        explicit_dtype = str(self.model_cfg.get("llm_dtype", "")).lower()
        if explicit_dtype:
            if explicit_dtype == "bf16":
                return torch.bfloat16
            if explicit_dtype == "fp16":
                return torch.float16
            if explicit_dtype in {"fp32", "float32"}:
                return torch.float32
            raise ValueError(f"Unsupported llm_dtype: {self.model_cfg['llm_dtype']}")

        mixed_precision = str(config.get("training", {}).get("mixed_precision", "none")).lower()
        if mixed_precision == "bf16":
            return torch.bfloat16
        if mixed_precision == "fp16":
            return torch.float16
        return None

    def _resolve_llm_body(self, llm: nn.Module) -> nn.Module:
        if hasattr(llm, "model") and hasattr(llm.model, "language_model"):
            return llm.model.language_model
        if hasattr(llm, "model"):
            return llm.model
        if hasattr(llm, "get_base_model"):
            return llm.get_base_model()
        raise AttributeError("Unable to locate the base decoder model inside the LLM.")

    def _resolve_lm_head(self, llm: nn.Module) -> nn.Module:
        if hasattr(llm, "lm_head"):
            return llm.lm_head
        output_embeddings = llm.get_output_embeddings()
        if output_embeddings is None:
            raise AttributeError("Unable to locate the output projection head of the LLM.")
        return output_embeddings

    def freeze_modules(
        self,
        freeze_strategy: Iterable[str],
        trainable_modules: Iterable[str],
    ) -> None:
        module_map = {
            "vision_backbone": self.vision_encoder.backbone,
            "alignment_head": self.vision_encoder.alignment_head,
            "llm_base": self.llm,
            "adapter": self.adapter,
        }

        for module_name in freeze_strategy:
            module = module_map.get(module_name)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

        for module_name in trainable_modules:
            module = module_map.get(module_name)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

    def get_visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        aligned_tokens = self.vision_encoder(pixel_values)
        visual_embeds = self.adapter(aligned_tokens)
        llm_embed_dtype = self.llm.get_input_embeddings().weight.dtype
        if visual_embeds.dtype != llm_embed_dtype:
            visual_embeds = visual_embeds.to(llm_embed_dtype)
        return visual_embeds

    def get_text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding_layer = self.llm.get_input_embeddings()
        return embedding_layer(input_ids)

    def build_multimodal_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        visual_embeds = self.get_visual_embeds(pixel_values)
        text_embeds = self.get_text_embeds(input_ids)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attention_mask = torch.ones(
            visual_embeds.size(0),
            visual_embeds.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        merged_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1,
        )

        model_inputs: Dict[str, torch.Tensor] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": merged_attention_mask,
        }

        if labels is not None:
            visual_ignore = torch.full(
                (labels.size(0), visual_embeds.size(1)),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            model_inputs["labels"] = torch.cat([visual_ignore, labels], dim=1)

        return model_inputs

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        model_inputs = self.build_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        decoder_outputs = self.llm_body.forward(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            **kwargs,
        )

        hidden_states = decoder_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if "labels" in model_inputs:
            loss = self._compute_loss(logits=logits, labels=model_inputs["labels"])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(decoder_outputs, "past_key_values", None),
            hidden_states=getattr(decoder_outputs, "hidden_states", None),
            attentions=getattr(decoder_outputs, "attentions", None),
        )

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]
