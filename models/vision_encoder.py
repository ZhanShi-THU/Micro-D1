from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torchvision import models as tv_models


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}")

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, num_tokens, dim)
        return self.proj(attended)


class DinoTextHeadBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim=dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUMLP(dim=dim, hidden_dim=mlp_hidden_dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOTextAlignmentHead(nn.Module):
    """
    dinotxt visual head extracted from the provided checkpoint.

    Expected tensor shape:
      input:  [B, N, D_dino]
      output: [B, N, D_dino]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 16,
        num_blocks: int = 2,
        mlp_hidden_dim: int = 2752,
    ) -> None:
        super().__init__()
        if input_dim != output_dim:
            raise ValueError(
                f"dinotxt head expects equal input/output dims, got {input_dim} and {output_dim}"
            )
        self.blocks = nn.ModuleList(
            [
                DinoTextHeadBlock(
                    dim=input_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln_final = nn.LayerNorm(output_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        aligned_tokens = visual_tokens
        for block in self.blocks:
            aligned_tokens = block(aligned_tokens)
        return self.ln_final(aligned_tokens)

    def load_pretrained(self, checkpoint_path: str) -> None:
        state_dict = self._normalize_checkpoint(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Failed to load dinotxt head cleanly. "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )

    def _normalize_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("dinotxt checkpoint must be a mapping-like object.")

        normalized_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix = "visual_model.head."
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith(prefix):
                normalized_state[key[len(prefix) :]] = value

        if not normalized_state:
            raise RuntimeError(
                "No compatible dinotxt visual head weights were found in the checkpoint."
            )
        return normalized_state


class VisionEncoder(nn.Module):
    """
    Vision-only encoder for the modular VLM.

    Flow:
      1. image [B, C, H, W]
      2. DINOv3 -> patch tokens [B, N, D_dino]
      3. dinotxt projection -> aligned tokens [B, N, D_dino]
    """

    def __init__(
        self,
        backbone_name: str,
        embed_dim_dino: int,
        alignment_dim: int,
        alignment_head_weights: Optional[str] = None,
        vision_source: str = "torch_hub",
        vision_repo: str = "facebookresearch/dinov3",
        vision_model_name: str = "dinov3_vitl16",
        vision_pretrained: bool = True,
        vision_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim_dino = embed_dim_dino
        self.alignment_dim = alignment_dim
        self.backbone = self._build_backbone(
            source=vision_source,
            repo=vision_repo,
            model_name=vision_model_name,
            pretrained=vision_pretrained,
        )
        if vision_checkpoint_path:
            checkpoint_path = Path(vision_checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Vision checkpoint not found: {vision_checkpoint_path}")
            self._load_backbone_checkpoint(checkpoint_path)
        self.alignment_head = DINOTextAlignmentHead(
            input_dim=embed_dim_dino,
            output_dim=alignment_dim,
        )

        if alignment_head_weights:
            checkpoint_path = Path(alignment_head_weights)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"dinotxt checkpoint not found: {alignment_head_weights}"
                )
            self.alignment_head.load_pretrained(str(checkpoint_path))

    def _build_backbone(
        self,
        source: str,
        repo: str,
        model_name: str,
        pretrained: bool,
    ) -> nn.Module:
        if source == "torchvision":
            if not hasattr(tv_models, "get_model"):
                raise ValueError(
                    "torchvision.get_model is unavailable in the installed torchvision."
                )
            return tv_models.get_model(model_name, weights="DEFAULT")
        if source == "torch_hub":
            return torch.hub.load(repo, model_name, pretrained=pretrained)
        if source == "local":
            local_path = Path(model_name)
            if not local_path.exists():
                raise FileNotFoundError(f"Local vision backbone not found: {model_name}")
            return torch.jit.load(str(local_path), map_location="cpu")
        raise ValueError(f"Unsupported vision source: {source}")

    def _load_backbone_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError("Vision checkpoint must contain a mapping-like state dict.")

        normalized_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_key = str(key)
            for prefix in ("module.", "backbone.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            normalized_state[normalized_key] = value

        missing_keys, unexpected_keys = self.backbone.load_state_dict(normalized_state, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading vision checkpoint {checkpoint_path}: {unexpected_keys}"
            )
        if not normalized_state:
            raise RuntimeError(f"No tensor weights found in vision checkpoint: {checkpoint_path}")

    def _extract_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "forward_features"):
            raise RuntimeError("DINOv3 backbone must expose a forward_features method.")

        features = self.backbone.forward_features(pixel_values)

        if isinstance(features, Mapping):
            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
            elif "patch_tokens" in features:
                patch_tokens = features["patch_tokens"]
            elif "x_prenorm" in features and isinstance(features["x_prenorm"], torch.Tensor):
                patch_tokens = features["x_prenorm"][:, 1:, :]
            else:
                available = ", ".join(sorted(str(key) for key in features.keys()))
                raise RuntimeError(
                    "Unable to find patch tokens in DINOv3 outputs. "
                    f"Available keys: {available}"
                )
        elif isinstance(features, torch.Tensor):
            if features.ndim != 3:
                raise RuntimeError(
                    "Expected DINOv3 forward_features to return [B, N, D] tokens."
                )
            patch_tokens = features
        else:
            raise RuntimeError(
                f"Unsupported DINOv3 output type: {type(features).__name__}"
            )

        if patch_tokens.ndim != 3:
            raise RuntimeError("Patch tokens must have shape [B, N, D_dino].")
        if patch_tokens.size(-1) != self.embed_dim_dino:
            raise RuntimeError(
                "Unexpected DINOv3 hidden size: "
                f"got {patch_tokens.size(-1)}, expected {self.embed_dim_dino}"
            )
        return patch_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        visual_tokens = self._extract_patch_tokens(pixel_values)
        aligned_tokens = self.alignment_head(visual_tokens)
        return aligned_tokens
