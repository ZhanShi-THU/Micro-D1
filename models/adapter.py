import torch
from torch import nn


class VisualAdapter(nn.Module):
    """Project aligned visual features into the LLM embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.norm(visual_tokens)
        visual_tokens = self.fc1(visual_tokens)
        visual_tokens = self.act(visual_tokens)
        visual_tokens = self.fc2(visual_tokens)
        return self.dropout(visual_tokens)


class LinearVisualAdapter(nn.Module):
    """Single-layer adapter used for linear-projection ablations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.norm(visual_tokens)
        visual_tokens = self.proj(visual_tokens)
        return self.dropout(visual_tokens)


def build_visual_adapter(
    *,
    adapter_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 2048,
    dropout: float = 0.0,
) -> nn.Module:
    normalized = str(adapter_type).strip().lower()
    if normalized in {"mlp", "default"}:
        return VisualAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if normalized == "linear":
        return LinearVisualAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported adapter_type: {adapter_type!r}")
