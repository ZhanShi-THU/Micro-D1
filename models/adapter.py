import torch
from torch import nn


class VisualAdapter(nn.Module):
    """Project aligned visual features into the LLM embedding space.
    For the DINO token, register tokens should be delete.
    """

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
