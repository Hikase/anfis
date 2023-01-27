import torch
from torch import nn

__all__ = ["InterpolationLayer"]


class InterpolationLayer(nn.Module):
    def __init__(self, *, in_features: int, n_rules: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros([n_rules, out_features, in_features + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        x = torch.matmul(self.weight, x.t())
        return x.transpose(0, 2)

    def extra_repr(self) -> str:
        return f"is_hybrid={self.is_hybrid}"
