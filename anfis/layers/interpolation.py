import torch
from torch import nn

__all__ = ["InterpolationLayer"]


class InterpolationLayer(nn.Module):
    """Interpolation layer.

    Attributes:
        weight (torch.Tensor): Weight matrix.
    """

    def __init__(self, *, in_variables: int, n_rules: int, out_classes: int) -> None:
        """Interpolation layer constructor.

        Args:
            in_variables (int): Number of input linguistic variables.
            n_rules (int): Number of rules.
            out_classes (int): Number of output classes.
        """

        super().__init__()

        self.weight = nn.Parameter(torch.zeros([n_rules, out_classes, in_variables + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the interpolation layer.

        Args:
            x (torch.Tensor): Input tensor (expected tensor shape would be
                              [batch_size, in_variables]).

        Returns:
            torch.Tensor: Output tensor (tensor shape will be [batch_size, out_classes, n_rules]).

        """

        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        x = torch.matmul(self.weight, x.t())
        return x.transpose(0, 2)

    def extra_repr(self) -> str:
        """Return a string representation of the layer.

        Returns:
            str: String representation of the layer.
        """

        return (
            f"in_features={self.weight.shape[2] - 1}, "
            f"out_classes={self.weight.shape[1]}, "
            f"n_rules={self.weight.shape[0]}"
        )
