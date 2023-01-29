from typing import List

import torch
from pydantic.dataclasses import dataclass
from torch import nn

from anfis.membership_functions import MembershipFunction, UniformlyBuilder

__all__ = ["GaussianParameter", "GaussianMembershipFunction", "GaussianUniformlyBuilder"]


@dataclass(kw_only=True, frozen=True)
class GaussianParameter:
    """Gaussian membership function parameter.

    Attributes:
        mean (float): Mean of the Gaussian membership function.
        std (float): Standard deviation of the Gaussian membership function.
    """

    mean: float
    std: float


class GaussianMembershipFunction(MembershipFunction):
    """Gaussian membership function.

    Attributes:
        mean (nn.Parameter): Mean values of the Gaussian membership function.
        std (nn.Parameter): Standard deviation values of the Gaussian membership function.
    """

    def __init__(self, parameters: List[GaussianParameter]):
        """Gaussian membership function constructor.

        Args:
            parameters (List[GaussianParameter]): Parameters of the Gaussian membership function.
        """

        super().__init__(parameters)

        mean: List[float] = []
        std: List[float] = []

        for parameter in parameters:
            mean.append(parameter.mean)
            std.append(parameter.std)

        self.mean = nn.Parameter(torch.tensor(mean))
        self.std = nn.Parameter(torch.tensor(std))

    def get_peak_values(self) -> torch.Tensor:
        return self.std.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.pow(x - self.mean, 2) / (2 * torch.pow(self.std, 2)))


class GaussianUniformlyBuilder(UniformlyBuilder):
    """Uniformly builder of the Gaussian membership function."""

    _width_scaling_percentage: float = 0.5

    def _build(self) -> "GaussianMembershipFunction":
        step = (self._max_value - self._min_value) / (self._n_terms - 1)

        parameters: List[GaussianParameter] = []
        for peak in torch.arange(self._min_value, self._max_value, step):
            parameters.append(GaussianParameter(mean=peak, std=step * self._width_scaling_percentage))
        parameters.append(GaussianParameter(mean=self._max_value, std=step * self._width_scaling_percentage))

        return GaussianMembershipFunction(parameters)

    def width_scaling_percentage(self, value: float) -> "GaussianUniformlyBuilder":
        """Set the width scaling percentage.

        Args:
            value (float): Width scaling percentage.

        Returns:
            GaussianUniformlyBuilder: Self.

        Raises:
            ValueError: If the width scaling percentage is not between 0 and 1!
        """

        if not 0 <= value <= 1:
            raise ValueError("The width scaling percentage must be between 0 and 1!")
        self._width_scaling_percentage = value
        return self

    def reset(self) -> None:
        super().reset()
        self._width_scaling_percentage = 0.5
