from typing import List

import torch
from pydantic.dataclasses import dataclass
from torch import nn

from anfis.membership_functions import MembershipFunction, UniformlyBuilder

__all__ = ["BellParameter", "BellMembershipFunction", "BellUniformlyBuilder"]


@dataclass(kw_only=True, frozen=True)
class BellParameter:
    peak: float
    center_width: float
    slope: float


class BellMembershipFunction(MembershipFunction):
    def __init__(self, parameters: List[BellParameter]):
        super().__init__(parameters)

        peak: List[float] = []
        center_width: List[float] = []
        slope: List[float] = []

        for parameter in parameters:
            peak.append(parameter.peak)
            center_width.append(parameter.center_width)
            slope.append(parameter.slope)

        self.peak = nn.Parameter(torch.tensor(peak))
        self.center_width = nn.Parameter(torch.tensor(center_width))
        self.slope = nn.Parameter(torch.tensor(slope))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.pow(torch.abs((x - self.peak) / self.center_width), 2 * self.slope))


class BellUniformlyBuilder(UniformlyBuilder):
    _center_width_scaling_percentage: float = 0.8
    _slope = 2.0

    def _build(self) -> "BellMembershipFunction":
        step = (self._max_value - self._min_value) / (self._n_classes - 1)

        parameters: List[BellParameter] = []
        for peak in torch.arange(self._min_value, self._max_value, step):
            parameters.append(
                BellParameter(peak=peak, slope=self._slope, center_width=step * self._center_width_scaling_percentage)
            )
        parameters.append(
            BellParameter(
                peak=self._max_value, slope=self._slope, center_width=step * self._center_width_scaling_percentage
            )
        )

        return BellMembershipFunction(parameters)

    def center_width_scaling_percentage(self, value: float) -> "BellUniformlyBuilder":
        if not 0 <= value <= 1:
            raise ValueError("The center width scaling percentage must be between 0 and 1!")
        self._center_width_scaling_percentage = value
        return self

    def slope(self, value: float) -> "BellUniformlyBuilder":
        self._slope = value
        return self

    def reset(self) -> None:
        super().reset()
        self._center_width_scaling_percentage = 0.8
        self._slope = 2.0
