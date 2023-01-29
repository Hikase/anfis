from typing import Dict, List

import torch
from pydantic import root_validator
from pydantic.dataclasses import dataclass
from torch import nn

from anfis.membership_functions import MembershipFunction, UniformlyBuilder

__all__ = [
    "TriangularParameter",
    "TriangularMembershipFunction",
    "TriangularUniformlyBuilder",
    "TrapezoidalParameter",
    "TrapezoidalMembershipFunction",
    "TrapezoidalUniformlyBuilder",
]


@dataclass(kw_only=True, frozen=True)
class TriangularParameter:
    """Triangular membership function parameter.

    Attributes:
        left_feet (float): Left feet value of the Triangular membership function.
        peak (float): Peak value of the Triangular membership function.
        right_feet (float): Right feet value of the Triangular membership function.
    """

    left_feet: float
    peak: float
    right_feet: float

    @root_validator(pre=True)
    def check_parameters_not_overlap(cls, parameters: Dict[str, float]) -> Dict[str, float]:
        """Check that the parameters do not overlap.

        Args:
            parameters (Dict[str, float]): Membership function parameters.

        Returns:
            Dict[str, float]: Membership function parameters.

        Raises:
            ValueError: If the parameters do overlap.
        """

        if not parameters["left_feet"] <= parameters["peak"] <= parameters["right_feet"]:
            raise ValueError("Triangular membership function parameters must have left_feet <= peak <= right_feet!")
        return parameters

    @classmethod
    def isosceles(cls, width: float, center: float) -> "TriangularParameter":
        """Returns an isosceles parameter.

        Args:
            width (float): Width of the Triangular membership function.
            center (float): Center of the Triangular membership function.

        Returns:
            TriangularParameter: Triangular membership function parameter.
        """

        return cls(left_feet=center - width, peak=center, right_feet=center + width)


class TriangularMembershipFunction(MembershipFunction):
    """Triangular membership function.

    Attributes:
        left_feet (nn.Parameter): Left feet values of the Triangular membership function.
        peak (nn.Parameter): Peak values of the Triangular membership function.
        right_feet (nn.Parameter): Right feet values of the Triangular membership function.
    """

    def __init__(self, parameters: List[TriangularParameter]) -> None:
        """Triangular membership function constructor.

        Args:
            parameters (List[TriangularParameter]): Triangular membership function parameters.
        """

        super().__init__(parameters)

        left_feet: List[float] = []
        peak: List[float] = []
        right_feet: List[float] = []

        for parameter in parameters:
            left_feet.append(parameter.left_feet)
            peak.append(parameter.peak)
            right_feet.append(parameter.right_feet)

        self.left_feet = nn.Parameter(torch.tensor(left_feet, dtype=torch.float))
        self.peak = nn.Parameter(torch.tensor(peak, dtype=torch.float))
        self.right_feet = nn.Parameter(torch.tensor(right_feet, dtype=torch.float))

    def get_peak_values(self) -> torch.Tensor:
        return self.peak.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(
            torch.min(
                (x - self.left_feet) / (self.peak - self.left_feet),
                (self.right_feet - x) / (self.right_feet - self.peak),
            ),
            torch.zeros_like(x),
        )


class TriangularUniformlyBuilder(UniformlyBuilder):
    """Uniformly builder of the Triangular membership function."""

    def _build(self) -> "TriangularMembershipFunction":
        step = (self._max_value - self._min_value) / (self._n_terms - 1)

        parameters: List[TriangularParameter] = []
        for peak in torch.arange(self._min_value, self._max_value, step):
            parameters.append(TriangularParameter.isosceles(width=step, center=peak))
        parameters.append(TriangularParameter.isosceles(width=step, center=self._max_value))

        return TriangularMembershipFunction(parameters)


@dataclass(kw_only=True, frozen=True)
class TrapezoidalParameter:
    """Trapezoidal membership function parameter.

    Attributes:
        left_feet (float): Left feet value of the Trapezoidal membership function.
        left_peak (float): Left peak value of the Trapezoidal membership function.
        right_peak (float): Right peak value of the Trapezoidal membership function.
        right_feet (float): Right feet value of the Trapezoidal membership function.
    """

    left_feet: float
    left_peak: float
    right_peak: float
    right_feet: float

    @root_validator(pre=True)
    def check_parameters_not_overlap(cls, parameters: Dict[str, float]) -> Dict[str, float]:
        """Check that the parameters do not overlap.

        Args:
            parameters (Dict[str, float]): Membership function parameters.

        Returns:
            Dict[str, float]: Membership function parameters.

        Raises:
            ValueError: If the parameters do overlap.
        """

        if (
            not parameters["left_feet"]
            <= parameters["left_peak"]
            <= parameters["right_peak"]
            <= parameters["right_feet"]
        ):
            raise ValueError(
                "Trapezoidal membership function parameters must have left_feet <= left_peak "
                "<= right_peak <= right_feet!"
            )
        return parameters

    @classmethod
    def isosceles(cls, width: float, center: float, peak_width_percentage: float = 0.2) -> "TrapezoidalParameter":
        """Returns an isosceles parameter.

        Args:
            width (float): Width of the Trapezoidal membership function.
            center (float): Center of the Trapezoidal membership function.
            peak_width_percentage (float, optional): Peak width percentage of the Trapezoidal membership function.
                                                     The default is 0.2.

        Returns:
            TriangularParameter: Triangular membership function parameter.

        Raises:
            ValueError: If the peak width percentage is not between 0 and 1.
        """

        if not 0 <= peak_width_percentage <= 1:
            raise ValueError("The peak width percentage must be between 0 and 1!")

        return cls(
            left_feet=center - width,
            left_peak=center - width * peak_width_percentage,
            right_peak=center + width * peak_width_percentage,
            right_feet=center + width,
        )

    @classmethod
    def triangular(cls, parameter: TriangularParameter) -> "TrapezoidalParameter":
        """Returns the parameter of the Trapezoidal membership function of a triangular shape
        (cast a TriangularParameter to a TrapezoidalParameter).

        Returns:
            TrapezoidalParameter: Trapezoidal membership function parameter.
        """

        return cls(
            left_feet=parameter.left_feet,
            left_peak=parameter.peak,
            right_peak=parameter.peak,
            right_feet=parameter.right_feet,
        )

    @classmethod
    def rectangular(cls, left_peak: float, right_peak: float) -> "TrapezoidalParameter":
        """Returns the parameter of the Trapezoidal membership function of a rectangular shape.

        Args:
            left_peak (float): Left peak value of the Trapezoidal membership function.
            right_peak (float): Right peak value of the Trapezoidal membership function.

        Returns:
            TrapezoidalParameter: Trapezoidal membership function parameter.
        """

        return cls(left_feet=left_peak, left_peak=left_peak, right_peak=right_peak, right_feet=right_peak)


class TrapezoidalMembershipFunction(MembershipFunction):
    """Trapezoidal membership function.

    Attributes:
        left_feet (nn.Parameter): Left feet values of the Trapezoidal membership function.
        left_peak (nn.Parameter): Left peak values of the Trapezoidal membership function.
        right_peak (nn.Parameter): Right peak values of the Trapezoidal membership function.
        right_feet (nn.Parameter): Right feet values of the Trapezoidal membership function.
    """

    def __init__(self, parameters: List[TrapezoidalParameter]) -> None:
        """Trapezoidal membership function constructor.

        Args:
            parameters (List[TrapezoidalParameter]): Trapezoidal membership function parameters.
        """

        super().__init__(parameters)

        left_feet: List[float] = []
        left_peak: List[float] = []
        right_peak: List[float] = []
        right_feet: List[float] = []

        for parameter in parameters:
            left_feet.append(parameter.left_feet)
            left_peak.append(parameter.left_peak)
            right_peak.append(parameter.right_peak)
            right_feet.append(parameter.right_feet)

        self.left_feet = nn.Parameter(torch.tensor(left_feet))
        self.left_peak = nn.Parameter(torch.tensor(left_peak))
        self.right_peak = nn.Parameter(torch.tensor(right_peak))
        self.right_feet = nn.Parameter(torch.tensor(right_feet))

    def get_peak_values(self) -> torch.Tensor:
        return (self.left_peak.data + self.right_peak.data) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(
            torch.min(
                (x - self.left_feet) / (self.left_peak - self.left_feet),
                torch.min(torch.ones_like(x), (self.right_feet - x) / (self.right_feet - self.right_peak)),
            ),
            torch.zeros_like(x),
        )


class TrapezoidalUniformlyBuilder(UniformlyBuilder):
    """Uniformly builder of the Trapezoidal membership function."""

    _peak_width_percentage: float = 0.2

    def _build(self) -> "TrapezoidalMembershipFunction":
        step = (self._max_value - self._min_value) / (self._n_terms - 1)

        parameters: List[TrapezoidalParameter] = []
        for peak in torch.arange(self._min_value, self._max_value, step):
            parameters.append(
                TrapezoidalParameter.isosceles(
                    width=step, center=peak, peak_width_percentage=self._peak_width_percentage
                )
            )
        parameters.append(
            TrapezoidalParameter.isosceles(
                width=step, center=self._max_value, peak_width_percentage=self._peak_width_percentage
            )
        )

        return TrapezoidalMembershipFunction(parameters)

    def peak_width_percentage(self, value: float) -> "TrapezoidalUniformlyBuilder":
        """Sets the peak width percentage.

        Args:
            value (float): The peak width percentage.

        Returns:
            TrapezoidalUniformlyBuilder: Self.

        Raises:
            ValueError: If the peak width percentage is not between 0 and 1.
        """

        if not 0 <= value <= 1:
            raise ValueError("The peak width percentage must be between 0 and 1!")
        self._peak_width_percentage = value
        return self

    def reset(self) -> None:
        super().reset()
        self._peak_width_percentage = 0.2
