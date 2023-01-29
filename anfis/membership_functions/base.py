from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch import nn

__all__ = ["MembershipFunction", "UniformlyBuilder"]


class MembershipFunction(nn.Module, ABC):
    """Base class for membership functions."""

    def __init__(self, parameters: List[Any]):
        """Membership function constructor.

        Attributes:
            parameters (List[Any]): List of parameters.

        Raises:
            ValueError: If the number of parameters is less than 3.
        """

        super().__init__()

        self._n_terms = len(parameters)

        if self.n_terms < 3:
            raise ValueError("Membership function must have at least 3 terms!")

    @abstractmethod
    def get_peak_values(self) -> torch.Tensor:
        """Returns the peak values of the membership function.

        Returns:
            torch.Tensor: Peak values.
        """

        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the membership function.

        Args:
            x (torch.Tensor): Input tensor (expected tensor shape would be
                              [batch_size, 1]).

        Returns:
            torch.Tensor: Output tensor (tensor shape will be [batch_size, n_terms]).
        """

        pass

    @property
    def n_terms(self) -> int:
        """Returns the number of terms in the membership function.

        Returns:
            int: Number of terms.
        """
        return self._n_terms

    def extra_repr(self) -> str:
        """Returns the extra representation of the membership function.

        Returns:
            str: String representation of the membership function.
        """

        return f"n_terms={self.n_terms}"


class UniformlyBuilder(ABC):
    """Base class for the builder of uniformly distributed membership functions."""

    def __init__(self) -> None:
        """Builder constructor."""

        super().__init__()

        self._min_value: float = 0
        self._max_value: float = 0
        self._n_terms: int = 3

    @abstractmethod
    def _build(self) -> MembershipFunction:
        """Builds the concrete membership function."""

        pass

    def _check_readiness_for_build(self) -> None:
        """Checks if the builder is ready for building.

        Raises:
            ValueError: If the minimum value is greater than the maximum!
        """

        if self._min_value >= self._max_value:
            raise ValueError("The minimum value is greater than the maximum!")

    def min_value(self, value: float) -> "UniformlyBuilder":
        """Sets the minimum value.

        Args:
            value (float): Minimum value.

        Returns:
            UniformlyBuilder: Self.
        """

        self._min_value = value
        return self

    def max_value(self, value: float) -> "UniformlyBuilder":
        """Sets the maximum value.

        Args:
            value (float): Maximum value.

        Returns:
            UniformlyBuilder: Self.
        """

        self._max_value = value
        return self

    def n_terms(self, value: int) -> "UniformlyBuilder":
        """Sets the number of terms.

        Args:
            value (int): Number of terms.

        Returns:
            UniformlyBuilder: Self.

        Raises:
            ValueError: If the number of terms is less than 3!
        """

        if value < 3:
            raise ValueError("The number of terms must be at least 3!")
        self._n_terms = value
        return self

    def reset(self) -> None:
        """Resets the builder."""

        self._min_value = 0
        self._max_value = 0
        self._n_terms = 3

    def build(self, *, reset: bool = True) -> MembershipFunction:
        """Builds the membership function.

        Args:
            reset (bool, optional): Whether to reset the builder. Defaults to True.

        Returns:
            MembershipFunction: Membership function.
        """

        self._check_readiness_for_build()
        result = self._build()
        if reset:
            self.reset()
        return result
