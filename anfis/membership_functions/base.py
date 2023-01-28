from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch import nn

__all__ = ["MembershipFunction", "UniformlyBuilder"]


class MembershipFunction(nn.Module, ABC):
    def __init__(self, parameters: List[Any]):
        super().__init__()
        self._n_classes = len(parameters)

        if self.n_classes < 3:
            raise ValueError("The number of parameters (number of classes) must be greater than or equal to 3!")

    @abstractmethod
    def get_peak_values(self) -> torch.Tensor:
        pass

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def extra_repr(self) -> str:
        return f"n_classes={self.n_classes}"


class UniformlyBuilder(ABC):
    def __init__(self) -> None:
        self._min_value: float = 0
        self._max_value: float = 0
        self._n_classes: int = 3

    @abstractmethod
    def _build(self) -> "MembershipFunction":
        pass

    def _check_readiness_for_build(self) -> None:
        if self._min_value >= self._max_value:
            raise ValueError("The minimum value is greater than the maximum!")

    def min_value(self, value: float) -> "UniformlyBuilder":
        self._min_value = value
        return self

    def max_value(self, value: float) -> "UniformlyBuilder":
        self._max_value = value
        return self

    def n_classes(self, value: int) -> "UniformlyBuilder":
        if value < 3:
            raise ValueError("The number of parameters (number of classes) must be greater than or equal to 3!")
        self._n_classes = value
        return self

    def reset(self) -> None:
        self._min_value = 0
        self._max_value = 0
        self._n_classes = 3

    def build(self, *, reset: bool = True) -> "MembershipFunction":
        self._check_readiness_for_build()
        result = self._build()
        if reset:
            self.reset()
        return result
