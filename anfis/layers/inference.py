import itertools
from collections import OrderedDict
from typing import List

import torch
from torch import nn

from anfis.membership_functions import MembershipFunction

__all__ = ["FuzzyLayer", "InferenceLayer"]


class FuzzyLayer(nn.Module):
    def __init__(self, membership_functions: List[MembershipFunction]) -> None:
        super().__init__()

        self._n_classes = membership_functions[0].n_classes
        for membership_function in membership_functions[1:]:
            if self.n_classes != membership_function.n_classes:
                raise ValueError("The number of membership function classes must be the same!")

        self._n_membership_functions = len(membership_functions)
        self.membership_functions = nn.ModuleDict(
            OrderedDict(
                zip([f"membership_function_{i}" for i in range(len(membership_functions))], membership_functions)
            )
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_membership_functions(self) -> int:
        return self._n_membership_functions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_membership_functions:
            raise ValueError("The number of input features is not equal to the number of membership functions!")
        x = torch.stack(
            [
                membership_function(x[:, k : k + 1])
                for k, membership_function in enumerate(self.membership_functions.values())
            ],
            dim=1,
        )
        return x.transpose(1, 2)


class InferenceLayer(nn.Module):
    def __init__(self, *, n_classes: int, n_membership_functions: int):
        super().__init__()

        self._n_classes = n_classes
        self._n_membership_functions = n_membership_functions

        self._combination_of_fuzzy_inferences = torch.tensor(
            list(itertools.product(*[range(n) for n in [self.n_classes for _ in range(self.n_membership_functions)]]))
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_membership_functions(self) -> int:
        return self._n_membership_functions

    @property
    def n_rules(self) -> int:
        return self._combination_of_fuzzy_inferences.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.prod(torch.gather(x, 1, self._combination_of_fuzzy_inferences.expand((x.shape[0], -1, -1))), dim=2)
        return nn.functional.normalize(x, p=1, dim=1)

    def extra_repr(self) -> str:
        return f"n_classes={self.n_classes}, n_rules={self.n_rules}"
