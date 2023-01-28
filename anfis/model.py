from typing import List

import torch
from torch import nn

from anfis.layers import FuzzificationLayer, FuzzyInferenceLayer, InterpolationLayer
from anfis.membership_functions import MembershipFunction, UniformlyBuilder

__all__ = ["Anfis", "create_anfis"]


class Anfis(nn.Module):
    def __init__(self, *, out_features: int, membership_functions: List[MembershipFunction]) -> None:
        super().__init__()
        self._in_features = len(membership_functions)
        self._out_features = out_features

        self.fuzzification = FuzzificationLayer(membership_functions)
        self.fuzzy_inference = FuzzyInferenceLayer(
            n_classes=self.fuzzification.n_classes, n_membership_functions=self.fuzzification.n_membership_functions
        )
        self.interpolation = InterpolationLayer(
            in_features=self.in_features, n_rules=self.fuzzy_inference.n_rules, out_features=out_features
        )

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inference = self.fuzzy_inference(self.fuzzification(x))
        interpolation = self.interpolation(x)
        return torch.bmm(interpolation, inference.unsqueeze(2)).squeeze(2)


def create_anfis(
    *,
    out_features: int,
    min_values: List[float],
    max_values: List[float],
    membership_function_builder: UniformlyBuilder,
    n_classes: int = 3,
) -> "Anfis":
    if len(min_values) != len(max_values):
        raise ValueError("min_values and max_values must have the same length!")

    membership_functions: List[MembershipFunction] = []
    for min_value, max_value in zip(min_values, max_values):
        membership_functions.append(
            membership_function_builder.min_value(min_value).max_value(max_value).n_classes(n_classes).build()
        )

    return Anfis(out_features=out_features, membership_functions=membership_functions)
