from typing import List

import torch
from torch import nn

from anfis.layers import FuzzificationLayer, FuzzyInferenceLayer, InterpolationLayer
from anfis.membership_functions import MembershipFunction, UniformlyBuilder

__all__ = ["Anfis", "create_anfis"]


class Anfis(nn.Module):
    """Anfis model.

    Attributes:
        fuzzification (FuzzificationLayer): Fuzzification layer.
        fuzzy_inference (FuzzyInferenceLayer): Fuzzy inference layer.
        interpolation (InterpolationLayer): Interpolation layer.
    """

    def __init__(self, *, out_classes: int, membership_functions: List[MembershipFunction]) -> None:
        """Anfis constructor.

        Args:
            out_classes (int): The number of classes of the output variable.
            membership_functions (List[MembershipFunction]): List of membership functions (the number of membership
                                                             functions is equal to the input linguistic variables).
        """
        super().__init__()
        self._in_features = len(membership_functions)
        self._out_classes = out_classes

        self.fuzzification = FuzzificationLayer(membership_functions)
        self.fuzzy_inference = FuzzyInferenceLayer(
            n_terms=self.fuzzification.n_terms, n_membership_functions=self.fuzzification.n_membership_functions
        )
        self.interpolation = InterpolationLayer(
            in_variables=self.in_variables, n_rules=self.fuzzy_inference.n_rules, out_classes=out_classes
        )

    @property
    def in_variables(self) -> int:
        """Returns the number of input linguistic variables.

        Returns:
            int: The number of input linguistic variables.
        """
        return self._in_features

    @property
    def out_classes(self) -> int:
        """Returns the number of output classes.

        Returns:
            int: The number of output classes.
        """
        return self._out_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the anfis model.

        Args:
            x (torch.Tensor): Input tensor (expected tensor shape would be
                              [batch_size, in_variables]).

        Returns:
            torch.Tensor: Output tensor (tensor shape will be [batch_size, out_classes]).
        """

        inference = self.fuzzy_inference(self.fuzzification(x))
        interpolation = self.interpolation(x)
        return torch.bmm(interpolation, inference.unsqueeze(2)).squeeze(2)


def create_anfis(
    *,
    out_classes: int,
    min_values: List[float],
    max_values: List[float],
    membership_function_builder: UniformlyBuilder,
    n_terms: int = 3,
) -> Anfis:
    """Creates anfis model.

    Args:
        out_classes (int): The number of output classes.
        min_values (List[float]): The minimum values of the input linguistic variables.
        max_values (List[float]): The maximum values of the input linguistic variables.
        membership_function_builder (UniformlyBuilder): The membership function builder.
        n_terms (int, optional): The number of classes. Defaults to 3.

    Returns:
        Anfis: The anfis model.

    Raises:
        ValueError: If min_values and max_values have different lengths!
    """

    if len(min_values) != len(max_values):
        raise ValueError("min_values and max_values must have the same length!")

    membership_functions: List[MembershipFunction] = []
    for min_value, max_value in zip(min_values, max_values):
        membership_functions.append(
            membership_function_builder.min_value(min_value).max_value(max_value).n_terms(n_terms).build()
        )

    return Anfis(out_classes=out_classes, membership_functions=membership_functions)
