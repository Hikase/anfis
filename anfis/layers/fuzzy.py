import itertools
from typing import List

import torch
from torch import nn

from anfis.membership_functions import MembershipFunction

__all__ = ["FuzzificationLayer", "FuzzyInferenceLayer"]


class FuzzificationLayer(nn.Module):
    """Fuzzification layer.

    Attributes:
        membership_functions (nn.ModuleDict): Dictionary of the membership functions.
    """

    def __init__(self, membership_functions: List[MembershipFunction]) -> None:
        """Fuzzification layer constructor.

        Args:
            membership_functions (List[MembershipFunction]): List of the membership functions.

        Raises:
            ValueError: Membership functions must have the same number of terms!
        """

        super().__init__()

        self._n_terms = membership_functions[0].n_terms
        for membership_function in membership_functions[1:]:
            if self.n_terms != membership_function.n_terms:
                raise ValueError("Membership functions must have the same number of terms!")

        self._n_membership_functions = len(membership_functions)
        self.membership_functions = nn.ModuleDict(
            dict(zip([f"membership_function_{i}" for i in range(len(membership_functions))], membership_functions))
        )

    @property
    def n_terms(self) -> int:
        """Return the number of terms.

        Returns:
            int: Number of terms.
        """

        return self._n_terms

    @property
    def n_membership_functions(self) -> int:
        """Return the number of membership functions.

        Returns:
            int: Number of membership functions.
        """

        return self._n_membership_functions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the fuzzification layer.

        Args:
            x (torch.Tensor): Input tensor (expected tensor shape would be
                              [batch_size, n_membership_functions]).

        Returns:
            torch.Tensor: Output tensor (tensor shape will be [batch_size, n_terms, n_membership_functions]).

        Raises:
            ValueError: The number of input features is not equal to the number of membership functions!
        """

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


class FuzzyInferenceLayer(nn.Module):
    """Fuzzy inference layer."""

    def __init__(self, *, n_terms: int, n_membership_functions: int) -> None:
        """Fuzzy inference layer constructor.

        Args:
            n_terms (int): Number of terms.
            n_membership_functions (int): Number of membership functions.
        """

        super().__init__()

        self._n_terms = n_terms
        self._n_membership_functions = n_membership_functions

        self._combination_of_fuzzy_inferences = torch.tensor(
            list(itertools.product(*[range(n) for n in [self.n_terms for _ in range(self.n_membership_functions)]]))
        )

    @property
    def n_terms(self) -> int:
        """Return the number of terms.

        Returns:
            int: Number of terms.
        """

        return self._n_terms

    @property
    def n_membership_functions(self) -> int:
        """Return the number of membership functions.

        Returns:
            int: Number of membership functions.
        """

        return self._n_membership_functions

    @property
    def n_rules(self) -> int:
        """Return the number of rules.

        Returns:
            int: Number of rules.
        """

        return self._combination_of_fuzzy_inferences.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the fuzzy inference layer.

        Args:
            x (torch.Tensor): Input tensor (expected tensor shape would be
                              [batch_size, n_terms, n_membership_functions])

        Returns:
            torch.Tensor: Output tensor (tensor shape will be [batch_size, n_rules])
        """

        x = torch.prod(torch.gather(x, 1, self._combination_of_fuzzy_inferences.expand((x.shape[0], -1, -1))), dim=2)
        return nn.functional.normalize(x, p=1, dim=1)

    def extra_repr(self) -> str:
        """Return a string representation of the layer.

        Returns:
            str: String representation of the layer.
        """

        return f"n_terms={self.n_terms}, n_rules={self.n_rules}"
