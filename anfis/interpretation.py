import itertools
from typing import List, Optional, cast

import torch
from pydantic.dataclasses import dataclass
from torch import nn

from anfis import Anfis
from anfis.membership_functions import MembershipFunction

__all__ = ["LinguisticVariable", "OutputVariable", "create_dataset_to_extract_rules", "get_list_of_rules"]


@dataclass(kw_only=True, frozen=True)
class LinguisticVariable:
    """Represents a linguistic variable.

    Attributes:
        name (str): The name of the linguistic variable.
        term_set (List[str]): The term set of the linguistic variable.
    """

    name: str
    term_set: List[str]

    @property
    def n_terms(self) -> int:
        """Returns the number of terms in the linguistic variable.

        Returns:
            int: The number of terms in the linguistic variable.
        """
        return len(self.term_set)


@dataclass(kw_only=True, frozen=True)
class OutputVariable:
    """Represents an output variable.

    Attributes:
        name (str): The name of the output variable.
        class_names (List[str]): The class names of the output variable.
    """

    name: str
    class_names: List[str]

    @property
    def n_classes(self) -> int:
        """Returns the number of classes in the output variable.

        Returns:
            int: The number of classes in the output variable.
        """
        return len(self.class_names)


def create_dataset_to_extract_rules(*, model: Anfis) -> torch.Tensor:
    """Creates a dataset to extract rules from.

    Args:
        model (Anfis): The model to extract rules from.

    Returns:
        torch.Tensor: The dataset to extract the rules (tensor shape will be [n_rules, in_variables]).
    """

    return torch.tensor(
        list(
            itertools.product(
                *[
                    cast(MembershipFunction, membership_function).get_peak_values()
                    for membership_function in model.fuzzification.membership_functions.values()
                ]
            )
        )
    )


def get_list_of_rules(
    *,
    model: Anfis,
    in_variables: List[LinguisticVariable],
    out_variable: OutputVariable,
    data: Optional[torch.Tensor] = None,
) -> List[str]:
    """
    Extracts a list of rules from a model.

    Args:
        model (Anfis): The model to extract rules from.
        in_variables (List[LinguisticVariable]): The input linguistic variables.
        out_variable (OutputVariable): The output variable.
        data (Optional[torch.Tensor], optional): The dataset to extract the rules from. By default, the
                                                 create_dataset_to_extract_rules function will be used to
                                                 create the dataset.

    Returns:
        List[str]: The list of extracted rules.

    Raises:
        ValueError: If the size of the list of input variables
                    does not match the number of input variables in the model and
                    if the size of the class list of the output variable
                    does not match the number of output classes in the model.
    """

    if len(in_variables) != model.in_variables:
        raise ValueError(f"Expected {model.in_variables} input variables, got {len(in_variables)}!")
    if out_variable.n_classes != model.out_classes:
        raise ValueError(f"Expected {model.out_classes} output variables, got {out_variable.n_classes}!")

    if data is None:
        data = create_dataset_to_extract_rules(model=model)

    softmax = nn.Softmax(dim=1)
    result = softmax(model(data)).argmax(dim=1)
    fuzzification = softmax(model.fuzzification(data)).argmax(dim=1)

    rules: List[str] = []
    for in_terms, out_class in zip(fuzzification, result):
        condition = " AND ".join(
            [f"{variable.name} is '{variable.term_set[term]}'" for variable, term in zip(in_variables, in_terms)]
        )
        rules.append(f"{condition} THEN {out_variable.name} is {out_variable.class_names[out_class]}")

    return rules
