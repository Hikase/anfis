import itertools
from typing import List, Optional, cast

import torch
from pydantic.dataclasses import dataclass
from torch import nn

from anfis import Anfis
from anfis.membership_functions import MembershipFunction


@dataclass(kw_only=True, frozen=True)
class LinguisticVariable:
    name: str
    term_set: List[str]

    @property
    def n_terms(self) -> int:
        return len(self.term_set)


@dataclass(kw_only=True, frozen=True)
class OutputVariable:
    name: str
    class_names: List[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def create_dataset_to_extract_rules(*, model: Anfis) -> torch.Tensor:
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
    if len(in_variables) != model.in_features:
        raise ValueError(f"Expected {model.in_features} input variables, got {len(in_variables)}!")
    if out_variable.n_classes != model.out_features:
        raise ValueError(f"Expected {model.out_features} output variables, got {out_variable.n_classes}!")

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
