"""
Implements various functions that are useful or common between various
fuzzy logic rule generation procedures, such as the namedtuple "Rule".
"""

from typing import List

import torch
import numpy as np
from fuzzy.sets import FuzzySet, Membership


def find_maximum_fuzzy_terms(
    data_observation: torch.Tensor,
    antecedents: List[FuzzySet],
    offset: int = 0,
):
    """
    Given a data observation and antecedents, find the fuzzy set in each dimension that has
    maximum membership.

    Args:
        data_observation: A data observation.
        antecedents: The antecedents.
        offset: An offset to start counting from with respect to variable index, default is 0.

    Returns:
        A compound proposition (frozenset) of 2-tuple elements, where the first item in the element
        refers to the indexed variable, and the second item in the element refers to the indexed
        term or value for that variable. Also, a second object is returned as well, a list of
        membership degrees (represented as float).
    """
    compound_proposition, all_membership_degrees = [], []
    for variable_index, variable_value in enumerate(data_observation):
        # SM_jps = vertices[variable_index]['data'](observation[variable_index]).detach().numpy()
        linguistic_variable = antecedents[variable_index]
        if (
            linguistic_variable.get_centers().dim() > 0
        ):  # i.e., there must be linguistic terms available
            membership: Membership = linguistic_variable(variable_value)
            if membership.degrees.is_sparse:
                membership_degrees = (
                    membership.degrees.to_dense().cpu().detach().numpy()
                )
            else:
                membership_degrees = membership.degrees.cpu().detach().numpy()
            indexed_fuzzy_set_with_highest_membership = np.argmax(membership_degrees)
            compound_proposition.append(
                (variable_index + offset, indexed_fuzzy_set_with_highest_membership)
            )
            all_membership_degrees.append(np.max(membership_degrees))
    return frozenset(compound_proposition), all_membership_degrees
