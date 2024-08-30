"""
Functions related to the fuzzy logic rule creation process outlined in the Wang-Mendel Method.
"""

from typing import Type, List
from operator import itemgetter

import torch
import numpy as np
from regime import Node
from fuzzy.relations.t_norm import TNorm
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.rule import Rule

from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.rulemaking.common import find_maximum_fuzzy_terms


class WangMendelMethod(Node):
    """
    The Wang-Mendel Method for fuzzy logic rule generation.
    """

    def __init__(self, t_norm: Type[TNorm], resource_name: str = "rules"):
        super().__init__(resource_name=resource_name)
        self.t_norm = t_norm

    def __call__(
        self,
        exemplars: LabeledDataset,
        linguistic_variables: LinguisticVariables,
        device: torch.device,
    ) -> List[Rule]:
        """
        The Wang-Mendel Method for fuzzy logic rule generation.

        Args:
            exemplars: The exemplary dataset.
            linguistic_variables: The linguistic variables involved and their terms.
            device: The torch device to use.

        Returns:
            A list of Rule objects, kept in the order for which they are created.
        """
        rules, consequence_resolution = [], {}
        for observation in exemplars.data:
            # this block of code is for antecedents
            premise, _ = find_maximum_fuzzy_terms(
                observation, linguistic_variables.inputs
            )
            consequence = set()
            if (
                exemplars.labels is not None
                and linguistic_variables.targets is not None
            ):
                for output_observation in exemplars.labels:
                    possible_consequence, membership_degrees = find_maximum_fuzzy_terms(
                        output_observation,
                        linguistic_variables.targets,
                        offset=len(observation),
                    )
                    for variable_term_pair, membership_degree in zip(
                        possible_consequence, membership_degrees
                    ):
                        consequence_resolution[variable_term_pair] = (
                            consequence_resolution.get(variable_term_pair, 0)
                            + membership_degree
                        )

                self.wang_mendel_method_helper(
                    consequence,
                    consequence_resolution,
                    linguistic_variables.targets,
                    offset=len(observation),
                )
            else:
                for output_idx in range(exemplars.out_features):
                    consequence.add((len(rules), output_idx))
            rules.append(
                Rule(
                    premise=self.t_norm(*premise, device=device),
                    consequence=self.t_norm(*consequence, device=device),
                )
            )

        return rules

    @staticmethod
    def wang_mendel_method_helper(
        consequence, consequence_resolution, consequences, offset: int = 0
    ) -> None:
        """
        A helper method for the Wang-Mendel Method. Given the consequence, consequence resolution,
        and consequences, assist in generating the consequence of the current fuzzy logic rule.

        Args:
            consequence:
            consequence_resolution:
            consequences:
            offset: The offset to apply to the output variable indices. Default is zero.
            For example, if the input variables are [0, 1, 2] and the output variables are
            [3, 4, 5], then the offset should be 3.

        Returns:
            None
        """
        for output_variable_idx, _ in enumerate(consequences):
            possible_output_variable_terms = tuple(
                variable_term_pair
                for variable_term_pair in consequence_resolution
                if variable_term_pair[0] == (output_variable_idx + offset)
            )
            # get the scalar cardinalities of that output var's terms
            if len(possible_output_variable_terms) > 0:
                scalar_cardinalities = itemgetter(*possible_output_variable_terms)(
                    consequence_resolution
                )
                consequence.add(
                    possible_output_variable_terms[int(np.argmax(scalar_cardinalities))]
                )
