"""
Test the reduction of fuzzy logic rules with rough sets (via rpy2).
"""

import unittest

import torch
from fuzzy.logic import Rule
from fuzzy.relations.continuous.t_norm import Product
from fuzzy.relations.continuous.n_ary import NAryRelation

from fuzzy_ml.pruning.rpy2.rough_theory import (
    find_unique_premise_variables,
    reduce_fuzzy_logic_rules_with_rough_sets,
)


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRuleReduction(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rules = [
            Rule(
                premise=Product(
                    (2, 3), (1, 1), (0, 3), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 1), (1, 1), (2, 1), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((1, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (3, 2), (1, 3), (2, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((2, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (3, 2), (2, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((3, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (2, 1), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((4, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (0, 2), (2, 4), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((5, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (2, 1), (3, 0), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((6, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (2, 3), (1, 0), (0, 3), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((7, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 1), (2, 3), (3, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((1, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 3), (3, 3), (2, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((9, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (2, 4), (1, 2), (3, 3), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((10, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 2), (1, 2), (2, 1), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((11, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 1), (1, 1), (2, 3), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((12, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (2, 0), (3, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((13, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 2), (2, 2), (3, 0), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((14, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 1), (2, 4), (0, 4), (3, 2), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((15, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (2, 0), (3, 0), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((16, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 1), (0, 2), (1, 2), (2, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((17, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 2), (2, 0), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((18, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 2), (1, 3), (2, 1), (3, 2), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((19, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 1), (2, 4), (0, 4), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((20, 0), device=AVAILABLE_DEVICE),
            ),
        ]

    def test_find_unique_premise_variables(self) -> None:
        unique_premise_variables = find_unique_premise_variables(self.rules)
        self.assertEqual(unique_premise_variables, [0, 1, 2, 3])

    def test_reduce_fuzzy_logic_rules_with_rough_sets(self) -> None:
        """
        Test the reduction of fuzzy logic rules with rough sets.

        Returns:
            None
        """
        actual_rules, output_values = reduce_fuzzy_logic_rules_with_rough_sets(
            self.rules,
            t_norm=Product,
            device=AVAILABLE_DEVICE,
            output_values=torch.tensor(
                [
                    [0.5982, 0.3638],
                    [-0.4473, -1.3327],
                    [-1.7578, 0.6419],
                    [-1.3145, 1.7113],
                    [1.4630, 0.9406],
                    [0.3238, 0.7816],
                    [0.7368, 0.4987],
                    [-1.4179, 0.0476],
                    [-0.2674, -0.3777],
                    [0.3399, -1.2760],
                    [1.1519, 2.5623],
                    [-1.5076, -0.9596],
                    [1.1151, -0.3059],
                    [0.4684, 0.7088],
                    [-0.6936, -0.5474],
                    [1.2732, -0.8924],
                    [-0.5352, -0.7027],
                    [-0.3630, 0.3177],
                    [-0.6501, 0.1239],
                    [-1.0044, 0.7831],
                    [-1.7858, -0.2264],
                ]
            ),
        )
        expected_rules = [
            Rule(
                premise=Product(
                    (2, 3), (1, 1), (0, 3), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 1), (2, 3), (3, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((1, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 1), (1, 1), (2, 3), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((12, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (2, 1), (3, 0), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((6, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (2, 0), (3, 0), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((16, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 1), (0, 4), (2, 4), (3, 2), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((15, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (3, 3), (1, 3), (2, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((9, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (3, 1), (1, 1), (2, 1), (0, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((1, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (2, 1), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((4, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product((1, 2), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((9, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (3, 2), (2, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((3, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (1, 0), (2, 0), (3, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((13, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 0), (0, 2), (2, 4), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((5, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (1, 1), (0, 4), (2, 4), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((20, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 1), (3, 2), (1, 3), (2, 1), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((2, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (0, 2), (1, 3), (2, 1), (3, 2), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((19, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product(
                    (2, 3), (1, 0), (0, 3), (3, 0), device=AVAILABLE_DEVICE
                ),
                consequence=NAryRelation((7, 0), device=AVAILABLE_DEVICE),
            ),
        ]
        expected_output_values = torch.tensor(
            [
                [0.5982, 0.3638],
                [-0.2674, -0.3777],
                [1.1151, -0.3059],
                [0.7368, 0.4987],
                [-0.5352, -0.7027],
                [1.2732, -0.8924],
                [0.3399, -1.2760],
                [-0.4473, -1.3327],
                [1.4630, 0.9406],
                [-0.4125, 0.2994],
                [-1.3145, 1.7113],
                [0.4684, 0.7088],
                [0.3238, 0.7816],
                [-1.7858, -0.2264],
                [-1.7578, 0.6419],
                [-1.0044, 0.7831],
                [-1.4179, 0.0476],
            ]
        )
        self.assertEqual(len(expected_rules), len(actual_rules))
        self.assertEqual(len(output_values), len(actual_rules))
        idx = 0
        for expected_rule, actual_rule in zip(expected_rules, actual_rules):
            print(idx)
            idx += 1
            self.assertEqual(
                set(expected_rule.premise.indices[0]),
                set(actual_rule.premise.indices[0]),
            )
            self.assertEqual(
                expected_rule.consequence.indices[0], actual_rule.consequence.indices[0]
            )
        self.assertTrue(
            torch.allclose(expected_output_values, output_values, atol=1e-4, rtol=1e-4)
        )

    def test_reduce_fuzzy_logic_rules_with_rough_sets_when_nothing_can_be_reduced(
        self,
    ) -> None:
        """
        Test that the fuzzy logic rules remained unchanged when nothing can be reduced using
        rough sets.

        Returns:
            None
        """
        expected_output_values = torch.tensor(
            [
                [0.0, 1.0],
            ]
            * len(self.rules),
        )  # the expected output values all have column 1 > column 0; thus, no rules can be reduced
        rules, output_values = reduce_fuzzy_logic_rules_with_rough_sets(
            self.rules,
            t_norm=Product,
            device=AVAILABLE_DEVICE,
            output_values=expected_output_values,
        )
        self.assertEqual(len(rules), len(self.rules))
        self.assertEqual(len(rules), len(output_values))
        for rule, expected_rule in zip(rules, self.rules):
            self.assertEqual(rule.premise, expected_rule.premise)
            self.assertEqual(rule.consequence, expected_rule.consequence)
        self.assertTrue(
            torch.allclose(
                output_values.cpu(), expected_output_values.cpu(), atol=1e-4, rtol=1e-4
            )
        )
