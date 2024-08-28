"""
Test the Fuzzy Temporal Association Rule Mining algorithm
and its necessary helper functions.
"""

import unittest
from typing import Tuple, Dict, Any, List

import torch
import numpy as np
import pandas as pd
from fuzzy.sets.continuous.impl import Gaussian
from fuzzy.sets.continuous.membership import Membership
from fuzzy.sets.continuous.group import GroupedFuzzySets
from fuzzy.logic import LinguisticVariables, KnowledgeBase

from fuzzy_ml.utils import set_rng
from fuzzy_ml.association.temporal import (
    TemporalInformationTable as TI,
    FuzzyTemporalAssocationRuleMining as FTARM,
    AssociationRule,
)
from examples.ftarm import make_example


set_rng(5)
AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def big_data_example(seed: int) -> Tuple[pd.DataFrame, KnowledgeBase]:
    """
    Generate an example with a large amount of data for benchmarking computational performance.

    Args:
        seed: Random number generator seed.

    Returns:
        The data and the knowledge base.
    """
    set_rng(seed)
    dataframe = pd.DataFrame(np.random.rand(4000, 4))
    dataframe["date"] = 0
    if seed == 0:
        variables = {
            0: Gaussian(
                centers=np.array([1.5410, -0.2934, -2.1788, 0.5684]),
                widths=np.array([0.4556, 0.6323, 0.3489, 0.4017]),
                device=AVAILABLE_DEVICE,
            ),
            1: Gaussian(
                centers=np.array([0.4033, 0.8380, -0.7193, -0.4033]),
                widths=np.array([0.6816, 0.9152, 0.3971, 0.8742]),
                device=AVAILABLE_DEVICE,
            ),
            2: Gaussian(
                centers=np.array([-0.8567, 1.1006, -1.0712, 0.1227]),
                widths=np.array([0.1759, 0.2698, 0.1507, 0.0317]),
                device=AVAILABLE_DEVICE,
            ),
            3: Gaussian(
                centers=np.array([-0.8920, -1.5091, 0.3704, 1.4565]),
                widths=np.array([0.1387, 0.2422, 0.8155, 0.7932]),
                device=AVAILABLE_DEVICE,
            ),
        }
    elif seed == 5:
        variables = {
            0: Gaussian(
                centers=np.array([-0.4868, -0.6038, -0.5581, 0.6675]),
                widths=np.array([0.0333, 0.9942, 0.6064, 0.5646]),
                device=AVAILABLE_DEVICE,
            ),
            1: Gaussian(
                centers=np.array([-1.4017, -0.7626, 0.6312, -0.8991]),
                widths=np.array([0.6698, 0.2615, 0.0407, 0.7850]),
                device=AVAILABLE_DEVICE,
            ),
            2: Gaussian(
                centers=np.array([0.2225, -0.6662, 0.6846, 0.5740]),
                widths=np.array([0.0442, 0.4884, 0.7965, 0.7432]),
                device=AVAILABLE_DEVICE,
            ),
            3: Gaussian(
                centers=np.array([0.0571, -1.1894, -0.5659, -0.8327]),
                widths=np.array([0.8796, 0.9009, 0.9186, 0.5979]),
                device=AVAILABLE_DEVICE,
            ),
        }
    else:
        raise ValueError(f"The seed, {seed}, is not recognized.")

    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=list(variables.values()), targets=[]
        ),
        rules=[],
    )
    return dataframe, knowledge_base


class TestFTARM(unittest.TestCase):
    """
    Test the Fuzzy Temporal Association Rule Mining algorithm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_support: float = 0.3
        self.min_confidence: float = 0.8

    def test_hyperparameters(self) -> None:
        """
        Test that the hyperparameters are correctly recognized/recovered.

        Returns:
            None
        """
        hyperparameters = FTARM._hyperparameters
        # uninitialized hyperparameters should be None
        self.assertEqual({"min_support": None, "min_confidence": None}, hyperparameters)

        # test that module path to the class hyperparameters is correctly recognized
        actual_hyperparameters_dict = FTARM.make_hyperparameters_dict()
        expected_hyperparameters_dict: Dict[str, Any] = {
            "fuzzy_ml": {
                "association": {
                    "temporal": {
                        FTARM.__name__: {
                            "min_support": None,
                            "min_confidence": None,
                        }
                    }
                }
            }
        }
        self.assertEqual(expected_hyperparameters_dict, actual_hyperparameters_dict)

    def test_fuzzy_representation(self) -> None:
        """
        Test the fuzzy representation calculated by the KnowledgeBase.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        input_granulation: List[GroupedFuzzySets] = knowledge_base.select_by_tags(
            tags={"premise", "group"}
        )["item"]
        assert len(input_granulation) == 1  # there should only be 1 matching item
        input_granulation: GroupedFuzzySets = input_granulation[0]
        cols = sorted(set(dataframe.columns) - {"date"})
        actual_memberships: Membership = input_granulation(
            torch.tensor(
                dataframe[cols].values, dtype=torch.float32, device=AVAILABLE_DEVICE
            )
        )
        expected_membership = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(
            actual_memberships.degrees.to_dense().reshape(5, 10).float(),
            expected_membership,
        )

    def test_temporal_information_table(self) -> None:
        """
        Test the information being stored by the Temporal Information (TI) table is correct.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        attribute_names = [col for col in dataframe.columns if col != "date"]
        ti_table = TI(dataframe, variables=attribute_names)
        # temporal item A occurs in the first transaction,
        # B occurs in the second, C occurs in the first, and so on
        assert np.allclose(
            ti_table.first_transaction_indices, np.array([0, 1, 0, 3, 4])
        )
        # temporal items D and E come in the second time period,
        # all others occur in the first time period
        assert np.allclose(
            ti_table.starting_periods.values, np.array([[0, 0, 0, 1, 1]])
        )

        # now checking that FTARM creates the same TI Table as above
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        # temporal item A occurs in the first transaction,
        # B occurs in the second, C occurs in the first, and so on
        assert np.allclose(
            ftarm.ti_table.first_transaction_indices, np.array([0, 1, 0, 3, 4])
        )
        # temporal items D and E come in the second time period,
        # all others occur in the first time period
        assert np.allclose(
            ftarm.ti_table.starting_periods.values, np.array([[0, 0, 0, 1, 1]])
        )

    def test_step_2(self) -> None:
        """
        Test that the described 'step 2' of the Fuzzy Temporal Association Rule Mining algorithm
        is working as intended.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        actual_scalar_cardinality = ftarm.scalar_cardinality()
        expected_scalar_cardinality = torch.tensor(
            [2.5, 0.0, 1.75, 0.25, 4 / 3, 2 / 3, 0.0, 2.0, 1.0, 0.0],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(
            actual_scalar_cardinality.to_dense().flatten().float(),
            expected_scalar_cardinality,
        )

    def test_step_3(self) -> None:
        """
        Test that the described 'step 3' of the Fuzzy Temporal Association Rule Mining algorithm
        is working as intended.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports()
        expected_output = torch.tensor(
            [
                [0.5, 0.0],  # low A, high A
                [0.35, 0.05],  # low B, high B
                [0.26666665, 0.13333333],  # low C, high C
                [0.0, 1.0],  # low D, high D
                [0.5, 0.0],
            ],
            device=AVAILABLE_DEVICE,
        )  # low E, high E
        assert torch.allclose(actual_fuzzy_temporal_supports.float(), expected_output)

    def test_step_4(self) -> None:
        """
        Test that the described 'step 4' of the Fuzzy Temporal Association Rule Mining algorithm
        is working as intended.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        # start of step 4
        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports()
        # L1 --> low A, low B, high D, and low E
        assert torch.equal(
            (actual_fuzzy_temporal_supports >= ftarm.min_support),
            torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [False, False],
                    [False, True],
                    [True, False],
                ],
                device=AVAILABLE_DEVICE,
            ),
        )

        c2_indices = ftarm.make_candidates()

        # step 7:
        # (low A, low B), (low A, high D), (low A, low E),
        # (low B, high D), (low B, low E), (high D, low E)
        assert c2_indices == [
            ((0, 0), (1, 0)),
            ((0, 0), (3, 1)),
            ((0, 0), (4, 0)),
            ((1, 0), (3, 1)),
            ((1, 0), (4, 0)),
            ((3, 1), (4, 0)),
        ]

    def test_candidate_fuzzy_representation(self) -> None:
        """
        Test that the generated candidates found with the Fuzzy Temporal Association Rule
        Mining algorithm are consistent with the expected results - and that the intermediate
        results are also consistent.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        # step 8.1:

        actual_antecedents_memberships: Membership = ftarm.granulation(
            torch.tensor(
                dataframe[ftarm.variables].values,
                dtype=torch.float32,
                device=AVAILABLE_DEVICE,
            )
        )

        expected_memberships = torch.tensor(
            [
                [
                    [1.0000, 0.0000],
                    [0.0000, 0.0000],
                    [2 / 3, 1 / 3],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                ],
                [
                    [0.5000, 0.0000],
                    [0.5000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                ],
                [
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                    [2 / 3, 1 / 3],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000],
                ],
                [
                    [0.5000, 0.0000],
                    [0.5000, 0.0000],
                    [0.0000, 0.0000],
                    [0.0000, 1.0000],
                    [0.0000, 0.0000],
                ],
                [
                    [0.5000, 0.0000],
                    [0.7500, 0.2500],
                    [0.0000, 0.0000],
                    [0.0000, 1.0000],
                    [1.0000, 0.0000],
                ],
            ],
            device=AVAILABLE_DEVICE,
        )

        assert torch.allclose(
            actual_antecedents_memberships.degrees.to_dense().float(),
            expected_memberships,
            equal_nan=True,
        )

    def test_candidate_fuzzy_representation_ftarm(self) -> None:
        """
        Test that the fuzzy representation of the generated candidates is correctly calculated.

        Returns:
            None
        """
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(
            dataframe,
            linguistic_variables,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        c2_indices = ftarm.make_candidates()

        # step 8.1:

        # now checking that FTARM calculates the same as above
        expected_output = torch.tensor(
            [
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.50, 0.50, 0.00, 0.50, 0.00, 0.00],
                [0.50, 0.50, 0.50, 0.75, 0.75, 1.00],
            ],
            device=AVAILABLE_DEVICE,
        )

        assert torch.allclose(ftarm.fuzzy_representation(c2_indices), expected_output)

    def test_candidate_scalar_cardinality(self) -> None:
        """
        Test that the scalar cardinality of the generated candidates is correctly calculated.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        c2_indices = ftarm.make_candidates()

        # step 8.2:

        actual_scalar_cardinality = ftarm.scalar_cardinality(c2_indices)
        expected_scalar_cardinality = torch.tensor(
            [1.5, 1.0, 0.5, 1.25, 0.75, 1.0], device=AVAILABLE_DEVICE
        )
        assert torch.allclose(actual_scalar_cardinality, expected_scalar_cardinality)

    def test_candidate_fuzzy_temporal_supports(self) -> None:
        """
        Test that the fuzzy temporal supports of the generated candidates is correctly calculated.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        c2_indices = ftarm.make_candidates()
        # the following candidate order is assumed for the following assertions
        assert c2_indices == [
            ((0, 0), (1, 0)),
            ((0, 0), (3, 1)),
            ((0, 0), (4, 0)),
            ((1, 0), (3, 1)),
            ((1, 0), (4, 0)),
            ((3, 1), (4, 0)),
        ]

        # step 8.3

        # we need to get each temporal item's corresponding starting period
        item_indices_in_each_candidate = [
            tuple(pair[0] for pair in candidate) for candidate in c2_indices
        ]
        # (0, 1) means the first and second items in ti_table.terms.keys(), and so on
        assert item_indices_in_each_candidate == [
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 3),
            (1, 4),
            (3, 4),
        ]

        starting_periods_per_item_in_each_candidate = [
            [
                ftarm.ti_table.starting_periods.values[0, var_idx]
                for var_idx in candidate_indices
            ]
            for candidate_indices in item_indices_in_each_candidate
        ]
        assert starting_periods_per_item_in_each_candidate == [
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
        ]

        # get the maximum starting period within each candidate to calculate fuzzy temporal support
        max_starting_periods = np.array(
            starting_periods_per_item_in_each_candidate
        ).max(axis=1)

        assert np.allclose(max_starting_periods, np.array([0, 1, 1, 1, 1, 1]))

        num_of_transactions_per_candidate = [
            ftarm.ti_table.size_of_transactions_per_time_granule.values[idx:].sum()
            for idx in max_starting_periods
        ]
        num_of_transactions_per_candidate = np.array(num_of_transactions_per_candidate)

        assert np.allclose(
            num_of_transactions_per_candidate, np.array([5, 2, 2, 2, 2, 2])
        )

        fuzzy_temporal_supports = ftarm.scalar_cardinality(c2_indices) / torch.tensor(
            num_of_transactions_per_candidate, device=AVAILABLE_DEVICE
        )
        expected_fuzzy_temporal_supports = torch.tensor(
            [0.3, 0.5, 0.25, 0.625, 0.375, 0.5], device=AVAILABLE_DEVICE
        )

        assert torch.allclose(fuzzy_temporal_supports, expected_fuzzy_temporal_supports)

        # now checking that FTARM calculates the same as above

        assert torch.allclose(
            ftarm.fuzzy_temporal_supports(c2_indices), expected_fuzzy_temporal_supports
        )

        l2_indices = torch.where(
            fuzzy_temporal_supports
            >= torch.tensor(ftarm.min_support, device=AVAILABLE_DEVICE)
        )[
            0
        ]  # L2 items' indices

        assert torch.equal(
            l2_indices, torch.tensor([0, 1, 3, 4, 5], device=AVAILABLE_DEVICE)
        )

    def test_make_candidate_3_itemsets(self) -> None:
        """
        Test that the algorithm produces the correct 3-itemsets,
        as outlined in the original paper.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        c2_indices = ftarm.make_candidates()
        c3_indices = ftarm.make_candidates(c2_indices)
        expected_candidate_indices = [
            {(1, 0), (3, 1), (0, 0)},
            {(1, 0), (4, 0), (3, 1)},
        ]
        assert c3_indices == expected_candidate_indices

        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports(c3_indices)
        expected_fuzzy_temporal_supports = torch.tensor(
            [0.5000, 0.3750], device=AVAILABLE_DEVICE
        )
        assert torch.allclose(
            actual_fuzzy_temporal_supports, expected_fuzzy_temporal_supports
        )

    def test_make_candidate_4_itemsets(self) -> None:
        """
        Test that the algorithm does not incorrectly find any 4-itemset candidates,
        as outlined in the original paper.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        c2_indices = ftarm.make_candidates()
        c3_indices = ftarm.make_candidates(c2_indices)
        c4_indices = ftarm.make_candidates(c3_indices)
        expected_candidate_indices = None
        assert c4_indices == expected_candidate_indices

    def test_execute(self) -> None:
        """
        Test that the algorithm - when fully executed, finds the expected candidates,
        as outlined in the original paper.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 0), (1, 0)),
            ((0, 0), (3, 1)),
            ((0, 0), (4, 0)),
            ((1, 0), (3, 1)),
            ((1, 0), (4, 0)),
            ((3, 1), (4, 0)),
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [
            {(1, 0), (3, 1), (0, 0)},
            {(1, 0), (4, 0), (3, 1)},
        ]

    def test_find_association_rules(self) -> None:
        """
        Test that the algorithm - when fully executed, finds the expected association rules,
        as outlined in the original paper.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        # required to build the lattice structure which is later used to find rules
        _ = ftarm.find_candidates()

        actual_rules = ftarm.find_association_rules(min_confidence=self.min_confidence)
        expected_rules = [
            AssociationRule(
                antecedents=frozenset({(0, 0)}),
                consequents=frozenset({(3, 1)}),
                confidence=1.0,
            ),
            AssociationRule(
                antecedents=frozenset({(1, 0)}),
                consequents=frozenset({(3, 1)}),
                confidence=1.0,
            ),
            AssociationRule(
                antecedents=frozenset({(4, 0)}),
                consequents=frozenset({(3, 1)}),
                confidence=1.0,
            ),
            AssociationRule(
                antecedents=frozenset({(1, 0), (0, 0)}),
                consequents=frozenset({(3, 1)}),
                confidence=1.0,
            ),
            AssociationRule(
                antecedents=frozenset({(3, 1), (0, 0)}),
                consequents=frozenset({(1, 0)}),
                confidence=1.0,
            ),
            AssociationRule(
                antecedents=frozenset({(1, 0), (4, 0)}),
                consequents=frozenset({(3, 1)}),
                confidence=1.0,
            ),
        ]

        for actual_rule, expected_rule in zip(actual_rules, expected_rules):
            assert actual_rule.antecedents == expected_rule.antecedents
            assert actual_rule.consequents == expected_rule.consequents
            assert actual_rule.confidence == expected_rule.confidence

    def test_execute_big_data_0(self) -> None:
        """
        This unit test was introduced to identify that frequent itemsets
        were not being created that contained more than one linguistic variable assignment.
        Also, some operations originally would result in an empty list
        being returned that the above unit tests do not capture.

        Returns:
            None
        """
        dataframe, knowledge_base = big_data_example(seed=0)
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 3), (1, 0)),
            ((0, 3), (1, 1)),
            ((0, 3), (1, 3)),
            ((0, 3), (3, 2)),
            ((1, 0), (3, 2)),
            ((1, 1), (3, 2)),
            ((1, 3), (3, 2)),
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [
            {(1, 1), (0, 3), (3, 2)},
            {(1, 0), (3, 2), (0, 3)},
            {(3, 2), (0, 3), (1, 3)},
        ]

    def test_execute_big_data_5(self) -> None:
        """
        This unit test was introduced to identify that frequent itemsets were not being
        created that contained more than one linguistic variable assignment. Also, some
        operations originally would result in an empty list
        being returned that the above unit tests do not capture.

        Returns:
            None
        """
        dataframe, knowledge_base = big_data_example(seed=5)
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 1), (2, 2)),
            ((0, 1), (2, 3)),
            ((0, 1), (3, 0)),
            ((0, 3), (2, 2)),
            ((0, 3), (2, 3)),
            ((0, 3), (3, 0)),
            ((2, 2), (3, 0)),
            ((2, 3), (3, 0)),
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [
            {(0, 1), (2, 3), (3, 0)},
            {(2, 3), (0, 3), (3, 0)},
            {(0, 3), (3, 0), (2, 2)},
            {(0, 1), (3, 0), (2, 2)},
        ]

    def test_closed_itemsets(self) -> None:
        """
        Test that the correct itemsets are being identified as closed itemsets.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        # required to build the lattice structure which is later used to find rules
        _ = ftarm.find_candidates()

        actual_closed_itemsets = ftarm.find_closed_itemsets()
        assert set(actual_closed_itemsets) == {
            frozenset({(3, 1), (4, 0)}),
            frozenset({(1, 0), (3, 1), (0, 0)}),
            frozenset({0, 1}),
            frozenset({(1, 0), (4, 0), (3, 1)}),
            frozenset({(1, 0), (3, 1)}),
            frozenset({1, 3}),
        }

    def test_maximal_itemsets(self) -> None:
        """
        Test that the correct itemsets are being identified as maximal itemsets.

        Returns:
            None
        """
        dataframe, knowledge_base = make_example()
        ftarm = FTARM(
            dataframe,
            knowledge_base,
            min_support=self.min_support,
            device=AVAILABLE_DEVICE,
        )

        # required to build the lattice structure which is later used to find rules
        _ = ftarm.find_candidates()

        actual_maximal_itemsets = ftarm.find_maximal_itemsets()
        assert actual_maximal_itemsets == {
            frozenset({(1, 0), (3, 1), (0, 0)}),
            frozenset({(1, 0), (4, 0), (3, 1)}),
        }
