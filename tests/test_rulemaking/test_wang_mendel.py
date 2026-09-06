"""
Tests for WangMendelMethod's rule-consequence resolution.

No test file existed for this module before this one - this path (labeled/classification
rule induction, not the unlabeled-regression-only usage exercised elsewhere) was never
covered.
"""

import unittest

import torch
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import Product
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning as CLIP
from fuzzy_ml.rulemaking.wang_mendel import WangMendelMethod

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _binary_classification_dataset():
    """20 rows, 1 input feature, a clean linear separation at x=0 - deliberately simple
    so each rule's expected consequence is unambiguous, isolating whether
    WangMendelMethod resolves it against the right observation's own label.
    Created directly on AVAILABLE_DEVICE - WangMendelMethod's find_maximum_fuzzy_terms
    does not itself move observations onto a fuzzy set's own device (unlike CLIP's own
    algorithm(), which does this per-observation), so a CPU-created tensor evaluated
    against CUDA-resident linguistic variables raises a device-mismatch error rather
    than silently working, on a CUDA-capable machine."""
    torch.manual_seed(0)
    x_negative = torch.rand(10, 1, device=AVAILABLE_DEVICE) * 2 - 3  # roughly [-3, -1]
    x_positive = torch.rand(10, 1, device=AVAILABLE_DEVICE) * 2 + 1  # roughly [1, 3]
    data = torch.cat([x_negative, x_positive], dim=0)
    labels = torch.cat(
        [
            torch.zeros(10, 1, device=AVAILABLE_DEVICE),
            torch.ones(10, 1, device=AVAILABLE_DEVICE),
        ],
        dim=0,
    )
    return data, labels


class TestWangMendelMethodConsequenceResolution(unittest.TestCase):
    """Regression test for a real bug: consequence_resolution was created once outside
    the per-observation loop and accumulated across every training row's label (and
    iterated over ALL of exemplars.labels on every pass, not just the current
    observation's own), so every rule converged to whichever target term matched the
    most rows overall (in practice, the majority class) regardless of that rule's own
    row. Confirmed empirically on German Credit before this fix: 100% of 700 induced
    rules shared the identical majority-class consequence."""

    def setUp(self) -> None:
        self.data, self.labels = _binary_classification_dataset()

        input_linguistic_variables = CLIP()(
            train_dataset=LabeledDataset(data=self.data, out_features=1),
            epsilon=0.6,
            adjustment=0.2,
            device=AVAILABLE_DEVICE,
        )
        # Separate, fresh CLIP() instance for the target - CLIP.make_linguistic_
        # variables() (inherited from MetaPartitioner) reuses self.terms/self.minimums/
        # self.maximums across its own internal inputs-then-targets calls with no reset
        # in between, so passing `labels=` into ONE CLIP() call's train_dataset does not
        # produce usable target linguistic variables (confirmed empirically: they come
        # out as a corrupted copy of the input variables' own scale). A second, fresh
        # instance sidesteps that entirely.
        target_linguistic_variables = CLIP()(
            train_dataset=LabeledDataset(data=self.labels, out_features=1),
            epsilon=0.6,
            adjustment=0.2,
            device=AVAILABLE_DEVICE,
        )
        self.linguistic_variables = LinguisticVariables(
            inputs=input_linguistic_variables.inputs,
            targets=target_linguistic_variables.inputs,
        )
        # sanity check the fixture itself produced exactly 2 target terms (one per
        # class) - if CLIP's epsilon ever needs adjusting, this fails loudly here
        # rather than the real assertions below failing for an unrelated reason.
        self.assertEqual(len(self.linguistic_variables.targets), 1)
        self.assertEqual(self.linguistic_variables.targets[0].get_centers().numel(), 2)

    def test_each_rules_consequence_matches_its_own_rows_true_label(self) -> None:
        rules = WangMendelMethod(t_norm=Product)(
            exemplars=LabeledDataset(data=self.data, labels=self.labels),
            linguistic_variables=self.linguistic_variables,
            device=AVAILABLE_DEVICE,
        )
        self.assertEqual(len(rules), len(self.data))

        # target term 0 is centered near label 0, term 1 near label 1 (CLIP creates
        # terms in the order it first encounters sufficiently-different values, and
        # label 0 rows come first in the fixture) - read off the actual center order
        # rather than assuming it, so this doesn't silently pass for the wrong reason.
        centers = self.linguistic_variables.targets[0].get_centers().flatten().tolist()
        term_for_label_1 = int(centers[0] < centers[1])
        term_for_label_0 = 1 - term_for_label_1

        for rule, true_label in zip(rules, self.labels.flatten().tolist()):
            expected_term = term_for_label_1 if true_label == 1.0 else term_for_label_0
            (consequence_pair,) = rule.consequence.indices[0]
            _, actual_term = consequence_pair
            self.assertEqual(
                actual_term,
                expected_term,
                f"rule's consequence term {actual_term} does not match its own row's "
                f"true label {true_label} (expected term {expected_term})",
            )

    def test_consequence_is_not_a_single_constant_across_all_rules(self) -> None:
        """The specific failure mode this bug caused: every rule ends up with the
        IDENTICAL consequence, regardless of the dataset actually containing two
        classes. A fixed dataset with 10 rows of each class must produce more than one
        distinct consequence."""
        rules = WangMendelMethod(t_norm=Product)(
            exemplars=LabeledDataset(data=self.data, labels=self.labels),
            linguistic_variables=self.linguistic_variables,
            device=AVAILABLE_DEVICE,
        )
        distinct_consequences = {
            tuple(sorted(rule.consequence.indices[0])) for rule in rules
        }
        self.assertGreater(
            len(distinct_consequences),
            1,
            "all rules share one consequence - the bug this test guards against "
            "collapses every rule onto the majority class regardless of its own row",
        )


if __name__ == "__main__":
    unittest.main()
