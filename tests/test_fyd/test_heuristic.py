"""
White-box tests for fuzzy_ml.fyd.heuristic.find_heuristic_cutoff().

Only find_heuristic_cutoff() is covered here - the rest of this module
(frequent_discernible(), calc_fyd_heuristic(), calc_normalized_scalar_
cardinalities(), calc_scalar_cardinality(), induce_subgraph()) all require a
real KnowledgeBase produced by fuzzy_ml.partitioning.clip - as of this
writing, that pipeline raises `AttributeError: 'Gaussian' object has no
attribute '_widths'` (confirmed reproducible even in complete isolation,
with fresh random data and no fyd code involved at all - not something this
migration introduced or can fix here), so those functions cannot be
exercised with a real KnowledgeBase right now.
"""

import unittest

from fuzzy_ml.fyd.heuristic import find_heuristic_cutoff


class TestFindHeuristicCutoff(unittest.TestCase):
    """
    Covers find_heuristic_cutoff()'s actual, verified behavior - each
    expected value below was confirmed by directly running the function,
    not assumed.
    """

    def test_clear_knee_is_detected(self) -> None:
        """
        A genuinely convex, increasing curve has a real, non-zero knee.
        """
        curve = [0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.4, 0.9, 0.95, 0.99]
        self.assertAlmostEqual(find_heuristic_cutoff(curve), 0.15)

    def test_all_equal_values_have_no_knee(self) -> None:
        """
        A flat curve (all values equal) has no distinguishable knee - falls
        back to the function's own knee_value is None -> 0.0 default.
        """
        self.assertEqual(find_heuristic_cutoff([0.5, 0.5, 0.5]), 0.0)

    def test_single_value_has_no_knee(self) -> None:
        """
        A single observation cannot have a knee - same 0.0 fallback.
        """
        self.assertEqual(find_heuristic_cutoff([0.3]), 0.0)

    def test_non_positive_values_are_filtered_out_before_detection(self) -> None:
        """
        Only strictly-positive values are considered (`val > 0` in the
        function's own filter) - negative and zero entries are dropped
        before the knee search runs, not treated as valid observations.
        """
        # after filtering: [0.1, 0.2, 0.9] - too few points for a knee
        self.assertEqual(find_heuristic_cutoff([-1, 0, 0.1, 0.2, 0.9]), 0.0)

    def test_no_positive_values_returns_zero_instead_of_raising(self) -> None:
        """
        Fixed: if every value is filtered out (all <= 0, or the input is
        empty), this used to let the underlying KneeLocator call raise
        ValueError on the resulting empty array, instead of ever reaching the
        function's own `if knee_value is None: knee_value = 0.0` fallback -
        that fallback only handled "a knee search ran and found nothing", not
        "no knee search could run at all". Found while wiring
        WangMendelMethod's own (separately fixed) consequence resolution into
        a real self-organizing pipeline - the more varied, now-correct rule
        set it produces triggered this exact "zero positive heuristics" case,
        which was never exercised via a real KnowledgeBase before. 0.0 is the
        semantically correct return here, not just a crash-avoidance
        placeholder: frequent_discernible() (this module's own caller)
        already special-cases cutoff_value == 0.0 to mean "delete every
        vertex whose heuristic isn't positive," which is exactly the right
        outcome when there are zero positive candidates in the first place.
        """
        self.assertEqual(find_heuristic_cutoff([0.0, 0.0, 0.0]), 0.0)
        self.assertEqual(find_heuristic_cutoff([]), 0.0)


if __name__ == "__main__":
    unittest.main()
