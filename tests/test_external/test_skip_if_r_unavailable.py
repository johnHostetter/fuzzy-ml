"""
White-box tests for tests.test_external.skip_if_r_unavailable - the shared
decorator that turns an RRuntimeError raised during R package setup into a
skipped test instead of a collection-aborting error.
"""

import unittest

from rpy2.rinterface_lib.embedded import RRuntimeError

from tests.test_external import skip_if_r_unavailable


class TestSkipIfRUnavailable(unittest.TestCase):
    """
    Test skip_if_r_unavailable's two branches directly, independent of any
    real R package install.
    """

    def test_passes_through_the_return_value_when_no_error_is_raised(self) -> None:
        """
        The wrapped function's own return value (and any side effects, e.g.
        attributes it sets on self) must be unaffected when it succeeds.
        """

        class _Case(unittest.TestCase):
            def runTest(self):  # pragma: no cover - never actually run
                pass

            @skip_if_r_unavailable
            def setUp(self):
                self.value = 42

        case = _Case()
        case.setUp()
        self.assertEqual(case.value, 42)

    def test_converts_an_rruntimeerror_into_a_skiptest(self) -> None:
        """
        An RRuntimeError raised by the wrapped function must be converted
        into unittest.SkipTest, not left to propagate as a plain error - this
        is what stops a module's entire test collection from aborting when
        R package setup fails (e.g. no writable R library configured).
        """

        class _Case(unittest.TestCase):
            def runTest(self):  # pragma: no cover - never actually run
                pass

            @skip_if_r_unavailable
            def setUp(self):
                raise RRuntimeError("no writable R library")

        case = _Case()
        with self.assertRaises(unittest.SkipTest):
            case.setUp()

    def test_other_exceptions_are_not_swallowed(self) -> None:
        """
        Only RRuntimeError is treated as an environmental gap - a genuine bug
        in the wrapped function (e.g. a plain ValueError) must still surface
        as a real error, not be silently converted into a skip.
        """

        class _Case(unittest.TestCase):
            def runTest(self):  # pragma: no cover - never actually run
                pass

            @skip_if_r_unavailable
            def setUp(self):
                raise ValueError("not an R problem")

        case = _Case()
        with self.assertRaises(ValueError):
            case.setUp()


if __name__ == "__main__":
    unittest.main()
