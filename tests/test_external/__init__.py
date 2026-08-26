"""
Shared test fixtures for tests/test_external, which exercise real R packages
via rpy2 (RKEEL, DChaos, frbs, RoughSets).
"""

import functools
import unittest

from rpy2.rinterface_lib.embedded import RRuntimeError


def skip_if_r_unavailable(func):
    """
    Decorator that turns an R-package-install/import failure into a skipped
    test instead of an error - or, if applied to setUp(), instead of an error
    that aborts collection of the whole module (unittest/pytest both treat an
    exception raised from a TestCase's own __init__ as a collection error,
    not a per-test failure, which previously took down every test in modules
    that did their R setup work there).

    A missing R package is a common, expected environmental gap (e.g. no
    write access to R's system library and no personal library configured -
    exactly what happens in a sandboxed environment) rather than a bug in the
    code under test, so it's surfaced as a skip with the underlying R error
    message attached, not a failure.

    Args:
        func: The setUp() (or test) method to wrap.

    Returns:
        The wrapped method.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RRuntimeError as error:
            raise unittest.SkipTest(
                f"Skipping: R package setup failed, likely because this "
                f"environment cannot install R packages (e.g. no writable "
                f"R library configured): {error}"
            ) from error

    return wrapper
