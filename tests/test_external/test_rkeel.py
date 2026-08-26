"""
Test that the RKEEL package written in R can be imported and used from Python.

The following may be necessary on Windows to set the 'R_HOME' for rpy2 correctly:

    from rpy2 import situation
    import os
    os.environ['R_HOME'] = situation.r_home_from_registry()
    situation.get_r_home()
"""

import unittest

import rpy2
from rpy2.robjects.packages import importr

from fuzzy_ml.rpy2.packages import install_rkeel
from tests.test_external import skip_if_r_unavailable


class TestRKEEL(unittest.TestCase):
    """
    Test the RKEEL package written in R.
    """

    @skip_if_r_unavailable
    def setUp(self) -> None:
        install_rkeel()
        self.keel_package = importr("RKEEL")

    def test_rkeel(self) -> None:
        """
        Test the RKEEL package. The following code is adapted from the
        RKEEL package documentation. See:

            https://cran.r-project.org/web/packages/RKEEL/RKEEL.pdf

        However, sufficient testing is not possible with RKEEL since it relies on
        Java, which must first be installed and configured correctly. To do this,
        in each session that is run (such as within a terminal), the following
        command must be run:

            R CMD javareconf -e

        This is not possible to do from within Python, as the command will terminate
        the Python session. Therefore, the user must first run the above command
        from within a terminal, and then run the Python session. This is not
        ideal, and so the RKEEL package is not tested here.

        Returns:
            None
        """
        iris_data: rpy2.robjects.vectors.DataFrame = self.keel_package.loadKeelDataset(
            "iris"
        )
        assert len(iris_data.rownames) == 150
