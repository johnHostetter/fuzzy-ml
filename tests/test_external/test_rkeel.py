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
import rpy2.robjects.packages as rpackages

# from rpy2 import robjects
from rpy2.robjects.packages import importr


class TestRKEEL(unittest.TestCase):
    """
    Test the RKEEL package written in R.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)

        # download the dependencies

        # dependencies = [
        #     "downloader",
        #     "R6",
        #     "XML",
        #     "doParallel",
        #     "foreach",
        #     "gdata",
        #     "RKEELjars",
        #     "RKEELdata",
        #     "pmml",
        #     "arules",
        #     "rJava",
        # ]
        # for dependency in dependencies:
        #     utils.install_packages(dependency)
        #
        # # install RKEEL from CRAN archive
        #
        # robjects.r(
        #     "jars <- 'https://cran.r-project.org/src/contrib/Archive/RKEELjars/RKEELjars_1.0.20.tar.gz'"
        # )  # RKEELjars was archived due to a policy violation
        # robjects.r(
        #     "rkeel <- 'https://cran.r-project.org/src/contrib/Archive/RKEEL/RKEEL_1.3.3.tar.gz'"
        # )  # RKEEL was archived due to RKEELjars being archived
        # robjects.r(
        #     "install.packages(jars, repos=NULL, type='source')"
        # )  # install RKEELjars from URL
        # robjects.r(
        #     "install.packages(rkeel, repos=NULL, type='source')"
        # )  # install RKEEL from URL
        # keel_package = importr("RKEEL")
        utils.install_packages("RKEEL")
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
