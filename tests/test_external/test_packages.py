"""
Test the project's interface to installing and importing R packages.
"""

from unittest import TestCase

import rpy2.robjects.packages as rpackages

from fuzzy_ml.rpy2.packages import install_r_packages


class TestExternal(TestCase):
    """
    Test the project's interface to installing and importing R packages.
    """

    def test_install_r_packages(self) -> None:
        """
        Test that R packages can be installed.
        """
        install_r_packages()

        package_names = ("DChaos", "frbs", "RoughSets")

        for package_name in package_names:
            self.assertTrue(rpackages.isinstalled(package_name))
