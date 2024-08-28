"""
This Python script will implement the following:

    1. Import the rpy2 package, which is a Python interface to the R language.
    2. Import the rpy2.robjects.packages module, which will be used to install R packages.
    3. Import the rpy2.robjects.packages module, which will be used to import R packages.
    4. Install the DChaos package written in R.
    5. Import the DChaos package written in R.
    6. Install the frbs package written in R.
    7. Import the frbs package written in R.
    8. Install the RoughSets package written in R.
    9. Import the RoughSets package written in R.
"""

import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector


def install_r_packages() -> None:
    """
    Install the R packages that will be used in this project.

    Returns:
        None
    """
    # import R's "utils" package
    utils = importr("utils")

    # R package names
    package_names = ("DChaos", "frbs", "RoughSets")

    # selectively install what needs to be installed.
    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))
