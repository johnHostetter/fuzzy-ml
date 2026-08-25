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
    10. Install RKEEL (and its own dependency, pmml) from CRAN's Archive.
    11. Import the RKEEL package written in R.
"""

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# RKEEL and pmml were both removed from CRAN's live index (pmml failed a
# routine CRAN check on 2026-01-29; RKEEL was auto-archived the same day
# since it depends on pmml), so install.packages("RKEEL")/("pmml") against
# the live repo index no longer finds them. Both tarballs remain permanently
# downloadable from CRAN's Archive - confirmed by actually installing and
# running RKEEL.loadKeelDataset("iris") end to end from these exact URLs.
# RKEELjars (an older RKEEL dependency some earlier notes reference) is not
# needed here - RKEEL 1.3.4 does not use it, and it is not archived anyway.
_RKEEL_VERSION = "1.3.4"
_PMML_VERSION = "2.6.0"
_ARCHIVED_PACKAGE_URLS = {
    "pmml": f"https://cran.r-project.org/src/contrib/Archive/pmml/pmml_{_PMML_VERSION}.tar.gz",
    "RKEEL": f"https://cran.r-project.org/src/contrib/Archive/RKEEL/RKEEL_{_RKEEL_VERSION}.tar.gz",
}

# RKEEL's own dependencies (per its DESCRIPTION file) that are still live on
# CRAN - installed normally, before the two archived packages above, since
# install.packages(url, repos=NULL, type="source") does not resolve
# dependencies from CRAN the way installing from the live repo index does.
_RKEEL_LIVE_DEPENDENCIES = (
    "R6",
    "XML",
    "doParallel",
    "foreach",
    "gdata",
    "RKEELdata",
    "arules",
    "Matrix",
    "rJava",
    "openssl",
    "downloader",
    "stringr",  # pmml's own dependency, not RKEEL's - installed here too since
    # pmml is installed from source alongside RKEEL below.
)


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


def install_rkeel() -> None:
    """
    Install RKEEL and its own archived dependency, pmml, from CRAN's Archive.

    See _ARCHIVED_PACKAGE_URLS's comment for why the live package names
    alone no longer resolve.

    Returns:
        None
    """
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)

    names_to_install = [
        name for name in _RKEEL_LIVE_DEPENDENCIES if not rpackages.isinstalled(name)
    ]
    if names_to_install:
        utils.install_packages(StrVector(names_to_install))

    for name, url in _ARCHIVED_PACKAGE_URLS.items():
        if not rpackages.isinstalled(name):
            utils.install_packages(url, repos=ro.NULL, type="source")
