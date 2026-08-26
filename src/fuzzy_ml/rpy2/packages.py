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
# since it depends on pmml). install_rkeel() below always tries the live
# CRAN index first for each - this URL is only the fallback for whichever
# one isn't there. As of this writing, pmml has ALREADY been restored to
# CRAN (2.6.1, published 2026-07-07 - confirmed empirically: the live
# install.packages("pmml") attempt below succeeds on its own, no fallback
# needed), while RKEEL itself is still archived and does still need this
# URL. Both tarballs remain permanently downloadable from CRAN's Archive
# regardless - confirmed by actually installing and running
# RKEEL.loadKeelDataset("iris") end to end from these exact URLs.
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
    Install RKEEL and its own dependency, pmml.

    Both were removed from CRAN's live index (see _ARCHIVED_PACKAGE_URLS's
    comment) as of this writing, but this always tries the live CRAN index
    first for each - if either is ever restored to CRAN, that plain
    install.packages(name) call succeeds on its own and the archived tarball
    is never touched. Only falls back to the exact CRAN Archive URL when the
    live attempt leaves the package not installed (confirmed empirically: a
    package not found on CRAN produces an R warning, not an exception, so
    checking rpackages.isinstalled() afterward is a safe way to detect this).

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

    for name, archive_url in _ARCHIVED_PACKAGE_URLS.items():
        if rpackages.isinstalled(name):
            continue
        utils.install_packages(name)  # try the live CRAN index first
        if not rpackages.isinstalled(name):
            # not (yet) restored to CRAN - fall back to the archived tarball
            utils.install_packages(archive_url, repos=ro.NULL, type="source")
