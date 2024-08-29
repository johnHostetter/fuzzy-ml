"""
This example shows how to use RKEEL from Python. Note that this requires
Java to be installed and configured correctly. To do this, in each session
that is run (such as within a terminal), the following command must be run:

    R CMD javareconf -e

This is not possible to do from within Python, as the command will terminate
the Python session. Therefore, the user must first run the above command
from within a terminal, and then run the Python session. Further, do not expect
this example to work from within an IDE such as PyCharm, as the IDE will not
have access to the Java environment variables that are set by the above command.
"""

import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr

# import rpy2.robjects.packages as rpackages

utils = importr("utils")
utils.install_packages("rJava")

# install RKEEL from CRAN archive

robjects.r(
    "jars <- 'https://cran.r-project.org/src/contrib/Archive/RKEELjars/RKEELjars_1.0.20.tar.gz'"
)  # RKEELjars was archived due to a policy violation
robjects.r(
    "rkeel <- 'https://cran.r-project.org/src/contrib/Archive/RKEEL/RKEEL_1.3.3.tar.gz'"
)  # RKEEL was archived due to RKEELjars being archived
robjects.r(
    "install.packages(jars, repos=NULL, type='source')"
)  # install RKEELjars from URL
robjects.r(
    "install.packages(rkeel, repos=NULL, type='source')"
)  # install RKEEL from URL
keel_package = importr("RKEEL")

train_data: rpy2.robjects.vectors.DataFrame = keel_package.loadKeelDataset("iris_train")
test_data: rpy2.robjects.vectors.DataFrame = keel_package.loadKeelDataset("iris_test")

robjects.globalenv["algorithm"] = keel_package.NNEP_C(
    train_data, test_data, generations=5
)
print(robjects.globalenv["algorithm"])

robjects.r("algorithm$run()")
predictions = robjects.r("algorithm$testPredictions")
print(predictions)
