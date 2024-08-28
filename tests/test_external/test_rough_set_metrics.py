"""
Test that the RoughSets package written in R can be imported and used from Python.

The following may be necessary on Windows to set the 'R_HOME' for rpy2 correctly:

    from rpy2 import situation
    import os
    os.environ['R_HOME'] = situation.r_home_from_registry()
    situation.get_r_home()
"""

import unittest

from pandas import DataFrame
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr, data


class TestRoughSets(unittest.TestCase):
    """
    Test the RoughSets package written in R.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("RoughSets")
        self.rs_package = importr("RoughSets")
        self.rs_data = data(self.rs_package).fetch("RoughSetData")["RoughSetData"]
        print(f"Available datasets: {list(self.rs_data.names)}")

    def test_rough_sets(self) -> None:
        """
        Test the rough sets package. The following code is adapted from the
        RoughSets package documentation. See:
            https://cran.r-project.org/web/packages/RoughSets/RoughSets.pdf

        Specifically, this is a reproduction of the example on page 20 of the
        documentation for the BC.boundary.reg.RST function.

        Note that typically the RST functions take a data.frame as input, but
        the data.frame is not a native Python object. Instead, we use the
        SF_asDecisionTable function to convert the data.frame into a native
        Python object, which is a list of lists. The first list is the header
        and the remaining lists are the rows of the data.frame.

        Also, the RST functions typically take a feature set as input, but
        the feature set is not a native Python object. Instead, we use the
        r['c'] function to convert a Python list into an R vector.

        Lastly, most of the functions are nested within the RoughSets package using
        a dot notation. For example, the BC.IND.relation.RST function is nested
        within the BC function. To access the BC.IND.relation.RST function in Python,
        we replace the dot notation with underscores. For example, the following:

            BC_IND_relation_RST

        is equivalent to the following:

            self.rs_package.BC.IND.relation.RST

        where self.rs_package is the RoughSets package imported from R. This is
        because the RoughSets package is imported as a Python object, which is
        a dictionary. The keys of the dictionary are the functions in the
        RoughSets package. The values of the dictionary are the functions in the
        RoughSets package as Python objects.

        Returns:
            None
        """
        rs_hiring_data = self.rs_data[0]  # index 0 is the hiring.dt dataset
        rs_hiring_table = self.rs_package.SF_asDecisionTable(rs_hiring_data)
        indiscernibility_relation = self.rs_package.BC_IND_relation_RST(
            rs_hiring_table, feature_set=robjects.r["c"](2)
        )
        roughset = self.rs_package.BC_LU_approximation_RST(
            rs_hiring_data, indiscernibility_relation
        )
        print(f"Possible approximations for the rough set: {list(roughset.names)}")
        # the zero index refers to the first option, which is the lower approximation
        print(
            f"Possible outcomes for the lower approximation: {list(roughset[0].names)}"
        )
        # the indexing of [0][0] refers to the first option, which is Accept
        # and the indexing of [0][1] refers to the second option, which is Reject
        self.assertTrue(list(roughset[0][0]), [2, 3, 4])  # Accept
        self.assertTrue(list(roughset[0][1]), [2, 3, 4])  # Reject

        pos_boundary = self.rs_package.BC_boundary_reg_RST(rs_hiring_data, roughset)
        print(
            f"Possible variables for the positive boundary: {list(pos_boundary.names)}"
        )
        self.assertTrue(list(pos_boundary[0]), [1, 7])
        self.assertTrue(pos_boundary[1][0], 0.25)

    def test_learning_from_examples(self) -> None:
        """
        Test the learning from examples function in the RoughSets package.
        """
        rs_wine_data = self.rs_data[-2]  # index -2 is the wine dataset
        rs_wine_train_data = rs_wine_data.rx(robjects.IntVector((1, 50)), True)
        rs_wine_table = self.rs_package.SF_asDecisionTable(
            rs_wine_train_data, decision_attr=14, indx_nominal=14
        )
        cut_values = self.rs_package.D_discretization_RST(
            rs_wine_table, type_method="local.discernibility", maxNOfCuts=1
        )
        rs_wine_decision_table = self.rs_package.SF_applyDecTable(
            rs_wine_table, cut_values
        )
        rules = self.rs_package.RI_LEM2Rules_RST(rs_wine_decision_table)
        print(rules)

        # get predictions using the rules
        predictions = self.rs_package.predict_RuleSetRST(rules, rs_wine_decision_table)
        pandas2ri.activate()
        pandas2ri.ri2py_vector(predictions)
        with (robjects.default_converter + pandas2ri.converter).context():
            predictions: DataFrame = robjects.conversion.get_conversion().rpy2py(
                predictions
            )
        self.assertEqual(predictions.values.squeeze().astype(int).tolist(), [1, 2])

        # we can also pass the Python variable reference to the R function
        robjects.globalenv["rules"] = rules
        robjects.globalenv["rs_wine_decision_table"] = rs_wine_decision_table
        predictions = robjects.r("predict(rules, rs_wine_decision_table)")
        with (robjects.default_converter + pandas2ri.converter).context():
            predictions: DataFrame = robjects.conversion.get_conversion().rpy2py(
                predictions
            )
        self.assertEqual(predictions.values.squeeze().astype(int).tolist(), [1, 2])
