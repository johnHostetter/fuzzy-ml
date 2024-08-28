"""
Test that the FuzzyRuleBasedSystem (frbs) package written in R can be imported and used from Python.

The following may be necessary on Windows to set the 'R_HOME' for rpy2 correctly:

    from rpy2 import situation
    import os
    os.environ['R_HOME'] = situation.r_home_from_registry()
    situation.get_r_home()
"""

import unittest

from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


class TestRoughSets(unittest.TestCase):
    """
    Test the RoughSets package written in R.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("frbs")
        self.frbs_package = importr("frbs")

    def test_frbs(self) -> None:
        """
        Test the frbs package. The following code is adapted from the
        frbs package documentation. See:

            https://cran.r-project.org/web/packages/frbs/frbs.pdf

        Specifically, this is a reproduction of the example on page 60 of the
        documentation just beneath the predict.frbs function within the Examples section.

        Returns:
            None
        """
        train_vector = robjects.FloatVector(
            [
                5.2,
                -8.1,
                4.8,
                8.8,
                -16.1,
                4.1,
                10.6,
                -7.8,
                5.5,
                10.4,
                -29.0,
                5.0,
                1.8,
                -19.2,
                3.4,
                12.7,
                -18.9,
                3.4,
                15.6,
                -10.6,
                4.9,
                1.9,
                -25.0,
                3.7,
                2.2,
                -3.1,
                3.9,
                4.8,
                -7.8,
                4.5,
                7.9,
                -13.9,
                4.8,
                5.2,
                -4.5,
                4.9,
                0.9,
                -11.6,
                3.0,
                11.8,
                -2.1,
                4.6,
                7.9,
                -2.0,
                4.8,
                11.5,
                -9.0,
                5.5,
                10.6,
                -11.2,
                4.5,
                11.1,
                -6.1,
                4.7,
                12.8,
                -1.0,
                6.6,
                11.3,
                -3.6,
                5.1,
                1.0,
                -8.2,
                3.9,
                14.5,
                -0.5,
                5.7,
                11.9,
                -2.0,
                5.1,
                8.1,
                -1.6,
                5.2,
                15.5,
                -0.7,
                4.9,
                12.4,
                -0.8,
                5.2,
                11.1,
                -16.8,
                5.1,
                5.1,
                -5.1,
                4.6,
                4.8,
                -9.5,
                3.9,
                13.2,
                -0.7,
                6.0,
                9.9,
                -3.3,
                4.9,
                12.5,
                -13.6,
                4.1,
                8.9,
                -10.0,
                4.9,
                10.8,
                -13.5,
                5.1,
            ]
        )
        train_data_matrix = robjects.r["matrix"](train_vector, ncol=3, byrow=True)
        train_data_matrix.colnames = robjects.StrVector(["inp.1", "inp.2", "out.1"])
        print(train_data_matrix)

        data_fit = train_data_matrix.rx(True, robjects.IntVector((1, 2)))
        print(data_fit)

        test_vector = robjects.FloatVector(
            [
                10.5,
                -0.9,
                5.8,
                -2.8,
                8.5,
                -0.6,
                13.8,
                -11.9,
                9.8,
                -1.2,
                11.0,
                -14.3,
                4.2,
                -17.0,
                6.9,
                -3.3,
                13.2,
                -1.9,
            ]
        )
        test_data_matrix = robjects.r["matrix"](test_vector, ncol=2, byrow=True)

        # load the 'base' R package for accessing R's basic functions
        base = importr("base")

        # calculate the range of each column using the 'range' function in R
        range_data = base.apply(train_data_matrix, 2, base.range)

        # set the method.type as a string in Python
        method_type = "WM"

        # collect control parameters into a list (named list in R)
        wang_mendel_control = robjects.ListVector(
            {
                "num.labels": 3,
                "type.mf": "GAUSSIAN",
                "type.tnorm": "MIN",
                "type.defuz": "WAM",
                "type.implication.func": "ZADEH",
                "name": "Sim-0",
            }
        )

        # learn the FRBS model using the 'frbs_learn' function in R and the Wang-Mendel method
        wang_mendel_object = self.frbs_package.frbs_learn(
            train_data_matrix, range_data, method_type, wang_mendel_control
        )

        # predict the output values using the 'predict_frbs' function in R
        wang_mendel_predictions = self.frbs_package.predict_frbs(
            wang_mendel_object, test_data_matrix
        )
        print(list(wang_mendel_predictions))

        expected_predictions = [
            4.99864192655538,
            4.537781270315337,
            4.815638970562045,
            4.126365559523999,
            4.888854200137037,
            4.099266450792342,
            3.5497141412115636,
            4.706904694862706,
            5.492127995281615,
        ]

        self.assertEqual(len(expected_predictions), len(wang_mendel_predictions))
        for idx, expected_prediction in enumerate(expected_predictions):
            self.assertAlmostEqual(
                expected_prediction, wang_mendel_predictions[idx], places=6
            )
