"""
Test the Empirical Fuzzy Set code.
"""

import os
import pathlib
import unittest

import torch
import numpy as np

from fuzzy_ml.utils import set_rng
from fuzzy_ml.clustering.empirical import (
    multimodal_density,
    find_local_maxima,
    select_prototypes,
    reduce_partitioning,
    find_empirical_fuzzy_sets,
)


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_example() -> torch.Tensor:
    """
    Creates simple data to be used for an example.

    Returns:
        torch.Tensor
    """
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [4.0, 1.0, 5.0],
            [7.0, 9.0, 5.0],
            [6.0, 12.0, 4.0],
            [3.0, 2.0, 1.0],
            [8.0, 4.0, 2.0],
        ],
        device=AVAILABLE_DEVICE,
    )


def iris_example() -> torch.Tensor:
    """
    Creates Iris data to be used for an example.

    Returns:
        torch.Tensor
    """
    directory = pathlib.Path(__file__).parent.resolve()
    file_location = os.path.join(directory, "iris.npy")
    return torch.as_tensor(np.load(file_location), device=AVAILABLE_DEVICE)


class TestEDA(unittest.TestCase):
    """
    Test the Empirical Data Analysis code.
    """

    def test_frequencies(self) -> None:
        """
        Test that the calculation of frequencies is correct.

        Returns:
            None
        """
        input_train_data: torch.Tensor = simple_example()
        unique_observations, frequencies = np.unique(
            input_train_data.cpu().detach().numpy(), axis=0, return_counts=True
        )

        expected_observations = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 2, 1],
                [4, 1, 5],
                [6, 12, 4],
                [7, 9, 5],
                [8, 4, 2],
            ]
        )
        expected_frequencies = np.array([2, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(unique_observations, expected_observations))
        self.assertTrue(np.allclose(frequencies, expected_frequencies))

    def test_multimodal_density(self) -> None:
        """
        Test that the multimodal density is calculated correctly.

        Returns:
            None
        """
        input_train_data = simple_example()
        results = multimodal_density(input_train_data)
        expected_densities = torch.tensor(
            [7.656387, 4.320169, 3.8905387, 3.7994518, 2.5289514, 3.1530216, 3.5224535],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(results.densities, expected_densities))

    def test_local_maxima(self) -> None:
        """
        Test the local maxima is correctly identified.

        Returns:
            None
        """
        input_train_data = iris_example()[:, :2]
        results = multimodal_density(input_train_data)
        local_maxima = find_local_maxima(results)

        expected_local_maxima = torch.tensor(
            [
                3.23374839,
                0.77578952,
                0.7705926,
                0.733761,
                0.72368752,
                0.70805462,
                1.3454415,
                0.63649625,
                0.74044803,
                0.78845521,
                0.81159558,
                0.82581218,
                1.65969865,
                0.84377649,
                0.84058463,
                1.65266184,
                0.83306323,
                0.80048431,
                0.75573392,
                1.62536776,
                0.82735396,
                1.65959074,
                0.84376176,
                1.70095529,
                0.84077189,
                0.80716842,
                0.79335597,
                0.76850556,
                0.74592828,
                0.71807904,
                1.47799761,
                0.75141253,
                0.78148801,
                0.74894585,
                1.47110903,
                0.70534742,
                2.16605344,
                0.68773373,
                0.68350761,
                1.30547685,
                0.61796814,
                1.95040448,
                0.6089472,
                0.57731642,
                1.74999492,
                0.54668909,
                0.4904588,
                0.49492512,
                0.52275212,
                0.46709065,
                0.43971638,
                0.40200818,
                0.3828889,
                0.37969623,
                0.3716908,
                0.6046947,
                0.70621172,
                0.5835891,
                1.36977602,
                0.6179943,
                0.5983501,
                1.23130721,
                0.68837984,
                0.6162729,
                0.56449299,
                0.54488261,
                0.54550252,
                0.47674997,
                0.44057758,
                0.47357205,
                0.47692861,
                0.45099107,
                0.47591004,
                0.53393304,
                0.53582173,
                1.1308536,
                0.60055101,
                1.26804756,
                0.63139731,
                0.66416303,
                1.19624351,
                0.66414471,
                0.65607055,
                0.6864634,
                0.67182307,
                1.28536428,
                1.24910124,
                0.60271988,
                0.57593937,
                1.16029656,
                0.51992114,
                0.49598228,
                0.60124926,
                1.7217679,
                0.63883753,
                0.65290757,
                1.17698472,
                0.5078332,
                0.50636729,
                0.46008658,
                0.57210811,
                0.63899407,
                0.72683227,
                1.47829351,
                0.69762315,
                0.67572364,
                1.30344179,
                0.78472353,
                0.82498323,
                0.76397519,
                0.7354239,
                0.71491259,
                1.48255367,
                1.25388572,
                0.45523111,
                0.3519608,
                0.32549378,
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            torch.allclose(local_maxima.float(), expected_local_maxima.float())
        )

    def test_select_prototypes(self) -> None:
        """
        Test that the selected prototypes are correct.

        Returns:
            None
        """
        input_train_data = iris_example()[:, :2]
        results = multimodal_density(input_train_data)
        local_maxima = find_local_maxima(results)
        prototypes = select_prototypes(results, local_maxima)

        expected_prototypes = torch.tensor(
            [
                [4.3, 3.0],
                [4.6, 3.2],
                [4.8, 3.4],
                [4.9, 3.0],
                [5.0, 2.3],
                [5.0, 3.2],
                [5.0, 3.4],
                [5.1, 3.7],
                [5.2, 2.7],
                [5.2, 3.5],
                [5.3, 3.7],
                [5.4, 3.7],
                [5.5, 2.3],
                [5.5, 2.6],
                [5.6, 2.7],
                [5.7, 3.0],
                [5.7, 4.4],
                [5.8, 2.8],
                [6.0, 2.7],
                [6.1, 2.6],
                [6.2, 2.8],
                [6.2, 3.4],
                [6.3, 2.7],
                [6.3, 3.3],
                [6.4, 2.7],
                [6.4, 3.2],
                [6.6, 2.9],
                [6.7, 3.0],
                [6.9, 3.2],
                [7.2, 3.0],
                [7.2, 3.6],
                [7.7, 2.6],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(torch.allclose(prototypes.float(), expected_prototypes.float()))

    def test_reduce_partitioning(self) -> None:
        """
        Test reducing the number of identified prototypes by only keeping those that strongly
        envelop their nearby prototypes.

        Returns:
            None
        """
        set_rng(0)  # has a randomness component to it
        input_train_data = iris_example()[:, :2]
        results = multimodal_density(input_train_data)
        local_maxima = find_local_maxima(results)
        prototypes = select_prototypes(results, local_maxima)
        prototypes = reduce_partitioning(results, prototypes)

        expected_prototypes_centers = torch.tensor(
            [
                [4.653333, 3.16],
                [4.86, 2.3],
                [5.175, 2.8],
                [5.15, 3.51875],
                [5.690909, 2.4636364],
                [5.55, 4.0666666],
                [5.9, 2.8611112],
                [6.1833334, 3.3166666],
                [6.5238094, 2.8809524],
                [7.125, 3.05],
                [7.6, 3.7333333],
                [7.675, 2.85],
            ],
            device=AVAILABLE_DEVICE,
        )
        expected_prototypes_widths = torch.tensor(
            [
                [0.21336309, 0.18822479],
                [0.20736441, 0.18708287],
                [0.17078251, 0.24494897],
                [0.17126977, 0.15152008],
                [0.23001976, 0.168954],
                [0.2258318, 0.21602469],
                [0.21420166, 0.11950333],
                [0.19407902, 0.09831921],
                [0.18413246, 0.25023798],
                [0.18322508, 0.15118579],
                [0.36055513, 0.11547005],
                [0.05, 0.19148542],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            torch.allclose(
                prototypes.get_centers(),
                expected_prototypes_centers,
                atol=1e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                prototypes.get_widths(),
                expected_prototypes_widths,
                atol=1e-1,
            )
        )

    def test_empirical_fuzzy_sets(self) -> None:
        """
        Test the full procedure of generating Empirical Fuzzy Sets.

        Returns:
            None
        """
        set_rng(0)  # has a randomness component to it
        input_train_data = iris_example()[:, :2]
        empirical_fuzzy_sets = find_empirical_fuzzy_sets(input_train_data)
        expected_prototypes_centers = torch.tensor(
            [
                [4.653333, 3.16],
                [4.86, 2.3],
                [5.175, 2.8],
                [5.15, 3.51875],
                [5.690909, 2.4636364],
                [5.55, 4.0666666],
                [5.9, 2.8611112],
                [6.1833334, 3.3166666],
                [6.5238094, 2.8809524],
                [7.125, 3.05],
                [7.6, 3.7333333],
                [7.675, 2.85],
            ],
            device=AVAILABLE_DEVICE,
        )
        expected_prototypes_widths = torch.tensor(
            [
                [0.21336309, 0.18822479],
                [0.20736441, 0.18708287],
                [0.17078251, 0.24494897],
                [0.17126977, 0.15152008],
                [0.23001976, 0.168954],
                [0.2258318, 0.21602469],
                [0.21420166, 0.11950333],
                [0.19407902, 0.09831921],
                [0.18413246, 0.25023798],
                [0.18322508, 0.15118579],
                [0.36055513, 0.11547005],
                [0.05, 0.19148542],
            ],
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            torch.allclose(
                empirical_fuzzy_sets.get_centers(),
                expected_prototypes_centers,
                atol=1e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                empirical_fuzzy_sets.get_widths(),
                expected_prototypes_widths,
                atol=1e-1,
            )
        )
