"""
Test the Evolving Clustering Method and its accompanying functions.
"""

import os
import pathlib
import unittest

import torch
import numpy as np
from fuzzy.sets.continuous.impl import Gaussian

from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.clustering.ecm import (
    EvolvingClusteringMethod as ECM,
    general_euclidean_distance,
)


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTANCE_THRESHOLD = 0.7


class TestECM(unittest.TestCase):
    """
    Test the Evolving Clustering Method and its accompanying functions, such as the general
    Euclidean distance metric, and the algorithm itself.
    """

    def test_general_euclidean_distance(self) -> None:
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:

        """
        vector_1 = np.array([0.5, 0.8])
        vector_2 = np.array([0.9, 2.5])
        distance = general_euclidean_distance(
            torch.tensor(np.array([vector_1])), torch.tensor(vector_2)
        ).float()
        expected_distance = torch.tensor([1.2349089]).float()
        self.assertTrue(torch.allclose(distance, expected_distance))

    def test_ecm_output(self) -> None:
        """
        The ECM that is originally defined without using PyTorch should identify the same
        number of exemplars as the PyTorch implementation (i.e., the new ECM directly induces
        Fuzzy Set Base PyTorch modules). However, the identified CENTERS and WIDTHS should
        be UNEQUAL, since the original implementation had some minor mistakes.

        Uses sample input from the Cart Pole FCQL demo.

        Returns:

        """
        directory = pathlib.Path(__file__).parent.resolve()
        file_location = os.path.join(directory, "ecm_input.npy")
        input_data = np.load(file_location)

        labeled_clusters = ECM()(
            train_dataset=LabeledDataset(data=torch.tensor(input_data), out_features=1),
            device=AVAILABLE_DEVICE,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        expected_clusters_centers = np.array(
            [
                [0.04537525, 0.16438581, -0.08087717, -0.45448843],
                [-0.11325858, -1.1463842, -0.17513412, 0.33120227],
                [0.07269111, 1.2100581, -0.03719315, -1.5913403],
                [-1.307729, -1.9523882, 0.05424051, 0.8693342],
                [-1.9041944, -0.68538105, 0.19603494, -0.08774516],
                [0.55375546, 1.2185665, 0.15094389, 0.40546915],
                [1.1379728, 2.4252014, 0.1489413, -0.6652429],
                [2.3874073, 3.068543, 0.11382611, -0.49637064],
                [-1.3382974, 0.91613597, 0.1543206, -0.7573316],
                [-0.06728873, -1.3561076, 0.09651103, 2.0083315],
                [-2.2547994, -3.0038352, -0.08159605, 0.66267693],
            ]
        )
        expected_clusters_widths = np.array(
            [
                [0.68368685],
                [0.6805301],
                [0.6811912],
                [0.6908445],
                [0.67449474],
                [0.56174624],
                [0.6951507],
                [0.0],
                [0.0],
                [0.0],
                [0.5624561],
            ]
        )
        expected_clusters = Gaussian(
            centers=expected_clusters_centers,
            widths=expected_clusters_widths,
            device=AVAILABLE_DEVICE,
        )
        self.assertTrue(
            np.allclose(
                labeled_clusters.clusters.get_centers().cpu().detach().numpy(),
                expected_clusters_centers,
            )
        )
        self.assertTrue(
            np.allclose(
                labeled_clusters.clusters.get_widths().cpu().detach().numpy(),
                expected_clusters_widths,
            )
        )
        # test that Gaussian was properly created
        self.assertTrue(
            torch.allclose(
                labeled_clusters.clusters.get_centers(),
                expected_clusters.get_centers(),
            )
        )
        self.assertTrue(
            torch.allclose(
                labeled_clusters.clusters.get_widths(),
                expected_clusters.get_widths(),
            )
        )
