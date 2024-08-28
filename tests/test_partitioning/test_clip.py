"""
Test various components necessary in implementing the
Categorical Learning Induced Partitioning algorithm.
"""

import os
import pathlib
import unittest

import torch
import numpy as np
from fuzzy.logic import LinguisticVariables
from fuzzy.sets.continuous.impl import Gaussian

from fuzzy_ml.utils import set_rng
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.partitioning.clip import (
    regulator,
    find_indices_to_closest_neighbors,
    CategoricalLearningInducedPartitioning as CLIP,
)


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCLIP(unittest.TestCase):
    """
    Test the Categorical Learning Induced Partitioning algorithm, and its supporting functions
    such as the 'regular' function.
    """

    def test_regulator(self) -> None:
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:
            None
        """
        assert regulator(sigma_1=1.0, sigma_2=1.0) == 1.0
        assert regulator(sigma_1=0.5, sigma_2=1.0) == 0.75
        assert regulator(sigma_1=1.0, sigma_2=0.5) == 0.75
        assert regulator(sigma_1=0.5, sigma_2=0.5) == 0.5
        assert regulator(sigma_1=0, sigma_2=1.0) == 0.5

    def test_find_indices_to_closest_neighbors(self) -> None:
        """
        Test that the 'find_indices_to_closest_neighbors' function correctly identifies the
        data observation's left and right neighbor indices.

        Returns:
            None
        """
        # a new cluster is created in the input dimension based on the presented value
        dimension = 1
        element = torch.tensor([0.0, 3.0, 2.0, 3], device=AVAILABLE_DEVICE)
        terms = [
            [],
            Gaussian(
                centers=np.array([-2.0, 2.0, 4.0, 6]),
                widths=np.array([0.4963, 0.7682, 0.0885, 0.1320]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        left_neighbor_idx, right_neighbor_idx = find_indices_to_closest_neighbors(
            element, terms, dimension
        )

        assert left_neighbor_idx == 1
        assert right_neighbor_idx == 2

    def test_clip_on_random_data(self) -> None:
        """
        Testing how CLIP performs when given some random data.

        Returns:
            None
        """
        set_rng(0)
        directory = pathlib.Path(__file__).parent.resolve()
        file_path = os.path.join(directory, "random_train_data.npy")
        input_data = np.load(file_path)
        linguistic_variables: LinguisticVariables = CLIP()(
            train_dataset=LabeledDataset(
                data=torch.tensor(input_data, dtype=torch.float32), out_features=1
            ),
            epsilon=0.6,
            adjustment=0.2,
            device=AVAILABLE_DEVICE,
        )
        expected_terms = [
            Gaussian(
                centers=np.array([0.5488135, 0.96366276, 0.0202184]),
                widths=np.array([0.38547637, 0.3542887, 0.38547637]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.71518937, 0.38344152, 0.10204481, 0.95274901]),
                widths=np.array([0.25629429, 0.27357153, 0.27357153, 0.25629429]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.60276338, 0.07103606, 0.94374808]),
                widths=np.array([0.33559839, 0.40241628, 0.33559839]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.54488318, 0.891773, 0.0871293]),
                widths=np.array([0.34311335, 0.32540311, 0.34311335]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        for linguistic_term, expected_term in zip(
            linguistic_variables.inputs, expected_terms
        ):
            assert torch.allclose(
                linguistic_term.get_centers(), expected_term.get_centers()
            )
            assert torch.allclose(
                linguistic_term.get_widths(), expected_term.get_widths()
            )

        # compare_results(oldCLIP_terms, newCLIP_terms)
