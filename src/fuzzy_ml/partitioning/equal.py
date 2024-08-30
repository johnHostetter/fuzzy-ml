"""
Implements the Expert Partitioning algorithm. Basically, Expert Partitioning finds the minimum
and maximum values possible in some dimension, and then creates (reasonable) fuzzy sets across it
such that each fuzzy set has reasonable spread or coverage of the domain space.
"""

from typing import List

import torch
from fuzzy.sets import FuzzySet
from fuzzy.logic.variables import LinguisticVariables
from regime import hyperparameter

from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.partitioning.meta import MetaPartitioner


class EqualPartitioning(MetaPartitioner):
    """
    The Equal Partitioning algorithm is a simple way to create linguistic variables.
    """

    @hyperparameter("set_type", "n_splits")
    def __call__(
        self,
        train_dataset: LabeledDataset,
        device: torch.device,
        set_type: FuzzySet,
        n_splits: int,
    ) -> LinguisticVariables:
        """
        Prepare linguistic variables according to Equal Partitioning
        (i.e., pass the relevant hyperparameters)

        Args:
            train_dataset: The given dataset.
            device: The device to use.
            set_type: The type of fuzzy set to create.
            n_splits: The number of splits, or partitions, to make in each dimension.

        Returns:
            Linguistic variables for the input and target variables (if data is labeled).
        """
        return self.make_linguistic_variables(
            train_dataset, device=device, set_type=set_type, n_splits=n_splits
        )

    @hyperparameter("set_type", "n_splits")
    def algorithm(
        self,
        observations: torch.Tensor,
        device: torch.device,
        set_type: FuzzySet,
        n_splits: int,
    ) -> List[FuzzySet]:
        """
        Expert partitioning, is an unsupervised algorithm that produces fuzzy sets.

        Args:
            observations: The selected observations from the dataset.
            device: The device to use.
            set_type: The type of fuzzy set to create.
            n_splits: The number of splits, or partitions, to make in each dimension.

        Returns:
            A list of fuzzy sets along the tensor space.
        """
        std_dev = torch.std(observations, dim=0)
        for minimum, maximum in zip(self.minimums, self.maximums):
            centers = torch.linspace(minimum, maximum, steps=n_splits)
            widths = (
                (
                    torch.nn.functional.pad(centers, (0, 1), "constant", 0.0)
                    - torch.nn.functional.pad(centers, (1, 0), "constant", 0.0)
                )
                + std_dev.mean()  # this was 0.1 in the original code
            ).abs() / 2
            self.terms.append(
                set_type(
                    centers=centers,
                    widths=widths[:-1],
                    device=device,
                )
            )

        return self.terms
