"""
Contains granulation or partition algorithms that use fuzzy theory (could be online or offline).
"""

from abc import abstractmethod
from typing import Union, List

import torch
from regime import Node
from fuzzy.sets import FuzzySet
from fuzzy.logic.variables import LinguisticVariables

from fuzzy_ml.datasets import LabeledDataset


class MetaPartitioner(Node):
    """
    A common metaclass for algorithms that facilitate partitioning of data.
    """

    def __init__(
        self,
        resource_name: str = "linguistic_variables",
        minimums=None,
        maximums=None,
        terms=None,
    ):
        """
        Initialize the partitioning algorithm with the minimum and maximum values of the
        observations. This assumes we are partitioning to create linguistic variables.

        Args:
            resource_name (str): The name of the resource to create. Default is
            "linguistic_variables".
            minimums (iterable): The minimum value per feature in X.
            maximums (iterable): The maximum value per feature in X.
            terms (list): A list of terms that have already been created that CLIP
                should consider when producing more terms. Default is None, which means
                no existing terms will be passed to CLIP.

        Returns:
            A list of Gaussian fuzzy sets along the tensor space.
        """
        super().__init__(resource_name=resource_name)
        if terms is None:
            terms = []
        # we can initialize the algorithm with existing values
        # (as if we're continuing from a previous run)
        self.minimums: Union[None, torch.Tensor] = minimums
        self.maximums: Union[None, torch.Tensor] = maximums
        self.terms: List[FuzzySet] = terms

    def setup(self, observations: torch.Tensor):
        """
        Set up the CLIP algorithm with the minimum and maximum values of the observations.

        Args:
            observations: The selected observations from the dataset.

        Returns:
            None
        """
        if self.minimums is None:
            self.minimums = torch.min(observations, dim=0).values
        if self.maximums is None:
            self.maximums = torch.max(observations, dim=0).values

    def make_linguistic_variables(
        self, train_dataset: LabeledDataset, device: torch.device, *args, **kwargs
    ) -> LinguisticVariables:
        """
        Create the linguistic variables for the training dataset.

        Args:
            train_dataset: The training dataset.
            device: The device to use.
            *args: Optional arguments.
            **kwargs: Optional keyword arguments.

        Returns:
            The linguistic variables for the training dataset.
        """
        self.setup(observations=train_dataset.data)
        input_variables = self.algorithm(
            observations=train_dataset.data, device=device, *args, **kwargs
        )
        target_variables = []
        if train_dataset.labels is not None:
            self.setup(observations=train_dataset.labels)
            target_variables = self.algorithm(
                observations=train_dataset.labels, device=device, *args, **kwargs
            )

        return LinguisticVariables(inputs=input_variables, targets=target_variables)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclass calls `make_linguistic_variables` with the necessary hyperparameters,"
            "and decorates this __call__ with the names of those hyperparameters."
        )

    @abstractmethod
    def algorithm(self, observations, device, *args, **kwargs) -> List[FuzzySet]:
        """
        The actual algorithm that partitions the data.

        Args:
            observations: The observations to partition.
            device: The device to use.
            *args: Optional arguments.
            **kwargs: Optional keyword arguments.

        Returns:

        """
        raise NotImplementedError(
            "Subclass overrides this with the actual algorithm (e.g., CLIP)."
        )
