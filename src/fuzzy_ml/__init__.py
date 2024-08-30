"""
This module contains helpful features used throughout the fuzzy-ml package.
"""

from typing import List, Union

import torch

from fuzzy.sets import Gaussian


class LabeledGaussian(Gaussian):
    """
    Gaussian cluster(s) with label(s).
    """

    def __init__(self, centers, widths, device: torch.device, labels):
        super().__init__(centers, widths, device)
        self.supports: List[int] = [1] * centers.shape[0]
        self._labels: Union[None, List[torch.Tensor]] = (
            labels if labels is not None else None
        )

    @property
    def labels(self) -> Union[None, torch.Tensor]:
        """
        Get the labels for the Gaussian clusters.

        Returns:
            The labels for the Gaussian clusters.
        """
        if self._labels is None:
            return None
        return torch.stack(self._labels)

    def increment_support(self, index: int):
        """
        Increment the support of the Gaussian cluster at the given index.

        Args:
            index: The index of the Gaussian cluster to increment the support.

        Returns:
            None
        """
        self.supports[index] += 1

    def add(self, centers: torch.Tensor, widths: torch.Tensor, support, label):
        """
        Add a new Gaussian cluster to the current Gaussian clusters.

        Args:
            centers: The centers of the new Gaussian cluster.
            widths: The widths of the new Gaussian cluster.
            support: The support of the new Gaussian cluster.
            label: The label of the new Gaussian cluster.

        Returns:

        """
        self.extend(centers, widths, mode="vertical")
        self.supports.append(support)
        if self._labels is not None and isinstance(label, torch.Tensor):
            self._labels.append(label)
