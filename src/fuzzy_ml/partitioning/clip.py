"""
Implements the Categorical Learning Induced Partitioning (CLIP) algorithm.
"""

import gc
from typing import List

import torch
import numpy as np
from fuzzy.sets import Gaussian, FuzzySet
from fuzzy.logic.variables import LinguisticVariables
from regime import hyperparameter

from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.partitioning.meta import MetaPartitioner
from .utils import find_widths, regulator


class CategoricalLearningInducedPartitioning(MetaPartitioner):
    """
    Categorical Learning Induced Partitioning (CLIP), is an unsupervised learning
    algorithm that produces and incrementally adjusts Gaussian fuzzy sets.
    """

    @hyperparameter("epsilon", "adjustment")
    def __call__(
        self,
        train_dataset: LabeledDataset,
        device: torch.device,
        epsilon: float,
        adjustment: float,
    ) -> LinguisticVariables:
        """
        Prepare linguistic variables according to CLIP (i.e., pass the relevant hyperparameters)

        Args:
            train_dataset: The given dataset.
            device: The device to use.
            epsilon: A threshold value to determine if a new fuzzy set should be created.
            adjustment: A hyperparameter to adjust the generated widths' coverage.

        Returns:
            Linguistic variables for the input and target variables (if data is labeled).
        """
        return self.make_linguistic_variables(
            train_dataset=train_dataset,
            device=device,
            epsilon=epsilon,
            adjustment=adjustment,
        )

    @hyperparameter("epsilon", "adjustment")
    def algorithm(
        self,
        observations: torch.Tensor,
        device: torch.device,
        epsilon: float,
        adjustment: float,
    ) -> List[FuzzySet]:
        """
        The inner call of the CLIP algorithm. This is where the actual work is done.

        Args:
            observations: The selected observations from the dataset.
            device: The device to use.
            epsilon: A threshold value to determine if a new fuzzy set should be created.
            adjustment: A hyperparameter to adjust the generated widths' coverage.

        Returns:
            A list of Gaussian fuzzy sets along the tensor space.
        """
        for observation in observations:
            if observation.device != device:
                observation = observation.to(device)

            if not self.terms:
                # no fuzzy clusters yet, create the first fuzzy cluster
                self.make_first_fuzzy_sets(
                    observation,
                    adjustment,
                    device=device,
                )
            else:
                # calculate the similarity between the input and existing fuzzy clusters
                for dim, _ in enumerate(observation):
                    membership_degrees: torch.Tensor = self.terms[dim](
                        observation[dim]
                    ).degrees
                    if membership_degrees.is_sparse:
                        membership_degrees = membership_degrees.to_dense()

                    if torch.max(membership_degrees) <= epsilon:
                        #  best matched cluster is unable to give satisfactory
                        #  description of the presented value
                        # a new cluster is created in the dimension based on the presented value
                        with torch.no_grad():
                            (
                                left_neighbor_idx,
                                right_neighbor_idx,
                            ) = find_indices_to_closest_neighbors(
                                observation, self.terms, dim
                            )
                            indices, sigmas = [], []

                            # --- is there a left neighbor to this new fuzzy set? ---
                            left_neighbor_sigma = update_neighbor_sigma(
                                observation,
                                self.terms,
                                dim,
                                left_neighbor_idx,
                                adjustment,
                            )
                            if left_neighbor_sigma is not None:
                                indices.append(left_neighbor_idx)
                                sigmas.append(left_neighbor_sigma)

                            # --- is there a right neighbor to this new fuzzy set? ---
                            right_neighbor_sigma = update_neighbor_sigma(
                                observation,
                                self.terms,
                                dim,
                                right_neighbor_idx,
                                adjustment,
                            )
                            if right_neighbor_sigma is not None:
                                indices.append(right_neighbor_idx)
                                sigmas.append(right_neighbor_sigma)

                            if len(sigmas) == 2:
                                new_sigma = regulator(
                                    right_neighbor_sigma, left_neighbor_sigma
                                )
                                new_sigma = torch.tensor([new_sigma] * 2, device=device)
                            else:
                                new_sigma = torch.tensor(sigmas, device=device)

                            indices = torch.tensor(indices, device=device)
                            new_sigma = new_sigma[None, :]

                            # update the existing terms to make room for the new term
                            self.terms[dim]._widths[0] = torch.nn.Parameter(
                                self.terms[dim]
                                ._widths[0]
                                .index_copy(
                                    dim=-1,  # was 0
                                    index=indices,
                                    source=new_sigma.float().to(
                                        self.terms[dim]._widths[0].device
                                    ),
                                )
                            )
                            # add the newly created fuzzy set to the
                            # linguistic terms on the current dimension
                            if (
                                len(sigmas) == 2
                            ):  # if we edited both left and right neighbors
                                new_sigma = new_sigma[0][-1]  # only use a single value
                            self.terms[dim].extend(
                                centers=observation[dim].reshape(
                                    1, 1
                                ),  # reshape to (1, 1) shape
                                widths=new_sigma.reshape(
                                    1, 1
                                ),  # reshape to (1, 1) shape
                                mode="horizontal",
                            )
                        # delete unused tensors & variables from memory
                        del (
                            left_neighbor_sigma,
                            right_neighbor_sigma,
                            indices,
                            sigmas,
                            new_sigma,
                        )

                    # no longer need membership degrees
                    del membership_degrees

        gc.collect()
        return self.terms

    def make_first_fuzzy_sets(
        self, data_point: np.ndarray, alpha: float, device: torch.device
    ):
        """
        The first encountered observation is used to initialize a fuzzy set
        within each input dimension.

        Args:
            data_point: A single input_data observation where
                each column is a feature/attribute.
            alpha: A hyperparameter to adjust the generated widths' coverage.
            device: The device to use.

        Returns:
            None
        """
        widths: torch.Tensor = find_widths(
            data_point, self.minimums, self.maximums, alpha
        )
        for center, width in zip(data_point, widths):
            self.terms.append(
                Gaussian(
                    centers=np.array([center.cpu().detach().item()]),
                    widths=np.array([width.cpu().detach().item()]),
                    device=device,
                )
            )
        del widths


def update_neighbor_sigma(data_point, terms, dim, neighbor_idx, alpha):
    """
    Given that we have identified a neighbor to a new fuzzy set that is being created,
    fetch its center and sigma values, and update its sigma values in response to the new fuzzy set.
    Does nothing and returns 'None' if the 'neighbor_idx' is 'None'.

    Args:
        data_point (1D Numpy array): A single input_data observation
            where each column is a feature/attribute.
        terms (list): A list of terms that have already been created that CLIP
            should consider when producing more terms. Default is None, which means
            no existing terms will be passed to CLIP.
        dim (int): The index of the current dimension.
        neighbor_idx (int): The index associated with this new fuzzy set's neighbor;
            could be left or right neighbor.
        alpha (float): A hyperparameter to adjust the generated widths' coverage.

    Returns:
        The new sigma for the newly created fuzzy set's neighbor.
        Is 'None' if the given neighbor's index is 'None'. A neighbor_idx of 'None' means that
        the newly created fuzzy set has no neighbor on its (left/right) side.
    """
    regulated_sigma = None
    if neighbor_idx is not None:
        center, sigma = (
            terms[dim].get_centers()[0][neighbor_idx],
            terms[dim].get_widths()[0][neighbor_idx],
        )
        adjusted_sigma = torch.sqrt(
            -1.0
            * (torch.pow(center - data_point[dim], 2) / torch.log(torch.tensor(alpha)))
        )
        regulated_sigma = regulator(adjusted_sigma, sigma)
    return regulated_sigma


def find_indices_to_closest_neighbors(data_point, terms, dim):
    """
    For the given dimension (i.e., 'dim'), calculates the distance between data_point[dim] and the
    existing linguistic terms that occupy said dimension. From there, it then determines
    which existing linguistic terms are to the left of data_point[dim], and which are to the
    right of data_point[dim]. If there are existing linguistic terms to the left of data_point[dim],
    the nearest one (w/ smallest distance to data_point[dim]) will have its index returned
    as 'left_neighbor_idx'; else, 'None' will be returned in its place. If there are
    existing linguistic terms to the right of data_point[dim], the nearest one (w/ smallest
    distance to data_point[dim]) will have its index returned as 'right_neighbor_idx'; else, 'None'
    will be returned in its place.

    Args:
        data_point (1D Numpy array): A single input_data observation
            where each column is a feature/attribute.
        terms (list): A list of terms that have already been created that CLIP
            should consider when producing more terms. Default is None, which means
            no existing terms will be passed to CLIP.
        dim (int): The index of the current dimension.

    Returns:
        left_neighbor_idx (None if no left neighbor exists),
        right_neighbor_idx (None if no right neighbor exists)
    """
    differences = (
        data_point[dim] - terms[dim].get_centers()
    )  # differences b/w this data_point's center and terms' centers
    left_neighbor_differences = torch.where(
        differences > 0, differences, float("inf")
    ).min(dim=-1)
    right_neighbor_differences = torch.where(
        differences < 0, differences, -float("inf")
    ).max(dim=-1)
    if left_neighbor_differences.values == float("inf"):
        left_neighbor_idx = None
    else:
        left_neighbor_idx = left_neighbor_differences.indices.item()
    if right_neighbor_differences.values == -float("inf"):
        right_neighbor_idx = None
    else:
        right_neighbor_idx = right_neighbor_differences.indices.item()

    del differences, left_neighbor_differences, right_neighbor_differences
    return left_neighbor_idx, right_neighbor_idx
