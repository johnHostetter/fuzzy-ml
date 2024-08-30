"""
Implements the Evolving Clustering Method as described in the DENFIS paper.
"""

import gc
from typing import List, Tuple, Any, Union

import torch
import numpy as np
from regime import Node, hyperparameter

from fuzzy_ml import LabeledGaussian
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.fetchers import fetch_labeled_dataset


def general_euclidean_distance(
    vector_1: torch.Tensor, vector_2: torch.Tensor, minkowski_distance_power: int = 2
) -> torch.Tensor:
    """
    The general Euclidean distance metric as described in the DENFIS paper.

    Args:
        vector_1: A two-dimensional Tensor, where rows are number of observations
            and columns are features.
        vector_2: A one-dimensional Tensor, where the columns are features.
        minkowski_distance_power: The power to raise the Minkowski distance to.

    Returns:
        A one-dimensional Tensor, where the columns are the distances
        from each identified cluster thus far.
    """
    distances_from_clusters = vector_1 - vector_2
    return torch.pow(
        torch.pow(torch.abs(distances_from_clusters), minkowski_distance_power).sum(1),
        1 / minkowski_distance_power,
    ) / np.power(vector_2.nelement(), 0.5)


class EvolvingClusteringMethod(Node):
    """
    The Evolving Clustering Method as described in the DENFIS paper.

    The Evolving Clustering Method is a method for clustering data points in a streaming
    fashion. It is used to create clusters of data points that are close to each other
    in the feature space determined by a hyperparameter distance_threshold.
    """

    def __init__(self, resource_name: str = "labeled_clusters"):
        super().__init__(resource_name=resource_name)

    def target_edges(self) -> List[Tuple[Any, Any, int]]:
        """
        Edges that distribute the output of the algorithm to the next algorithm in the pipeline.

        Returns:
            A list of 3-tuples, where the first element is the source, the second element is the
            target, and the third element is the order of the argument for the target.
        """
        return super().target_edges() + [
            ("train_dataset", fetch_labeled_dataset, 0),
            (self.resource_name, fetch_labeled_dataset, 1),
            (fetch_labeled_dataset, "exemplars", 0),
        ]

    @hyperparameter("distance_threshold")
    def __call__(
        self,
        train_dataset: LabeledDataset,
        device: torch.device,
        distance_threshold: float,
    ) -> LabeledGaussian:
        """
        Apply the Evolving Clustering Method to the input data.

        Args:
            train_dataset: The given dataset.
            device: The device to use.
            distance_threshold: A new cluster is made if data exceeds this distance from the core.

        Returns:
            The resulting clusters.
        """
        labeled_clusters: Union[None, LabeledGaussian] = None
        for idx, observation in enumerate(train_dataset.data):
            if isinstance(observation, torch.Tensor) and observation.device != device:
                observation = observation.to(device)

            if labeled_clusters is None:
                # Step 0: Create the first cluster by simply taking the
                # position of the first example from the input stream as the
                # first cluster center Cc_{1}^{0}, and setting a value 0 for its cluster
                # radius Ru_{1} [Fig. 2(a)].
                labeled_clusters = LabeledGaussian(
                    centers=observation[None, :].cpu().detach().numpy(),
                    widths=np.array([[0.0]]),
                    device=device,
                    labels=self.get_labels(train_dataset, idx),
                )
            else:
                # Step 1: If all examples of the data stream have been processed, the algorithm
                # is finished. Else, the current input example, $x_i$, is taken and the distances
                # between this example and all $n$ already created cluster
                # centers Cc_j, D_{ij} = ||x_{i} - Cc_{j}||, j = 1, 2, ..., n, are calculated.

                with torch.no_grad():
                    distances: torch.Tensor = general_euclidean_distance(
                        labeled_clusters.get_centers(), observation
                    )

                    # returns a named tuple
                    min_and_argmin = torch.min(distances, dim=0)
                    clusters_that_satisfy_input = labeled_clusters.get_widths()[
                        (min_and_argmin.values < labeled_clusters.get_widths())
                    ]
                    if clusters_that_satisfy_input.nelement() > 0:
                        # TLDR: find the index that satisfies it and increase its support
                        # Step 2: If there is any distance value, D_{ij} = ||x_{i} - Cc_{j}||,
                        # equal to, or less than, at least one of the radii,
                        # Ru_{j}, j = 1, 2, ..., n, it means that the current example belongs to a
                        # cluster C_{m} with the minimum distance
                        #
                        # D_{im} = ||x_{i} - Cc_{m}|| = min(||x_{i} - Cc_{j}||)
                        #
                        # subject to the constraint D_{ij} < Ru_{j},   j = 1, 2, ..., n.
                        #
                        # In this case, neither a new cluster is created, nor are any existing
                        # clusters updated (the cases of $x_4$ and $x_6$ in Fig. 2);
                        # the algorithm returns to Step 1. Else—go to the next step.
                        labeled_clusters.increment_support(
                            min_and_argmin.indices.item()
                        )

                    else:
                        # Step 3: Find cluster (with center Cc_{a} and cluster radius Ru_{a})
                        # from all existing cluster centers through calculating the values
                        # S_{ij} = D_{ij} + Ru_{j}, j = 1, 2, ..., n, and then
                        # choosing the cluster center with the minimum value S_{ia}:
                        #
                        #     S_{ia} = D_{ia} + Ru_{a} = min(S_{ij}), j = 1, 2, ..., n.
                        distances_from_farthest_edge = (
                            distances + labeled_clusters.get_widths().flatten()
                        )
                        # returns a named tuple
                        min_and_argmin = torch.min(distances_from_farthest_edge, dim=0)
                        nearest_cluster_idx = min_and_argmin.indices.item()
                        # Step 4: If S_{ia} is greater than $2 * distance_threshold$, the example
                        # $x_i$ does not belong to any existing clusters. A new cluster is
                        # created in the same way as described in Step 0
                        # (the cases of $x_3$ and $x_8$ in Fig. 2), and the algorithm returns to
                        # Step 1.
                        if min_and_argmin.values.item() > (2.0 * distance_threshold):
                            # Step 0: Create the first cluster by simply taking the
                            # position of the first example from the input stream as the
                            # first cluster center Cc_{1}^{0}, and setting a value 0 for its cluster
                            # radius Ru_{1} [Fig. 2(a)].
                            labeled_clusters.add(
                                centers=observation[None, :],
                                widths=torch.tensor([[0]], device=device),
                                support=1,
                                label=self.get_labels(train_dataset, idx),
                            )
                        else:
                            # Step 5: If S_{ia} is not greater than $2 * distance_threshold$,
                            # the cluster $C_{a}$ is updated by moving its center, Cc_{a},
                            # and increasing the value of its radius, Ru_{a}. The updated radius
                            # Ru_{a}^{new} is set to be equal to S_{ia} / 2 and the new center
                            # Cc_{a}^{new} is located at the point on the line connecting the $x_i$
                            # and Cc_{a}, and the distance from the new center Cc_{a}^{new} to the
                            # point $x_{i}$ is equal to Ru_{a}^{new}
                            # (the cases of $x_2$, $x_5$, $x_7$ and $x_9$ in Fig. 2).
                            # The algorithm returns to Step 1
                            with torch.no_grad():
                                values = labeled_clusters._widths[0].index_copy(
                                    dim=0,
                                    index=torch.tensor(
                                        nearest_cluster_idx, device=device
                                    ),
                                    source=(min_and_argmin.values / 2.0)
                                    .reshape(1, 1)
                                    .clone()
                                    .detach(),
                                )
                                labeled_clusters._widths[0] = torch.nn.Parameter(values)

                                # keep a running mean approximation of the cluster center
                                labeled_clusters._centers[0][nearest_cluster_idx] = (
                                    torch.nn.Parameter(
                                        (
                                            labeled_clusters.supports[
                                                nearest_cluster_idx
                                            ]
                                            * labeled_clusters._centers[0][
                                                nearest_cluster_idx
                                            ]
                                            + observation
                                        )
                                        / (
                                            labeled_clusters.supports[
                                                nearest_cluster_idx
                                            ]
                                            + 1
                                        )
                                    )
                                )

                        del distances_from_farthest_edge

                    # delete unused tensors & variables from memory
                    del (
                        distances,
                        min_and_argmin,
                        clusters_that_satisfy_input,
                    )

        gc.collect()
        return labeled_clusters

    def get_labels(
        self, labeled_dataset: LabeledDataset, idx: int
    ) -> Union[None, List[torch.Tensor]]:
        """
        Get the label for the given index if the dataset is labeled.

        Args:
            labeled_dataset: The labeled dataset.
            idx: The index to get the label for.

        Returns:
            The label for the given index. If the dataset is not labeled, None is returned.
        """
        labels: Union[None, List[torch.Tensor]] = None
        if labeled_dataset.labels is not None:
            labels: List[torch.Tensor] = [labeled_dataset.labels[idx]]
        return labels
