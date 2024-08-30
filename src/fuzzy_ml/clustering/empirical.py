"""
Implements the empirical fuzzy sets algorithm.
"""

from collections import namedtuple

import torch
import numpy as np
from fuzzy.sets import Gaussian as Cluster

MultimodalDensity = namedtuple(
    "MultimodalDensity", "uniques frequencies distances densities"
)


def memory_efficient_distance_matrix(
    vector: torch.Tensor, other_vector: torch.Tensor
) -> torch.Tensor:
    """
    A memory efficient distance matrix calculation that can
    substitute PyTorch's torch.cdist().

    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/4

    Args:
        vector: A vector.
        other_vector: Some other vector.

    Returns:
        The distance matrix between 'vector' and 'other_vector'.
    """
    vector = vector.unsqueeze(1).expand(
        vector.size(0), other_vector.size(0), vector.size(1)
    )
    other_vector = other_vector.unsqueeze(0).expand(
        vector.size(0), other_vector.size(0), vector.size(1)
    )

    return torch.pow(vector - other_vector, 2).sum(2)


def multimodal_density(input_data: torch.Tensor):
    """
    Calculates the multimodal density of the multivariate input data.

    Args:
        input_data: Multivariate input data, PyTorch tensor.

    Returns:
        A namedtuple, MultimodalDensity, with attributes:
            uniques, frequencies, distances, densities.

        The uniques field contains only the unique observations from input_data, the frequencies
        describes how often each unique observation occurs in input_data, the distances is a matrix
        showing the distances between each unique observation (as seen in uniques), and the
        densities attribute is the calculated density for each unique observation.
    """
    unique_observations, frequencies = input_data.unique(dim=0, return_counts=True)
    distances = torch.cdist(unique_observations, unique_observations)

    numerator = torch.pow(distances, 2).sum()
    denominator = 2 * input_data.shape[0] * distances.sum(dim=-1)
    densities = frequencies * (numerator / denominator)
    return MultimodalDensity(unique_observations, frequencies, distances, densities)


def find_local_maxima(results: namedtuple):
    """
    Prepares the results from the multimodal density calculations for the
    identification of the local maxima.

    Args:
        results: A namedtuple, MultimodalDensity, with attributes:
            uniques, frequencies, distances, densities.

    Returns:
        The multimodal densities, after being sorted by a special distance heuristic.
    """
    # finding the maximum multimodal density
    visited_indices, cluster = [], set()
    index = results.densities.max(
        dim=0
    ).indices.item()  # the index to the largest multimodal density
    visited_indices.append(index)  # keep track that the above index is/has been used
    cluster.add(
        results.uniques[index]
    )  # add the observation to the cluster that is being developed
    results.distances.fill_diagonal_(
        float("inf")
    )  # get rid of the entries that are equal to zero

    while len(visited_indices) < results.uniques.shape[0]:
        # get the distances from the current (index) observation
        temp = results.distances[index]
        # ignore observations' distances already used
        temp[torch.LongTensor(visited_indices)] = float("inf")
        # get the observation index with the minimum distance to the current observation
        _, index = temp.min(dim=0)
        # keep track of the above index (order matters)
        visited_indices.append(index.item())
        # add the observation to the cluster that is being developed
        cluster.add(results.uniques[index.item()])

    # sort the multimodal densities using the above indices
    return results.densities[visited_indices]


def select_prototypes(results, local_maxima):
    """
    Selects the prototypes that are identified in the first iteration of the
    Empirical Fuzzy Sets algorithm by only considering the local maximum criteria.

    Args:
        results: A namedtuple, MultimodalDensity, with attributes:
            uniques, frequencies, distances, densities.
        local_maxima: The multimodal densities, after being sorted by a special distance heuristic.

    Returns:
        The prototypes that are identified in the first iteration
        of the Empirical Fuzzy Sets algorithm.
    """
    peak_mask = torch.cat(
        [
            torch.zeros(1, dtype=torch.bool, device=local_maxima.device),
            (local_maxima[:-2] < local_maxima[1:-1])
            & (local_maxima[2:] < local_maxima[1:-1]),
            torch.zeros(1, dtype=torch.bool, device=local_maxima.device),
        ],
        dim=0,
    )
    peak_mask[0] = local_maxima[0] > local_maxima[1]
    peak_mask[-1] = local_maxima[-1] > local_maxima[-2]
    return results.uniques[peak_mask]


def reduce_partitioning(results, prototypes) -> Cluster:
    """
    Reduce the number of identified prototypes by only keeping those that
    strongly envelop their nearby prototypes.

    Args:
        results: A namedtuple, MultimodalDensity, with attributes:
            uniques, frequencies, distances, densities.
        prototypes: The prototypes that are identified in the first iteration
            of the Empirical Fuzzy Sets algorithm.

    Returns:
        The final partitions, fuzzy.sets.Base.
    """
    continue_search = True
    while continue_search:
        distances = torch.cdist(results.uniques, prototypes)
        cloud_labels = distances.min(dim=1).indices
        cloud_centers, cloud_widths = [], []

        # find the centers of the data clouds and their widths
        for label in cloud_labels.unique():
            cloud_data = results.uniques[cloud_labels == label]
            cloud_centers.append(cloud_data.mean(dim=0).cpu().detach().numpy())
            cloud_widths.append(
                cloud_data.std(dim=0).nan_to_num(0.0).cpu().detach().numpy()
            )

        cloud_centers, cloud_widths = torch.tensor(
            np.array(cloud_centers), device=prototypes.device
        ), torch.tensor(np.array(cloud_widths), device=prototypes.device)
        cloud_distances = torch.cdist(prototypes, prototypes)
        distance_threshold = cloud_distances.std() * (
            1 - cloud_distances.std() / cloud_distances.mean()
        )
        prev_prototypes, prototypes, widths = find_data_clouds(
            cloud_centers, cloud_distances, cloud_widths, distance_threshold, prototypes
        )

        # continue the search until the partitions no longer change
        if prototypes.shape == prev_prototypes.shape:
            continue_search = False

    return Cluster(
        centers=prototypes.cpu().detach().numpy(),
        widths=widths.cpu().detach().numpy(),
        device=prototypes.device,
    )


def find_data_clouds(
    cloud_centers, cloud_distances, cloud_widths, distance_threshold, prototypes
):
    """
    Note: This code needs to be analyzed and its purpose should be better documented.

    Args:
        cloud_centers:
        cloud_distances:
        cloud_widths:
        distance_threshold:
        prototypes:

    Returns:

    """
    cloud_results: MultimodalDensity = multimodal_density(cloud_centers)
    prev_prototypes: torch.Tensor = prototypes
    next_cloud_centers, next_cloud_widths = [], []
    for cloud_index in range(cloud_centers.shape[0]):
        # find any nearby clouds to this current cloud (cloud_index)
        nearby_partitions = cloud_results.densities[
            cloud_distances[cloud_index] < distance_threshold
        ]
        if cloud_results.densities[cloud_index] == nearby_partitions.max():
            # if this current cloud (cloud_index) has the largest density, keep it, else toss it
            next_cloud_centers.append(cloud_centers[cloud_index].cpu().detach().numpy())
            next_cloud_widths.append(cloud_widths[cloud_index].cpu().detach().numpy())
    prototypes = torch.tensor(np.array(next_cloud_centers), device=prototypes.device)
    widths = torch.tensor(np.array(next_cloud_widths), device=prototypes.device)
    return prev_prototypes, prototypes, widths


def find_empirical_fuzzy_sets(input_data: torch.Tensor):
    """
    Discover empirical fuzzy sets.

    Args:
        input_data: Multivariate input data, PyTorch tensor.

    Returns:
        The final partitions (Empirical Fuzzy Sets), fuzzy.sets.Base.
    """
    results: MultimodalDensity = multimodal_density(input_data)
    local_maxima = find_local_maxima(results)
    prototypes = select_prototypes(results, local_maxima)
    return reduce_partitioning(results, prototypes)
