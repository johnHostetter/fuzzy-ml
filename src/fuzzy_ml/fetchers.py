"""
Implements helpful 'wrapper' methods primarily used by soft.computing.organize.SelfOrganize.
"""

from fuzzy_ml import LabeledGaussian
from fuzzy_ml.datasets import LabeledDataset


def fetch_labeled_dataset(
    train_dataset: LabeledDataset,
    labeled_clusters: LabeledGaussian,
) -> LabeledDataset:
    """
    Conveniently returns the centers of a sequence of fuzzy sets. This is useful for some
    self-organizing neuro-fuzzy networks, such as those that use the CLIP and ECM algorithms.

    Args:
        train_dataset: Training dataset, which is not used, except to get output data - if needed.
        labeled_clusters: The labeled clusters, where each cluster is a FuzzySet
        and the label is a torch.Tensor type.

    Returns:
        A dataset that has fuzzy set centers as the inputs,
        with their corresponding labels as targets.
    """
    if labeled_clusters.labels is None:
        return LabeledDataset(
            data=labeled_clusters.get_centers(),
            out_features=train_dataset.out_features,
        )
    return LabeledDataset(
        data=labeled_clusters.get_centers(), labels=labeled_clusters.labels
    )
