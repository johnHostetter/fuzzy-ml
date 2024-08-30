"""
The partitioning module contains classes that partition data into fuzzy sets.
"""

from .meta import MetaPartitioner
from .equal import EqualPartitioning
from .clip import CategoricalLearningInducedPartitioning

__all__ = [
    "MetaPartitioner",
    "EqualPartitioning",
    "CategoricalLearningInducedPartitioning",
]
