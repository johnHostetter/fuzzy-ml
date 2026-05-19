"""
The partitioning module contains classes that partition data into fuzzy sets.
"""

from .clip import CategoricalLearningInducedPartitioning
from .equal import EqualPartitioning
from .meta import MetaPartitioner

__all__ = [
    "MetaPartitioner",
    "EqualPartitioning",
    "CategoricalLearningInducedPartitioning",
]
