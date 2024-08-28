"""
This module contains helpful features used throughout the fuzzy-ml package.
"""

from collections import namedtuple

LabeledClusters = namedtuple(
    typename="LabeledClusters", field_names=("clusters", "labels", "supports")
)
