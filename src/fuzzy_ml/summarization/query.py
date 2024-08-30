"""
This module contains the Query class, which represents a query in the fuzzy-ml library. A query is
a fuzzy set that describes a property of the data, and the Query class provides a way to apply the
query to the data and determine the truth of the query based on how well the membership function
describes the selected attribute in the data.
"""

import torch
from fuzzy.sets import FuzzySet, Membership


class Query(torch.nn.Module):
    """
    A class to represent a query.

    Attributes
    ----------
    membership_function : FuzzySet
        The membership function of the query.
    attribute_index : int
        The index of the attribute in the dataset to match against the membership function.
    """

    def __init__(
        self, membership_func: FuzzySet, attribute_index: int, *args, **kwargs
    ):
        """
        Initialize the Query.

        Args:
            membership_func: The membership function of the query.
            attribute_index: The index of the attribute in the dataset to match against
            the membership function.
            *args: Additional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.membership_function: FuzzySet = membership_func
        self.attribute_index: int = attribute_index

    def forward(self, data: torch.Tensor) -> Membership:
        """
        Forward pass of the query.

        Args:
            data: Apply the query to this data, where we determine the truth of the query based
            on how well the membership function describes the selected attribute in the data.

        Returns:
            The truth of the query.
        """
        return self.membership_function(data[:, self.attribute_index])
