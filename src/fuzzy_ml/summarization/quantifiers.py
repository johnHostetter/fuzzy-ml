"""
This module contains classic and modern fuzzy quantifiers.
"""

import torch


def most_quantifier(element: torch.Tensor) -> torch.Tensor:
    """
    Zadeh's A Computational Approach to Fuzzy Quantifiers
    in Natural Languages (1983).

    Args:
        element: How much an element, y, satisfied some property
        (fuzzy set), F, across the entire data.

    Returns:
        The truth of the linguistically quantified proposition.
    """
    if element >= 0.8:
        return torch.ones(1, device=element.device)
    if 0.3 < element < 0.8:
        return 2 * element - 0.6
    return torch.zeros(1, device=element.device)  # e.g., element <= 0.3
