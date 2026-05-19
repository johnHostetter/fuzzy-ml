"""
The summarization module contains classes and functions for summarizing data.
"""

from .quantifiers import most_quantifier
from .query import Query
from .summary import Summary

__all__ = ["Query", "Summary", "most_quantifier"]
