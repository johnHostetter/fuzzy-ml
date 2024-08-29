"""
Implements the linguistic summary code and its necessary supporting functions.
"""

from collections import namedtuple

import torch
from fuzzy.logic.rulebase import RuleBase
from fuzzy.relations.continuous.t_norm import TNorm


Query = namedtuple("Query", "term index")


def most_quantifier(element):
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
    return torch.zeros(1, device=element.device)  # element <= 0.3


class Summary:
    """
    The linguistic summary class; provides methods to create and evaluate
    linguistic summaries of data.
    """

    def __init__(
        self,
        knowledge_base,
        quantifier,
        truth,
        device: torch.device,
        weights=None,
    ):
        self.device = device
        self.knowledge_base = knowledge_base
        # a family of fuzzy sets that describe a concept (e.g., young) in their dimension
        self.granulation = self.knowledge_base.select_by_tags(
            tags={"premise", "group"}
        )["item"][0]
        self.engine: TNorm = RuleBase(
            rules=self.knowledge_base.get_fuzzy_logic_rules(), device=self.device
        ).premises
        self.quantifier = (
            quantifier  # a fuzzy set describing a quantity in agreement (e.g., most)
        )
        self.truth = (
            truth  # function that describes the validity of the summary across the data
        )

        if weights is None:  # if the i'th item
            self.weights = torch.tensor(
                [1 / 5] * 5, device=self.device
            )  # weight vector for 5 degrees of validity
        elif weights.sum() == 1.0 and weights.nelements() != 5:
            raise AttributeError(
                "The provided weights vector to the Summary "
                "constructor must have exactly 5 elements."
            )
        else:
            self.weights = weights

    def summarizer_membership(self, input_data, query=None):
        """
        Calculate how well the summarizer in this class
        describes the given data observation, given an optional query
        that constrains the degree.

        Args:
            input_data (torch.Tensor): Data observations.
            query (namedtuple): A given query that needs
            to be adhered to, such as the attribute age must be 'young'.

        Returns:
            How well the summarizer in this class describes
            the given data observation, given an optional query
            that constrains the degree.
        """
        summary_applicability = self.engine(
            self.granulation(input_data)
        ).degrees.flatten()
        if query is None:
            return summary_applicability

        query_degrees = query.term(input_data[:, query.index]).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()

        return torch.minimum(
            summary_applicability,
            query_degrees.flatten(),
        )

    def overall_summary_membership(self, input_data, query=None):
        """
        Intermediate function in determining how well
        the summary describes the overall data.

        Args:
            input_data (torch.Tensor): A collection of data observations.
            query (namedtuple): A given query that needs to be adhered to,
            such as the attribute age must be 'young'.

        Returns:
            (torch.Tensor) How much the overall data
            satisfied the object's summarizer.
        """
        query_degrees = query.term(input_data[:, query.index]).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()

        return (
            self.summarizer_membership(input_data, query).nansum()
            / query_degrees.flatten().nansum()
        )

    def degree_of_truth(self, input_data, query=None):
        """
        Calculate the degree of truth given the quantifier.

        Args:
            input_data:
            query:

        Returns:

        """
        return self.quantifier(self.overall_summary_membership(input_data, query=query))

    def degree_of_imprecision(self, input_data, alpha=0.0):
        """
        Calculate the degree of imprecision of the linguistic summary.

        Args:
            input_data:
            alpha:

        Returns:

        """
        number_of_observations = input_data.shape[0]
        number_of_attributes: int = self.engine._original_shape[0][
            0
        ]  # get the # of vars

        # individual_memberships shape:
        # (number_of_observations, number_of_attributes, number_of_rules)
        values = self.calculate_dimension_cardinality(
            input_data, number_of_attributes, number_of_observations, alpha
        )

        # 1 - the m'th root of the product of the above
        # https://stackoverflow.com/questions/19255120/is-there-a-short-hand-for-nth-root-of-x-in-python
        return torch.ones(1, device=self.device) - torch.prod(values, dim=0) ** (
            1 / number_of_attributes
        )

    def calculate_dimension_cardinality(
        self, input_data, number_of_attributes, number_of_observations, alpha
    ):
        """
        Calculate the dimensions' cardinality; used for degree of appropriateness and degree of
        imprecision.

        Args:
            input_data:
            number_of_attributes:
            number_of_observations:
            alpha:

        Returns:

        """
        # individual_memberships shape:
        # (number_of_observations, number_of_attributes, number_of_rules)
        individual_memberships = self.engine.apply_mask(self.granulation(input_data))
        values = []
        for j in range(number_of_attributes):
            dimension_memberships = individual_memberships[:, j, :]
            dimension_cardinality = torch.count_nonzero(dimension_memberships > alpha)
            values.append(dimension_cardinality / number_of_observations)
        return torch.tensor(values, device=self.device)

    def degree_of_covering(self, input_data, alpha=0.0, query=None):
        """
        How many objects in the data set (database) corresponding
        to the query are 'covered' by the particular summary (i.e., self.summarizer).
        Its interpretation is simple. For example, if it is equal to 0.15,
        then this means that 15% of the objects are consistent with the summary
        in question.

        Args:
            input_data (torch.Tensor): A collection of data observations.
            alpha: The minimum degree of membership required
            in both the summary and query.
            query (namedtuple): A given query that needs to be adhered to,
            such as the attribute age must be 'young'.

        Returns:
            (torch.Tensor) A ratio between 0 and 1 that describes
            how many objects are covered by this summary.
        """
        numerator = torch.count_nonzero(
            self.summarizer_membership(input_data, query) > alpha
        )
        query_degrees = query.term(input_data[:, query.index]).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()
        denominator = torch.count_nonzero(query_degrees > alpha)
        return numerator / denominator

    def degree_of_appropriateness(self, input_data, alpha=0.0, query=None):
        """
        Considered to be the most relevant degree of validity.

        For example, if a database contained employees and 50% of them are less than 25 years old
        and 50% are highly qualified, then we might expect that 25% of the employees would be less
        than 25 years old and highly qualified; this would correspond to a typical, fully
        expected situation. However, if the degree of appropriateness is, say, 0.39
        (i.e., 39% are less than 25 years old and highly qualified), then the summary found reflects
        an interesting, not fully expected relation in our data.

        This degree describes how characteristic for the particular database the summary found is.
        It is very important because, for instance, a trivial summary like '100% of sales is of any
        articles' has full validity (truth) if we use the traditional degree of truth but its degree
        of appropriateness is equal to 0 which is correct.


        Args:
            input_data (torch.Tensor): A collection of data observations.
            alpha: The minimum degree of membership required in both the summary and query.
            query (namedtuple): A given query that needs to be adhered to, such as the attribute
            age must be 'young'.

        Returns:
            (torch.Tensor) A ratio between 0 and 1 that describes how interesting the relation
            described by the summary is.
        """
        covering = self.degree_of_covering(input_data, alpha, query)
        number_of_observations = input_data.shape[0]
        number_of_attributes: int = self.engine._original_shape[0][
            0
        ]  # get the # of vars

        values = self.calculate_dimension_cardinality(
            input_data, number_of_attributes, number_of_observations, alpha
        )

        # values = (torch.count_nonzero(values > alpha, dim=0) / input_data.shape[0]) - covering
        return torch.abs(torch.prod(values, dim=0) - covering)

    def length(self):
        """
        The length of a summary, which is relevant because a long
        summary is not easily comprehensible by the
        human user. This length may be defined in various ways,
        but the following has proven to be useful.

        Returns:
            (torch.Tensor) A ratio between 0 and 1 that describes
            how short a summary is, where 1 means extremely short
            and 0 means extremely long.
        """
        cardinality_of_summarizer: int = self.engine._original_shape[0][
            0
        ]  # get the # of vars
        return 2 * (
            torch.pow(torch.tensor(0.5, device=self.device), cardinality_of_summarizer)
        )

    def degree_of_validity(self, input_data, alpha=0.0, query=None):
        """
        The total degree of validity for a particular linguistic summary
        is defined as the weighted average of the above five degrees of validity
        (e.g., degree_of_truth, degree_of_covering, degree_of_appropriateness, length).

        Args:
            input_data:
            alpha:
            query:

        Returns:

        """
        validity = torch.tensor(
            [
                self.degree_of_truth(input_data, query),
                self.degree_of_imprecision(input_data, alpha),
                self.degree_of_covering(input_data, alpha, query),
                self.degree_of_appropriateness(input_data, alpha, query),
                self.length(),
            ],
            device=self.device,
        )
        return (self.weights * validity).sum()
