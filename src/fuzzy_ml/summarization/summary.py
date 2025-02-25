"""
Implements the linguistic summary code and its necessary supporting functions.
"""

from typing import Callable, Union

import torch
from fuzzy.relations.t_norm import TNorm
from fuzzy.logic.knowledge_base import KnowledgeBase

from fuzzy_ml.summarization.query import Query


class Summary:
    """
    The linguistic summary class; provides methods to create and evaluate
    linguistic summaries of data.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        quantifier: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        weights=None,
    ):
        self.device = device
        self.knowledge_base = knowledge_base
        # a family of fuzzy sets that describe a concept (e.g., young) in their dimension
        self.granulation = self.knowledge_base.select_by_tags(
            tags={"premise", "group"}
        )["item"][0]
        self.engine: TNorm = self.knowledge_base.rule_base.premises
        self.quantifier = (
            quantifier  # a fuzzy set describing a quantity in agreement (e.g., most)
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

    def summarizer_membership(
        self, input_data: torch.Tensor, query: Union[None, Query] = None
    ):
        """
        Calculate how well the summarizer in this class
        describes the given data observation, given an optional query
        that constrains the degree.

        Args:
            input_data: Data observations.
            query: A given query that needs to be adhered to, such as the attribute
            age must be 'young'.

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

        query_degrees = query(input_data).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()

        return torch.minimum(
            summary_applicability,
            query_degrees.flatten(),
        )

    def overall_summary_membership(
        self, input_data: torch.Tensor, query: Union[None, Query] = None
    ):
        """
        Intermediate function in determining how well
        the summary describes the overall data.

        Args:
            input_data: A collection of data observations.
            query: A given query that needs to be adhered to,
            such as the attribute age must be 'young'.

        Returns:
            (torch.Tensor) How much the overall data
            satisfied the object's summarizer.
        """
        query_degrees = query(input_data).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()

        return (
            self.summarizer_membership(input_data, query).nansum()
            / query_degrees.flatten().nansum()
        )

    def degree_of_truth(
        self, input_data: torch.Tensor, query: Union[None, Query] = None
    ) -> torch.Tensor:
        """
        Calculate the degree of truth given the quantifier.

        Args:
            input_data: A collection of data observations.
            query: A given query that needs to be adhered to, such as the attribute
            age must be 'young'.

        Returns:
            The truth of the linguistic quantified proposition.
        """
        return self.quantifier(self.overall_summary_membership(input_data, query=query))

    def degree_of_imprecision(
        self, input_data: torch.Tensor, alpha: float = 0.0
    ) -> torch.Tensor:
        """
        Calculate the degree of imprecision of the linguistic summary.

        Args:
            input_data: A collection of data observations.
            alpha: The minimum degree of membership required in both the summary and query.

        Returns:
            The degree of imprecision of the linguistic summary.
        """
        # get the number of attributes involved in the summarizer
        n_attributes: int = input_data.shape[-1]

        # individual_memberships shape:
        # (number_of_observations, number_of_attributes, number_of_rules)
        values = self.calc_dim_cardinality(input_data, alpha=alpha)

        # 1 - the m'th root of the product of the above
        # https://stackoverflow.com/questions/19255120/is-there-a-short-hand-for-nth-root-of-x-in-python
        return torch.ones(1, device=self.device) - torch.prod(values, dim=0) ** (
            1 / n_attributes
        )

    def calc_dim_cardinality(
        self,
        input_data: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Calculate the dimensions' cardinality; used for degree of appropriateness and degree of
        imprecision.

        Args:
            input_data: A collection of data observations.
            alpha: The minimum degree of membership required in both the summary and query.

        Returns:
            The cardinality of the dimensions.
        """
        # individual_memberships shape:
        # (number_of_observations, number_of_attributes, number_of_rules)
        individual_memberships = self.engine.apply_mask(self.granulation(input_data))
        return (
            torch.count_nonzero(individual_memberships > alpha, dim=0).squeeze()
            / input_data.shape[0]
        )

    def degree_of_covering(
        self,
        input_data: torch.Tensor,
        alpha: float = 0.0,
        query: Union[None, Query] = None,
    ) -> torch.Tensor:
        """
        How many objects in the data set (database) corresponding to the query are 'covered'
        by the particular summary (i.e., self.summarizer). Its interpretation is simple.
        For example, if it is equal to 0.15, then this means that 15% of the objects are
        consistent with the summary in question.

        Args:
            input_data (torch.Tensor): A collection of data observations.
            alpha: The minimum degree of membership required
            in both the summary and query.
            query (namedtuple): A given query that needs to be adhered to,
            such as the attribute age must be 'young'.

        Returns:
            A ratio between 0 and 1 that describes how many objects are covered by this summary.
        """
        numerator = torch.count_nonzero(
            self.summarizer_membership(input_data, query) > alpha
        )
        query_degrees = query(input_data).degrees
        if query_degrees.is_sparse:
            query_degrees = query_degrees.to_dense()
        denominator = torch.count_nonzero(query_degrees > alpha)
        return numerator / denominator

    def degree_of_appropriateness(
        self,
        input_data: torch.Tensor,
        alpha: float = 0.0,
        query: Union[None, Query] = None,
    ) -> torch.Tensor:
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
            input_data: A collection of data observations.
            alpha: The minimum degree of membership required in both the summary and query.
            query A given query that needs to be adhered to, such as the attribute
            age must be 'young'.

        Returns:
            A ratio between 0 and 1 that describes how interesting the relation
            described by the summary is.
        """
        covering = self.degree_of_covering(input_data, alpha, query)

        values = self.calc_dim_cardinality(input_data, alpha=alpha)

        # values = (torch.count_nonzero(values > alpha, dim=0) / input_data.shape[0]) - covering
        return torch.abs(torch.prod(values, dim=0) - covering)

    def length(self) -> torch.Tensor:
        """
        The length of a summary, which is relevant because a long summary is not easily
        comprehensible by the human user. This length may be defined in various ways,
        but the following has proven to be useful.

        Returns:
            A ratio between 0 and 1 that describes how short a summary is, where 1 means
            extremely short and 0 means extremely long.
        """
        summarizer_cardinality: int = torch.count_nonzero(self.engine.get_mask()).item()
        return 2 * (
            torch.pow(torch.tensor(0.5, device=self.device), summarizer_cardinality)
        )

    def degree_of_validity(
        self,
        input_data: torch.Tensor,
        alpha: float = 0.0,
        query: Union[None, Query] = None,
    ) -> torch.Tensor:
        """
        The total degree of validity for a particular linguistic summary
        is defined as the weighted average of the above five degrees of validity
        (e.g., degree_of_truth, degree_of_covering, degree_of_appropriateness, length).

        Args:
            input_data: A collection of data observations.
            alpha: The minimum degree of membership required in both the summary and query.
            query: A given query that needs to be adhered to, such as the attribute
            age must be 'young'.

        Returns:
            The total degree of validity for a particular linguistic summary.
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
