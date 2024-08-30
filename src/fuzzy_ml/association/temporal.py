"""
Implements the Fuzzy Temporal Association Rule Mining algorithm
and its necessary helper functions.
"""

import itertools
import collections
from typing import Tuple, Union, List

import torch
import numpy as np
import pandas as pd
import igraph as ig
from fuzzy.sets import Membership
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.relations.t_norm import Minimum
from regime import hyperparameter, Node
from matplotlib import pyplot as plt


AssociationRule = collections.namedtuple(
    "AssociationRule", ["antecedents", "consequents", "confidence"]
)


class TemporalInformationTable:
    """
    The TemporalInformationTable provides a convenient interface for
    handling the transactions' temporal information.
    """

    def __init__(self, dataframe, variables):
        self.dataframe, self.variables = dataframe, variables

        # find the earliest starting period for each temporal item in the dataframe
        self.first_transaction_indices = (
            self.dataframe[self.variables].values != 0
        ).argmax(axis=0)
        self.size_of_transactions_per_time_granule = self.dataframe.groupby(
            "date"
        ).size()
        self.transactions_per_time_granule = [
            transactions_df for time, transactions_df in self.dataframe.groupby("date")
        ]
        self.starting_periods = {}
        for idx, transactions_df in enumerate(self.transactions_per_time_granule):
            transactions_df = transactions_df.loc[:, (transactions_df != 0).any(axis=0)]
            self.transactions_per_time_granule[idx] = transactions_df
            df_columns = list(set(self.variables).intersection(transactions_df.columns))
            temp_dict = dict(zip(df_columns, [idx] * len(df_columns)))
            self.starting_periods = {
                key: [
                    min(
                        temp_dict.get(key, float("inf")),
                        self.starting_periods.get(key, [float("inf")])[0],
                    )
                ]
                for key in set(temp_dict) | set(self.starting_periods)
            }
        # sort by variable ordering
        self.starting_periods = pd.DataFrame(self.starting_periods)[self.variables]

    def relevant_transactions(
        self, time_column: str = "date", starting_period: int = None
    ) -> pd.DataFrame:
        """
        Obtain the transactions from the database that have a time_column value either at
        or after starting_period.

        Args:
            time_column: The column that should be used as timestamps for the transactions;
            default is 'date'.
            starting_period: The timestamp that all transactions should either occur at
            or after it; default is 'None'.

        Returns:
            (pandas.DataFrame) A dataframe that contains all the relevant transactions.
        """
        if starting_period is not None:
            dataframes = [
                transactions_df
                for time, transactions_df in self.dataframe.groupby(time_column)
            ]
            transactions_dataframe = pd.concat(dataframes[starting_period:])
        else:
            transactions_dataframe = self.dataframe
        return transactions_dataframe

    def max_starting_periods(self, candidates: list) -> np.array:
        """
        The maximum starting periods of the given candidates.

        Args:
            candidates: The candidate itemsets.

        Returns:
            Numpy array.
        """
        # we need to get each temporal item's corresponding starting period
        item_indices_in_each_candidate = [
            tuple(pair[0] for pair in candidate) for candidate in candidates
        ]
        starting_periods_per_item_in_each_candidate = [
            [self.starting_periods.values[0, var_idx] for var_idx in candidate_indices]
            for candidate_indices in item_indices_in_each_candidate
        ]
        # get the maximum starting period within each candidate to calculate fuzzy temporal support
        return np.array(starting_periods_per_item_in_each_candidate).max(axis=1)


def find_predecessors(predecessors_vertices: ig.VertexSeq) -> list:
    """
    Given the vertices referencing an itemset's predecessors,
    retrieve the values they are actually representing.

    Args:
        predecessors_vertices: VertexSeq object.

    Returns:
        A list of predecessors.
    """
    # there must be more than 1 predecessor to produce an association rule
    max_len_of_itemsets = list(predecessors_vertices["lattice"])[0]
    if max_len_of_itemsets == 1:
        predecessors = [
            [tuple(predecessor["item"])] for predecessor in predecessors_vertices
        ]
    else:
        predecessors = [
            tuple(predecessor["item"]) for predecessor in predecessors_vertices
        ]
    return predecessors


class FuzzyTemporalAssocationRuleMining(
    Node
):  # inherit Node to use the hyperparameter decorator
    """
    Implements the fuzzy temporal association rule mining (FTARM) algorithm.
    """

    @hyperparameter("min_support")
    def __init__(
        self,
        dataframe: pd.DataFrame,
        knowledge_base: KnowledgeBase,
        min_support: float,
        device: torch.device,
    ):
        """
        Initializes an object that performs the
        Fuzzy Temporal Association Rule Mining (FTARM) algorithm.

        Args:
            dataframe: A DataFrame that has two columns: 'date' and 'items'. The column 'date'
            can be any implementation of date or time, so long as it represents a consistent
            granulation of time (e.g., seconds, minutes, days, weeks). The column 'items' expects
            the values within it to be sets of 2-tuples, where each 2-tuple is a temporal item in
            the form of: (item, item's quantity). Again, 'item' can be any object - or at least,
            is intended to be.
            knowledge_base: An instance object of KnowledgeBase.
            min_support: The minimum support to use when determining the frequent itemsets.
            device: The device to use.
        """

        super().__init__(resource_name="None")
        self.dataframe, self.knowledge_base = dataframe, knowledge_base
        self.granulation = self.knowledge_base.select_by_tags(
            tags={"premise", "group"}
        )[0]["item"]
        self.variables = [col for col in self.dataframe.columns if col != "date"]
        self.ti_table = TemporalInformationTable(self.dataframe, self.variables)
        self.min_support: float = min_support
        self.device = device

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"The __call__ method is not implemented for {self.__class__} "
            f"as it has many potential methods of interest."
        )

    def fuzzy_representation(
        self, candidates: list = None, starting_period: int = None
    ) -> Membership:
        """
        Generate the 'fuzzy representations' for each transaction that is currently relevant
        (i.e., >= starting_period).

        Args:
            candidates: The candidate itemsets.
            starting_period: The starting period to begin counting transactions from.

        Returns:
            The fuzzy representations of each transaction in the database that occurs at or
            after the given starting_period.
        """
        dataframe = self.ti_table.relevant_transactions(starting_period=starting_period)

        if candidates is None:
            # step 1: simply calculate the membership of each fuzzy set individually
            membership: Membership = self.granulation(
                torch.tensor(
                    dataframe[self.variables].values, device=self.device
                ).float()
            )
            return membership.degrees * membership.mask
        # step 8.1: membership of some specific combination of fuzzy sets, resolved by a t-norm
        engine = Minimum(
            *[list(candidate) for candidate in candidates], device=self.device
        )
        antecedents_memberships = self.granulation(
            torch.tensor(dataframe[self.variables].values, device=self.device).float()
        )
        return engine(antecedents_memberships).degrees

    def scalar_cardinality(
        self, candidates: list = None, starting_period: int = None
    ) -> torch.Tensor:
        """
        The scalar cardinality, or 'count' is the summation of the membership
        degrees across the transactions. Here, dim=0 refers to the dimension that separates the
        transactions from one another
        (i.e., num of transactions, num of items, num of linguistic terms).

        Args:
            candidates: The candidate itemsets.
            starting_period: The starting period to begin counting transactions from.

        Returns:
            (torch.Tensor) The scalar cardinalities.
        """
        return self.fuzzy_representation(candidates, starting_period).sum(dim=0)

    def fuzzy_temporal_supports(
        self, candidates: list = None, starting_period: int = None
    ) -> torch.Tensor:
        """
        Calculate the fuzzy temporal supports of the candidates, given the possible
        restriction of starting_period (None means no restriction
        aka the earliest transaction is fine).

        Args:
            candidates: The candidate itemsets.
            starting_period: The starting period to begin counting transactions from.

        Returns:
            PyTorch tensor.
        """
        if candidates is None:  # 1-itemsets
            # step 3
            if starting_period is None:
                starting_periods = self.ti_table.starting_periods.values[0]
            else:
                number_of_temporal_items = self.ti_table.starting_periods.shape[1]
                starting_periods = np.array(
                    [starting_period] * number_of_temporal_items
                )
            num_of_possible_transactions_per_temporal_item = [
                self.ti_table.size_of_transactions_per_time_granule.values[idx:].sum()
                for idx in starting_periods
            ]
            denominator = torch.tensor(
                num_of_possible_transactions_per_temporal_item, device=self.device
            )[:, None]
        else:  # r-itemsets
            # step 8.3
            if starting_period is None:
                # max starting period within each candidate to calculate fuzzy temporal support
                max_starting_periods = self.ti_table.max_starting_periods(candidates)
            else:
                # given a max starting period to look on from; used by association rule mining
                max_starting_periods = np.array(
                    [starting_period] * len(candidates)
                ).flatten()
            num_of_transactions_per_candidate = [
                self.ti_table.size_of_transactions_per_time_granule.values[idx:].sum()
                for idx in max_starting_periods
            ]
            num_of_transactions_per_candidate = np.array(
                num_of_transactions_per_candidate
            )
            denominator = torch.tensor(
                num_of_transactions_per_candidate, device=self.device
            )

        scalar_cardinality = self.scalar_cardinality(
            candidates, starting_period=starting_period
        )
        if scalar_cardinality.is_sparse:
            scalar_cardinality = scalar_cardinality.to_dense()

        return scalar_cardinality / denominator

    def frequent_temporal_items(
        self, supports: torch.Tensor
    ) -> Tuple[list, torch.Tensor]:
        """
        Keep only the supports of the frequent 1-itemsets.
        """
        # returns a tuple of torch.Tensor, (row_indices, col_indices)
        row_col_indices_to_keep: Tuple[torch.Tensor, torch.Tensor] = torch.where(
            supports > torch.tensor(self.min_support, device=self.device)
        )
        # inner_supports is a torch.Tensor that has shape (num of temporal items, num of terms)
        supports = supports[row_col_indices_to_keep]
        items = list(
            zip(
                row_col_indices_to_keep[0].cpu().detach().tolist(),
                row_col_indices_to_keep[1].cpu().detach().tolist(),
            )
        )
        return items, supports

    def make_candidates(self, candidates: list = None) -> Union[list, None]:
        """
        Make the candidate itemsets.

        Args:
            candidates: Any existing candidate itemsets.

        Returns:
            The generated candidates.
        """
        supports = self.fuzzy_temporal_supports(candidates)
        if (
            torch.count_nonzero(
                supports >= torch.tensor(self.min_support, device=self.device)
            ).item()
            > 0
        ):  # step 5
            if candidates is None:
                return self.generate_candidates_of_length_two(supports)
            if (
                len(candidates) == 1
            ):  # only one candidate left, no more can be generated
                return None
            max_len_of_itemsets = len(candidates[0])
            frequent_itemset_indices = torch.where(
                supports >= torch.tensor(self.min_support, device=self.device)
            )[
                0
            ]  # r-itemsets
            frequent_itemsets = [
                set(candidates[idx]) for idx in frequent_itemset_indices
            ]
            candidate_indices = self.generate_superset_itemsets(
                frequent_itemsets, max_len_of_itemsets
            )

            if len(candidate_indices) == 0:
                return None  # although there were candidates,
                # no more could be generated from those
            temp_combinations = [set(candidate) for candidate in candidate_indices]
            # keep only the itemsets that do not repeat a variable
            # (e.g., each variable has only 1 term)
            mask = [
                len({item[0] for item in itemset}) == max_len_of_itemsets + 1
                for itemset in temp_combinations
            ]
            return [
                itemset for idx, itemset in enumerate(temp_combinations) if mask[idx]
            ]
        return None

    def generate_superset_itemsets(
        self, frequent_itemsets: list, max_len_of_itemsets: int
    ) -> set:
        """
        Generate the superset itemsets of the frequent itemsets.

        Args:
            frequent_itemsets: The frequent itemsets.
            max_len_of_itemsets: The maximum length of the itemsets.

        Returns:
            The superset itemsets.
        """
        candidate_indices = set()
        for itemset_1, itemset_2 in itertools.combinations(frequent_itemsets, 2):
            candidate = frozenset(itemset_1.union(itemset_2))
            # apriori principle
            available_subsets = {frozenset(items) for items in frequent_itemsets}
            required_subsets = {
                frozenset(items) for items in itertools.combinations(candidate, 2)
            }
            # the generated candidate must be size (r + 1) and satisfy apriori principle
            if len(candidate) == max_len_of_itemsets + 1 and len(
                required_subsets.intersection(available_subsets)
            ) == len(required_subsets):
                candidate_indices.add(candidate)
                target_vertex = self.knowledge_base.graph.add_vertex(
                    item=tuple(candidate),
                    frequent=True,
                    lattice=(max_len_of_itemsets + 1),
                )

                possible_subsets_vertices = self.knowledge_base.graph.vs.select(
                    lattice_eq=max_len_of_itemsets
                )
                edges_to_add = set()
                for possible_subset_vertex in possible_subsets_vertices:
                    if set(possible_subset_vertex["item"]).issubset(candidate):
                        edges_to_add.add((possible_subset_vertex, target_vertex))

                # add the edges between the frequent items and the new candidates
                self.knowledge_base.graph.add_edges(
                    edges_to_add, attributes={"lattice": True}
                )
        return candidate_indices

    def generate_candidates_of_length_two(self, supports: torch.Tensor) -> list:
        """
        Generate the candidate itemsets from the given fuzzy temporal supports of the itemsets.

        Args:
            supports: The fuzzy temporal supports of the itemsets.

        Returns:
            The generated candidates.
        """
        max_len_of_itemsets = 1
        frequent_items, _ = self.frequent_temporal_items(
            supports
        )  # iterable of (item idx, term idx)
        # add the new candidates to the KnowledgeBase as vertices
        self.knowledge_base.graph.add_vertices(
            len(frequent_items),
            attributes={
                "item": frequent_items,
                "frequent": True,
                "lattice": max_len_of_itemsets,  # length of frequent set
            },
        )
        frequent_item_vertices: ig.VertexSeq = self.knowledge_base.graph.vs.select(
            item_in=frequent_items
        )
        if len(frequent_item_vertices) == 0:
            return []  # no frequent items
        temp_combinations = list(
            itertools.combinations(frequent_items, r=max_len_of_itemsets + 1)
        )
        # keep only the itemsets that do not repeat a variable
        # (e.g., each variable has only 1 term)
        mask = [
            len({item[0] for item in items}) == max_len_of_itemsets + 1
            for items in temp_combinations
        ]
        new_candidates = [
            items for idx, items in enumerate(temp_combinations) if mask[idx]
        ]
        supports = self.fuzzy_temporal_supports(new_candidates)
        # add the new candidates to the KnowledgeBase as vertices
        self.knowledge_base.graph.add_vertices(
            len(new_candidates),
            attributes={
                "item": list(new_candidates),
                "frequent": list(
                    (supports > torch.tensor(self.min_support, device=self.device))
                    .cpu()
                    .detach()
                    .numpy()
                ),
                "lattice": max_len_of_itemsets + 1,  # length of frequent set
            },
        )
        # collect the edges between the frequent items and the new candidates
        edges_to_add = set()
        frequent_set_vertices = self.knowledge_base.graph.vs.select(
            lattice=max_len_of_itemsets + 1
        )
        for frequent_item_vertex in frequent_item_vertices:
            for (
                frequent_set_vertex
            ) in frequent_set_vertices:  # new & frequent supersets
                if frozenset({frequent_item_vertex["item"]}).issubset(
                    frequent_set_vertex["item"]
                ):
                    edges_to_add.add((frequent_item_vertex, frequent_set_vertex))
        # add the edges between the frequent items and the new candidates
        self.knowledge_base.graph.add_edges(edges_to_add, attributes={"lattice": True})
        return new_candidates

    def find_candidates(self, candidates: list = None) -> list:
        """
        Executes the FTARM algorithm to produce the frequent temporal itemsets.

        Args:
            candidates: A list of candidates to start the FTARM algorithm from; default is 'None'.

        Returns:
            (nested list) A list of frequent temporal itemsets where the 0'th candidates family
            item are the 2-itemsets, the 1'th candidates family item are the 3-itemsets, etc.
        """
        candidates_family = []
        while True:
            candidates = self.make_candidates(candidates)
            if candidates is None:
                break
            candidates_family.append(candidates)
        return candidates_family

    def find_closed_itemsets(self) -> List[frozenset]:
        """
        Must be run after .find_candidates(), as it searches the saved
        lattice structure to produce the fuzzy temporal association rules.

        Returns:
            A set of closed itemsets, where each element is a closed itemset.
            An itemset is closed if none of its immediate supersets has the
            same support as the itemset.
        """
        closed_itemsets: List[frozenset] = []
        frequent_sets_vertices = self.knowledge_base.graph.vs.select(
            frequent_eq=True
        )  # get everything that is frequent
        for frequent_set_vertex in frequent_sets_vertices:
            try:  # 1-itemsets
                frequent_set_vertex_support: float = self.fuzzy_temporal_supports(
                    [[frequent_set_vertex["item"]]]
                ).item()
            except TypeError:  # r-itemsets for r > 1
                frequent_set_vertex_support: float = self.fuzzy_temporal_supports(
                    [frequent_set_vertex["item"]]
                ).item()
            is_closed = True
            # iterate over the successors of the frequent set
            for successor_vertex in frequent_set_vertex.successors():
                try:
                    target_support: float = self.fuzzy_temporal_supports(
                        [successor_vertex["item"]]
                    ).item()
                except TypeError:  # non-existent edge error handling
                    continue
                if frequent_set_vertex_support == target_support:
                    is_closed = False  # there exists superset w/ equal support,
                    # thus frequent_set_vertex is not closed
                    break
            if (
                is_closed
                and frozenset(frequent_set_vertex["item"]) not in closed_itemsets
            ):
                closed_itemsets.append(frozenset(frequent_set_vertex["item"]))
        return closed_itemsets

    def find_maximal_itemsets(self) -> set:
        """
        Must be run after .find_candidates(), as it searches the saved
        lattice structure to produce the fuzzy temporal association rules.

        Returns:
            A set of maximal itemsets, where each element is a maximal itemset.
            An itemset is maximal if none of its immediate supersets are frequent.
        """
        maximal_itemsets = set()
        frequent_sets_vertices = self.knowledge_base.graph.vs.select(
            frequent_eq=True
        )  # get everything that is frequent
        for frequent_set_vertex in frequent_sets_vertices:
            is_maximal = True
            # iterate over the successors of the frequent set
            for successor_vertex in frequent_set_vertex.successors():
                if successor_vertex["frequent"]:
                    is_maximal = False  # there exists a frequent superset,
                    # thus frequent_set_vertex is not maximal
            if is_maximal:
                maximal_itemsets.add(frozenset(frequent_set_vertex["item"]))
        return maximal_itemsets

    @hyperparameter("min_confidence")
    def find_association_rules(self, min_confidence: float) -> List[AssociationRule]:
        """
        Must be run after .find_candidates(), as it searches the saved
        lattice structure to produce the fuzzy temporal association rules.

        Args:
            min_confidence: The minimum confidence threshold.

        Returns:
            A set of association rules, where each element is an association rule.
        """
        rules, rules_already_made = [], {}
        frequent_sets_vertices = self.knowledge_base.graph.vs.select(
            frequent_eq=True
        )  # get everything that is frequent
        for frequent_set_vertex in frequent_sets_vertices:
            predecessors_indices = self.knowledge_base.graph.predecessors(
                frequent_set_vertex
            )  # the ids of the predecessors
            predecessors_vertices = self.knowledge_base.graph.vs[predecessors_indices]
            if len(predecessors_vertices) > 1:
                predecessors = find_predecessors(predecessors_vertices)
                node = frequent_set_vertex["item"]
                antecedents = self.find_antecedents(node, predecessors, min_confidence)
                for antecedent, confidence in antecedents:
                    rule = None
                    consequent = frozenset(node) - antecedent
                    if antecedent in rules_already_made:
                        if consequent not in rules_already_made[antecedent]:
                            rule = AssociationRule(
                                antecedent, consequent, confidence.item()
                            )
                            rules_already_made[antecedent].add(consequent)
                    else:
                        rule = AssociationRule(
                            antecedent, consequent, confidence.item()
                        )
                        rules_already_made[antecedent] = {consequent}
                    if rule is not None:
                        rules.append(rule)
        return rules

    @hyperparameter("min_confidence")
    def find_antecedents(
        self, node: tuple, predecessors: list, min_confidence: float
    ) -> list:
        """
        Find the antecedents' confidence and return only those with confidences
        greater than the minimum confidence threshold.
        Args:
            node: A frequent itemset.
            predecessors: The predecessors to that frequent itemset.
            min_confidence: The minimum confidence threshold.

        Returns:
            A list of the antecedents, where each element is a 2-tuple of an antecedent
            with its corresponding confidence.
        """
        confidences = self.calc_confidence(node, predecessors)
        boolean_mask = confidences > min_confidence
        antecedents = [
            (frozenset(antecedent), confidences[idx])
            for idx, antecedent in enumerate(predecessors)
            if boolean_mask[idx]  # rule's confidence is >
            # self.config.fuzzy_ml.association.temporal.min_confidence
        ]
        return antecedents

    def calc_confidence(self, node: tuple, predecessors: list) -> torch.Tensor:
        """
        Calculate the fuzzy temporal confidence.

        Args:
            node: A node, representing an itemset(s) of interest.
            predecessors: The predecessors of the itemset(s) (node).

        Returns:
            Fuzzy temporal confidences.
        """
        overall_support = self.fuzzy_temporal_supports({node})
        starting_period = self.ti_table.max_starting_periods({node})[0]
        antecedent_support = self.fuzzy_temporal_supports(predecessors, starting_period)
        return overall_support / antecedent_support

    def visualize_lattice(self, layout: str = "grid") -> None:
        """
        Visualize and display the constructed lattice.
        Args:
            layout: The chosen layout style; default is 'grid'.

        Returns:
            None
        """
        _, axs = plt.subplots()

        subgraph = self.knowledge_base.graph.subgraph(
            self.knowledge_base.graph.vs.select(frequent_eq=True)
        )
        ig.plot(
            obj=subgraph,
            mark_groups=True,
            palette=ig.RainbowPalette(),
            layout=layout,
            edge_width=0.5,
            target=axs,
            opacity=0.7,
            vertex_size=(np.array(self.knowledge_base.graph.authority_score()) / 3)
            + 0.1,
            edge_size=self.knowledge_base.graph.es["weight"],
        )
        plt.axis("off")
        plt.show()
