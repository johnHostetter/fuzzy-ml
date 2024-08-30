"""
This Python script will be used to implement fuzzy logic rule reduction algorithms.
"""

import os
from typing import List, Type, Iterable, Union, Tuple

import rpy2
import torch
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import TNorm
from fuzzy.logic.rule import Rule

from fuzzy_ml.rpy2.packages import install_r_packages


def reduce_fuzzy_logic_rules_with_rough_sets(
    rules: List[Rule],
    t_norm: Type[TNorm],
    device: torch.device,
    output_values: Union[torch.Tensor, np.array] = None,
    verbose: bool = False,
) -> Tuple[List[Rule], torch.Tensor]:
    """
    Reduce fuzzy logic rules' attributes using rough set theory.

    Args:
        rules: A list of fuzzy logic rules.
        t_norm: The t-norm to use for the fuzzy logic rules.
        device: The device to use for the fuzzy logic rules.
        output_values: The output values of the rules. If None, the output values will be
            inferred from the rules' consequences. Defaults to None. For example, if the
            output_values are given, then the fuzzy logic rules are assumed to belong to a
            zero-order TSK fuzzy logic controller. If the output_values are None, then the
            fuzzy logic rules are assumed to belong to a Mamdani fuzzy logic controller and
            the output_values will be inferred from the rules' consequences.
        verbose: Whether to print debug information. Defaults to False.

    Returns:
        A list of reduced fuzzy logic rules and the output values of the newly reduced rules.
    """
    with conversion.localconverter(default_converter):
        # https://stackoverflow.com/questions/75069677/
        # rpy2-throws-a-notimplementederror-concerning-conversion-rules

        # the 'with' block handles a NotImplementedError:
        #     Conversion rules for `rpy2.robjects` appear to be missing. Those
        #     rules are in a Python contextvars.ContextVar. This could be caused
        #     by multithreading code not passing context to the thread.
        install_r_packages()
        _ = importr("RoughSets")

        unique_premise_variables = find_unique_premise_variables(rules)

        # convert the rules to a data frame
        numpy_matrix, output_values = make_rule_matrix(
            rules, unique_premise_variables, output_values
        )
        _, counts = np.unique(output_values.argmax(axis=-1), return_counts=True)

        if verbose:
            print("Diversity of output values: ", counts)

        if counts[0] == len(rules):  # all rules have the same output value
            return rules, torch.Tensor(output_values)  # no need to reduce the rules

        # make a DataFrame from the numpy matrix
        rules_df = pd.DataFrame(
            numpy_matrix,
            # [[int(element) for element in array] for array in numpy_matrix.tolist()],
            columns=list(map(str, unique_premise_variables)) + ["consequence"],
            # dtype=int
            # dtype="category"
        )

        # temporarily save the DataFrame to a CSV file
        rules_df.to_csv("tmp.csv", index=False)
        # read the CSV file into R as a data frame (this is a workaround since rpy2 interface
        # is not working properly with RoughSets due to their conversion rules)
        # more specifically, the conversion of the Pandas DataFrame to an R data frame
        # results in IntSexpVector instead of IntVector, which causes the SF.asDecisionTable()
        # function to fail; as a result, the RI.LEM2Rules.RST() function fails as well with the
        # following error:
        #     Error in RI.LEM2Rules.RST(decision.table) : nothing to tabulate
        robjects.r("flc_rules <- read.csv('tmp.csv')")
        robjects.r(
            "decision.table <- SF.asDecisionTable("
            "dataset = flc_rules, "
            "decision.attr = ncol(flc_rules), "
            "indx.nominal = rep(c(TRUE), ncol(flc_rules)))"
        )
        reduced_rules = robjects.r("RI.LEM2Rules.RST(decision.table)")
        # delete the temporary CSV file
        os.remove("tmp.csv")
        final_rules, final_output_values = parse_reduced_rules(
            numpy_matrix, output_values, reduced_rules, rules, t_norm, device=device
        )

    return final_rules, torch.tensor(final_output_values)


def find_unique_premise_variables(rules: Iterable[Rule]) -> List[int]:
    """
    Find the unique variables in the rules' premises.

    Args:
        rules: A list of fuzzy logic rules.

    Returns:
        A list of unique variables in the rules' premises.
    """
    # get the unique variable and value pairs in the rules' premises
    unique_pairs = set.union(*[set(rule.premise.indices[0]) for rule in rules])

    # keep only the variables
    variable_indices = list(
        {  # the set() removes duplicates from the list comprehension
            variable_idx for variable_idx, _ in unique_pairs
        }
        # the above list comprehension may yield, for example:
        # [0, 4, 1, 2, 3, 0, 1, 3, 1, 4, 0, 1, 0, 2, 3]
    )
    variable_indices.sort()  # sort the list of variable indices in ascending order
    return variable_indices


def make_rule_matrix(
    rules: List[Rule],
    unique_premise_variables: List[int],
    output_values: Union[torch.Tensor, np.array] = None,
) -> (np.ndarray, np.ndarray):
    """
    Create a matrix representation of the rules. The matrix is of shape (number of rules, number of
    unique premise variables + 1). The last column of the matrix contains the rules' consequences.

    Args:
        rules: A list of fuzzy logic rules.
        unique_premise_variables: A list of unique variables in the rules' premises.
        output_values: The output values of the rules. If None, the rules' consequences are used.

    Returns:
        A matrix representation of the rules and a Numpy array of the rules' consequences.
    """
    numpy_matrix = np.empty(
        shape=(len(rules), len(unique_premise_variables) + 1), dtype=int
    )
    numpy_matrix[:, :] = -1
    for idx, rule in enumerate(rules):
        # add the rule's premise and consequence to the matrix
        # assuming the first variable is the output variable - ignoring MIMO systems
        numpy_matrix[idx, -1] = rule.consequence.indices[0][0][-1]
        for variable_idx, variable_value in rule.premise.indices[0]:
            numpy_matrix[idx, variable_idx] = variable_value

    if torch.is_tensor(output_values):
        output_values = output_values.cpu().detach().numpy()
    numpy_matrix[:, -1] = output_values.argmax(-1)
    return numpy_matrix, output_values


def parse_reduced_rules(
    numpy_matrix: np.ndarray,
    output_values: np.ndarray,
    reduced_rules: rpy2.robjects.vectors.ListVector,
    rules: List[Rule],
    t_norm: TNorm,
    device: torch.device,
) -> Tuple[List[Rule], List[float]]:
    """
    Go through the reduced rules and extract the rules' premises and consequences. The
    reduced rules are in the form of R objects, which are converted to Python objects
    using rpy2. The reduced rules are then parsed, and the final rules and output values
    are returned. Some reduced rules may not be valid, in which case they are ignored. For a
    reduced rule to be valid, it must have a Laplace value greater than the average and a support
    value greater than the minimum support value. The final rules are in the form of a list of Rule
    objects. The output values are in the form of a list of floats.

    Args:
        numpy_matrix: The matrix of the fuzzy logic rules.
        output_values: The output values of the fuzzy logic rules.
        reduced_rules: The reduced rules in the form of R objects.
        rules: The original fuzzy logic rules.
        t_norm: The t-norm to use for the fuzzy logic rules.

    Returns:
        The final rules and output values.
    """
    # the last column of the matrix is the output variable
    output_variable_index: int = numpy_matrix.shape[-1] - 1
    final_rules, final_output_values = [], []
    laplaces = np.array(
        [float(reduced_rule.rx("laplace")[0][0]) for reduced_rule in reduced_rules]
    )
    for reduced_rule in reduced_rules:
        premise = frozenset(
            zip(
                map(
                    lambda x: int(x) - 1,  # the R language counts from 1, Python from 0
                    reduced_rule.rx("idx")[0],
                ),  # get the indices of the variables
                map(
                    int, reduced_rule.rx("values")[0]
                ),  # list of strings to list of ints
            )  # map the (linguistic) variable indices to the (linguistic term) values
        )
        if output_values is None:
            # the following is for Mamdani fuzzy logic controllers
            consequence = frozenset(
                zip(  # the consequent will only have one variable
                    [
                        output_variable_index
                    ],  # singleton list of the output variable index
                    map(
                        int, reduced_rule.rx("consequent")[0]
                    ),  # list of strings to list of ints
                )  # map the (linguistic) variable indices to the (linguistic term) values
            )
        else:
            consequence = NAryRelation((len(final_rules), 0), device=device)
        laplace = float(reduced_rule.rx("laplace")[0][0])
        if (
            laplace > laplaces.mean()
            and (len(tuple(reduced_rule.rx("support")[0])) / len(rules)) > 0.2
        ):  # the confidence is above average and the support is above 20%
            final_rules.append(
                Rule(
                    premise=t_norm(*premise, device=device),
                    consequence=consequence,
                )
            )
            corresponding_output_values = (
                output_values[
                    np.ix_(
                        list(
                            map(
                                lambda x: int(x)
                                - 1,  # the R language counts from 1, Python from 0
                                reduced_rule.rx("support")[0],
                            )
                        )
                    )
                ]
                .mean(axis=0)
                .tolist()
            )
            final_output_values.append(corresponding_output_values)

        else:  # the confidence is below average
            for idx in tuple(reduced_rule.rx("support")[0]):
                # for each rule that supports the reduced rule
                final_rules.append(rules[idx - 1])

            corresponding_output_values = output_values[
                np.ix_(
                    list(
                        map(
                            lambda x: int(x)
                            - 1,  # the R language counts from 1, Python from 0
                            reduced_rule.rx("support")[0],
                        )
                    )
                )
            ].tolist()
            for val in corresponding_output_values:
                final_output_values.append(val)
    return final_rules, final_output_values
