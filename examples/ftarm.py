"""
Demo of working with the Fuzzy Temporal Association Rule Mining algorithm.
"""

import datetime

import torch
import numpy as np
import pandas as pd
from fuzzy.sets.continuous.impl import Triangular
from fuzzy.logic import LinguisticVariables, KnowledgeBase

from fuzzy_ml.association.temporal import (
    FuzzyTemporalAssocationRuleMining as FTARM,
)


AVAILABLE_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def make_example():
    """
    Make a toy example that is used in the original Fuzzy Temporal Association Rule Mining paper.

    Returns:
        pd.DataFrame, soft.computing.knowledge.KnowledgeBase
    """
    aug_5 = datetime.date(year=2011, month=8, day=5)
    aug_6 = datetime.date(year=2011, month=8, day=6)
    dates = [aug_5, aug_5, aug_5, aug_6, aug_6]
    example_dataframe = pd.DataFrame(
        {
            "date": dates,
            "A": [5, 2.5, 0, 2.5, 2.5],
            "B": [0, 2, 0, 2, 5],
            "C": [4, 0, 4, 0, 0],
            "D": [0, 0, 0, 4, 4],
            "E": [0, 0, 0, 0, 2],
        }
    )

    variables = {
        "A": Triangular(
            centers=np.array([5, 10]),
            widths=np.array([5] * 2),
            # labels=["low", "high"],
            device=AVAILABLE_DEVICE,
        ),
        "B": Triangular(
            centers=np.array([4, 8]),
            widths=np.array([4] * 2),
            # labels=["low", "high"],
            device=AVAILABLE_DEVICE,
        ),
        "C": Triangular(
            centers=np.array([3, 6]),
            widths=np.array([3] * 2),
            # labels=["low", "high"],
            device=AVAILABLE_DEVICE,
        ),
        "D": Triangular(
            centers=np.array([2, 4]),
            widths=np.array([2] * 2),
            # labels=["low", "high"],
            device=AVAILABLE_DEVICE,
        ),
        "E": Triangular(
            centers=np.array([2, 4]),
            widths=np.array([2] * 2),
            # labels=["low", "high"],
            device=AVAILABLE_DEVICE,
        ),
    }

    return example_dataframe, KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=list(variables.values()), targets=[]
        ),
        rules=[],
    )


if __name__ == "__main__":
    dataframe, knowledge_base = make_example()
    print(dataframe.head())
    ftarm = FTARM(
        dataframe,
        knowledge_base,
        min_support=0.3,
        device=AVAILABLE_DEVICE,
    )
    candidates_family = ftarm.find_candidates()
    rules = ftarm.find_association_rules(min_confidence=0.8)
    for rule in rules:
        print(
            f"{rule.antecedents} --> {rule.consequents} confidence: {rule.confidence}"
        )
