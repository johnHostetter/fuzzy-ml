"""
Provides utility functions that help guarantee reproducibility.
"""

import os
import random

import torch
import numpy as np


def set_rng(seed: int) -> None:
    """
    Set the random number generator.

    Args:
        seed: The seed to use for the random number generator.

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
