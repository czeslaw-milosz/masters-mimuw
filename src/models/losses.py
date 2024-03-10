import numpy as np
import torch
import torchmetrics

def l2_penalty(parameters) -> float:
    """Calculate the L2 penalty for the given parameters.

    Args:
        parameters (iterable): An iterable of parameters.

    Returns:
        float: The L2 penalty.
    """
    return torch.linalg.norm(parameters, ord="fro")**2


def cosine_penalty(parameters) -> float:
    """Calculate the cosine penalty for the given parameters.

    Args:
        parameters (iterable): An iterable of parameters.

    Returns:
        float: The cosine penalty.
    """
    return 1. - torchmetrics.functional.pairwise_cosine_similarity(parameters, reduction="sum").sum()
