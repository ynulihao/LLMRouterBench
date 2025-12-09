"""Matrix Factorization router components.

NOTE: This package is the primary focus in the current iteration.
Other routers are kept for reference.
"""

from .model import MFModel
from .train_matrix_factorization import PairwiseDataset, MFModel_Train

__all__ = ["MFModel", "PairwiseDataset", "MFModel_Train"]
