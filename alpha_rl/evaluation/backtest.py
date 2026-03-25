import numpy as np
import pandas as pd
from typing import List, Tuple, Generator

def purged_and_embargoed_kfold(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Marcos Lopez de Prado's Purged and Embargoed K-Fold split.
    Prevents lookahead bias and serial correlation leakage in financial time series.
    """
    indices = np.arange(n_samples)
    embargo = int(n_samples * embargo_pct)
    test_fold_size = n_samples // n_splits

    for i in range(n_splits):
        test_start = i * test_fold_size
        test_end = (i + 1) * test_fold_size if i < n_splits - 1 else n_samples
        test_indices = indices[test_start:test_end]

        # Purge overlapping training points & apply post-test embargo
        train_left = indices[:max(0, test_start)]
        train_right = indices[min(n_samples, test_end + embargo):]
        train_indices = np.concatenate([train_left, train_right])

        yield train_indices, test_indices
