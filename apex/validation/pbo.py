"""Probability of Backtest Overfitting (Bailey et al., 2015)."""

import math

import numpy as np


def probability_of_backtest_overfitting(is_scores: np.ndarray,
                                        oos_scores: np.ndarray) -> float:
    """Estimate the Probability of Backtest Overfitting (PBO).

    Parameters
    ----------
    is_scores : np.ndarray, shape (n_trials, n_folds)
        In-sample performance score for each trial in each fold.
    oos_scores : np.ndarray, shape (n_trials, n_folds)
        Out-of-sample performance score for each trial in each fold.

    Returns
    -------
    float
        Fraction of folds where the IS-best trial under-performs OOS
        (logit of its OOS rank < 0, i.e. rank < median).

    Notes
    -----
    For each fold *j*:
      1. Find the trial *i** that maximises ``is_scores[i, j]``.
      2. Compute the percentile rank of ``oos_scores[i*, j]`` among all
         trials' OOS scores in that fold.
      3. Compute ``logit(rank) = log(rank / (1 - rank))``.
      4. Count as "overfit" if logit < 0  (i.e. rank < 0.5).

    PBO = fraction of folds flagged as overfit.
    """
    is_scores = np.asarray(is_scores, dtype=np.float64)
    oos_scores = np.asarray(oos_scores, dtype=np.float64)

    n_trials, n_folds = is_scores.shape
    if n_trials < 2 or n_folds < 1:
        return 0.0

    overfit_count = 0
    for j in range(n_folds):
        # IS-best trial
        best_trial = int(np.argmax(is_scores[:, j]))
        oos_val = oos_scores[best_trial, j]

        # Percentile rank of the IS-best trial's OOS score
        rank = np.mean(oos_scores[:, j] <= oos_val)
        # Clamp to avoid log(0) or log(inf)
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)
        logit = math.log(rank / (1.0 - rank))

        if logit < 0.0:
            overfit_count += 1

    return overfit_count / n_folds
