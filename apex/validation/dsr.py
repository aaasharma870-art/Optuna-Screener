"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""

import math

from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649


def _expected_max_sr(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe ratio under the null hypothesis.

    Formula::

        E[max(SR)] = sqrt(V) * ((1 - gamma) * Z(1 - 1/N)
                                + gamma * Z(1 - 1/(N*e)))

    where *gamma* is the Euler-Mascheroni constant, *V* is the variance of
    individual Sharpe ratios, and *Z* is the standard normal quantile.
    """
    if n_trials <= 1:
        return 0.0
    std = math.sqrt(max(sr_variance, 1e-12))
    gamma = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return std * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(observed_sr: float, n_trials: int,
                          sr_variance: float, skew: float, kurtosis: float,
                          n_samples: int) -> float:
    """Compute the Deflated Sharpe Ratio.

    Returns the probability (0..1) that the observed Sharpe ratio exceeds
    the expected maximum under the null, after adjusting for non-normality.

    Parameters
    ----------
    observed_sr : float
        The observed (best) Sharpe ratio.
    n_trials : int
        Number of trials / strategies tested.
    sr_variance : float
        Variance of Sharpe ratios across trials.
    skew : float
        Skewness of the return distribution.
    kurtosis : float
        Kurtosis of the return distribution (excess kurtosis + 3 expected,
        but the formula uses ``kurtosis - 3``).
    n_samples : int
        Number of return observations (T).

    Returns
    -------
    float
        Probability in (0, 1).  High values mean the observed Sharpe is
        unlikely to be a fluke.
    """
    sr0 = _expected_max_sr(sr_variance, n_trials)
    if n_samples <= 1:
        return 0.5

    # Adjusted standard error of the Sharpe ratio
    sr_sq = observed_sr * observed_sr
    denom = (1.0
             - skew * observed_sr
             + (kurtosis - 3.0) / 4.0 * sr_sq)
    denom = max(denom, 1e-12)
    se = math.sqrt(denom / (n_samples - 1.0))

    if se < 1e-15:
        return 1.0 if observed_sr > sr0 else 0.0

    z = (observed_sr - sr0) / se
    return float(norm.cdf(z))
