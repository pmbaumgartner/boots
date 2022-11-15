from typing import Any, Callable, List

import numpy as np
import vose
from joblib import Parallel, delayed


def resample(data: List[Any], seed: int, scale: int = 1) -> List[Any]:
    size = len(data)
    rng = np.random.default_rng(seed=seed)
    sample_ix = rng.integers(low=0, high=size, size=(size * scale,))
    resampled = [data[i] for i in sample_ix]
    return resampled


def resample_dirichlet(
    data: List[Any], seed: int, scale: int = 1, alpha=4
) -> List[Any]:
    size = len(data)
    rng = np.random.default_rng(seed=seed)
    weights = rng.dirichlet(np.ones(size) * alpha)
    sampler = vose.Sampler(weights, seed=seed)
    sample_ix = sampler.sample(k=size * scale)
    resampled = [data[i] for i in sample_ix]
    return resampled


def estimate(
    resampler: Callable,
    statistic: Callable[[List[Any]], float],
    data: List[Any],
    seed: int,
    scale: int,
) -> float:
    return statistic(resampler(data, seed=seed, scale=scale))


def bootstrap(
    data: List[Any],
    statistic: Callable[[List[Any]], float],
    n_iterations: int,
    seed: int,
    n_jobs: int = -1,
    bayesian: bool = False,
    scale: int = 1,
) -> List[float]:
    """Perform bootstrap sampling on `data`, by calling the `statistic` function
    on the data for `n_iterations`.

    Args:
        data (List[Any]): The input data to calculate the statistic from.
        statistic (Callable[[List[Any]], float]): The statistic to calculate
          from given `data`.
        n_iterations (int): How many iterations to perform sampling.
        seed (int): Random seed for resampling.
        n_jobs (int, optional): Number of parallel jobs for sampling. -1 uses all
            available cores. Defaults to -1.
        bayesian (bool, optional): Perform the bayesian boostrap by sampling from a
            dirichlet prior with α=4.0 to determine sample weights. Defaults to False.
        scale (int, optional): Ratio to oversample, which can be useful when `data` has
            rare events that can cause errors in calculating `statistic`. WARNING: Increasing
            this value leads to false additional precision in the bootstrap distribution.
            Defaults to 1.

    Returns:
        List[float]: Bootstrap samples of the statistic with length `n_iterations`
    """
    resampler = resample_dirichlet if bayesian else resample
    result = Parallel(n_jobs=n_jobs)(
        delayed(estimate)(resampler, statistic, data, seed=seed + i, scale=scale)
        for i in range(n_iterations)
    )
    return result
