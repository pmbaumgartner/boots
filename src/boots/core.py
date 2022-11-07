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
):
    resampler = resample_dirichlet if bayesian else resample
    result = Parallel(n_jobs=n_jobs)(
        delayed(estimate)(resampler, statistic, data, seed=seed + i, scale=scale)
        for i in range(n_iterations)
    )
    return result
