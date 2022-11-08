from src.boots import bootstrap

import numpy as np


def test_basic():
    iterations = 1000
    x = np.random.pareto(2, 100).tolist()

    samples = bootstrap(
        data=x, statistic=np.median, n_iterations=iterations, seed=1234, n_jobs=-1
    )

    assert len(samples) == iterations


def test_bayesian():
    iterations = 1000
    x = np.random.pareto(2, 100).tolist()

    samples = bootstrap(
        data=x,
        statistic=np.median,
        n_iterations=iterations,
        seed=1234,
        n_jobs=-1,
        bayesian=True,
    )

    assert len(samples) == iterations
