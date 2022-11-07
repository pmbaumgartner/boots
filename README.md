# Boots - A Tiny Bootstrapping Library

This is a tiny library for doing bootstrap sampling and estimating. It pulls together various tricks to make the process as fast and painless as possible. The tricks included are:

- Parallel execution with [`joblib`](https://joblib.readthedocs.io/en/latest/parallel.html) 
- [The Bayesian bootstrap](https://matteocourthoud.github.io/post/bayes_boot/) with two-level sampling.
- The [Vose method](https://github.com/MaxHalford/vose) for fast weighted sampling with replacement

**Install**

```bash
pip install git+https://github.com/pmbaumgartner/boots
```

> No PyPI?

I'm working on it. The `vose` dependency is also not on PyPI, and packages uploaded to PyPI can't have git dependencies.

## Example

```python
import numpy as np

x = np.random.pareto(2, 100)

samples = bootstrap(
    data=x,
    statistic=np.median,
    n_iterations=1000,
    seed=1234,
    n_jobs=-1
)

# bayesian two-level w/ 4 parallel jobs
samples = bootstrap(
    data=x,
    statistic=np.median, 
    n_iterations=1000, 
    seed=1234, 
    n_jobs=4, 
    bayesian=True
)

# do something with it
import pandas as pd
posterior = pd.Series(samples)
posterior.describe(percentiles=[0.025, 0.5, 0.975])
```

