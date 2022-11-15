# 🥾 Boots 👢 - A Tiny Bootstrapping Library

This is a tiny library for doing bootstrap sampling and estimating. It pulls together various tricks to make the process as fast and painless as possible. The tricks included are:

- Parallel execution with [`joblib`](https://joblib.readthedocs.io/en/latest/parallel.html) 
- [The Bayesian bootstrap](https://matteocourthoud.github.io/post/bayes_boot/) with two-level sampling.
- The [Vose method](https://github.com/MaxHalford/vose) for fast weighted sampling with replacement

**Install**

```bash
pip install boots
```

For development:

```bash
pip install git+https://github.com/pmbaumgartner/boots
```

## Example

```python
from boots import bootstrap
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

**Paired Statistics**

```python
from boots import bootstrap
import numpy as np


# generate some fake correlated data by sorting two arrays and adding some noise
a = np.sort(np.random.normal(0, 1, 100)) + np.random.normal(0, 1, 100)
b = np.sort(np.random.normal(0, 1, 100)) + np.random.normal(0, 1, 100)
pairs = list(zip(a, b))

# for paired (or row-wise) metrics you might need to
# create a wrapper function that unpacks
# each row's values into array arguments for your metric function
def corr_unwrap(pairs):
    a1, a2 = zip(*pairs)
    corr = np.corrcoef(a1, a2)[0, 1]
    return corr


samples = bootstrap(
    data=pairs,
    statistic=corr_unwrap,
    n_iterations=1000,
    seed=1234,
    n_jobs=-1,
    bayesian=True
)

import pandas as pd
posterior = pd.Series(samples)
posterior.describe(percentiles=[0.025, 0.5, 0.975])
```

