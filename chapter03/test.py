import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.axes as axes
from matplotlib.table import Table
from typing import List, Optional
from scipy.stats import poisson

poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


if __name__ == "__main__":
    optimal_policy = []
    optimal_policy.append([])
    optimal_policy[-1].append('C')
    print(optimal_policy)
