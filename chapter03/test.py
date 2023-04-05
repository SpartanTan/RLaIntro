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
    testdict = dict()
    testdict[tuple([1, 2])] = dict()
    testdict[tuple([1, 2])][3] = [[2, 3], 1]
    testdict[tuple([1, 2])][5] = [[4, 5], 1]
    testdict[tuple([3, 4])] = dict()
    testdict[tuple([3, 4])][3] = [[2, 3], 1]
    testdict[tuple([3, 4])][5] = [[4, 5], 1]

    state_index = np.random.choice(range(len(testdict.keys())))
    print(state_index)
    state = list(testdict)[state_index]
    print(state)
    action_index = np.random.choice(range(len(testdict[state].keys())))
    print(action_index)
    action = list(testdict[state])[action_index]
    print(action)
