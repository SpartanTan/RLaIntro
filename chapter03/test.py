import matplotlib.pyplot as plt
import numpy as np
import os

# import pickle

if __name__ == "__main__":
    order = 3
    bases = []
    for i in range(0, order + 1):
        bases.append(lambda s, i=i: pow(s, i))

    feature = np.asarray([func(3) for func in bases])
    print(feature)
