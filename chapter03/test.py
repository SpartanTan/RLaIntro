import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    q = np.random.rand(10, 2)
    print(q)
    cc = np.zeros((3, 3, 3))
    print(q[[3, 4], :])
