import matplotlib.pyplot as plt
import numpy as np
import os

# import pickle

file_name = "true_value.npy"
if __name__ == "__main__":
    true_value = np.zeros(3)
    if os.path.exists(file_name):
        loaded_true_value = np.load(file_name)
        print(loaded_true_value)
    else:
        np.save(file_name, true_value)
        print(true_value)
