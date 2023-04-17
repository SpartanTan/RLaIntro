import numpy as np

order = 1
bases = []
for i in range(0, order + 1):
    bases.append(lambda s, i=i: pow(s, i))  # the i=i here is assign the i value to i
print(bases)
state = 3
feature = np.asarray([func(state) for func in bases])
print(feature)
