import matplotlib.pyplot as plt
import numpy as np

# create some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# create the figure and subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1, 2]})

# plot the data on the subplots
axs[0, 0].plot(x, y1)
axs[0, 1].plot(x, y2)
axs[1, 2].plot(x, y3)

# remove the fourth subplot
fig.delaxes(axs[1, 1])

# set the title and labels for each subplot
axs[0, 0].set_title('sin(x)')
axs[0, 1].set_title('cos(x)')
axs[1, 2].set_title('tan(x)')

axs[0, 0].set_xlabel('x')
axs[0, 1].set_xlabel('x')
axs[1, 2].set_xlabel('x')

axs[0, 0].set_ylabel('y')
axs[0, 1].set_ylabel('y')
axs[1, 2].set_ylabel('y')

# adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# show the plot
plt.show()
