import matplotlib.pyplot as plt
import numpy as np

# create some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(nrows=2, ncols=2)

# create the subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# plot the data on the subplots
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)

# set the title and labels for each subplot
ax1.set_title('sin(x)')
ax2.set_title('cos(x)')
ax3.set_title('tan(x)')

ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax3.set_xlabel('x')

ax1.set_ylabel('y')
ax2.set_ylabel('y')
ax3.set_ylabel('y')

# adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# show the plot
plt.show()
