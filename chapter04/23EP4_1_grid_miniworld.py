import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure as figure
import matplotlib.axes as axes
from matplotlib.table import Table

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)


def is_terminal(state: list):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state: list, action: np.ndarray):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def compute_state_value(in_place=True, discount=1.0):
    # v_k+1(s) = Sigma(pi(a|s))*Sigma(p(s',r|s,a))*[r+discount*v_k(s')]

    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0

    while True:
        if in_place:
            # use the state_values to update value table
            state_values = new_state_values
        else:
            # use the old state value to update the table
            state_values = new_state_values.copy()
        # save the old state values
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1
    return new_state_values, iteration


def figure_4_1():
    async_values, async_iteration = compute_state_value(in_place=True)
    sync_values, sync_iteration = compute_state_value(in_place=False)
    draw_image(np.round(async_values, decimals=2))
    # draw_image(sync_values)
    plt.show()
    print('In-place: {} iterations'.format(async_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))


if __name__ == "__main__":
    figure_4_1()
