import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# matplotlib.use('Agg')

WORLD_SIZE = 4
DISCOUNT = 0.9

ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']

ACTION_PROB = 0.25

fig, ax = plt.subplots()


def is_terminal(state: list) -> bool:
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state: list, action: np.ndarray) -> (list, int):
    """step change function

    :param state: current coordinate, [i, j]
    :type state: list
    :param action: a 2D array for moving the coordinate, [i j]
    :type action: np.ndarray
    :return: next_state, reward
    :rtype: (list, int)

    """
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def draw_image(image: np.ndarray):
    # fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1 / ncols, 1 / nrows

    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # Row anc column labels
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        ax.add_table(tb)


def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values  # sate_value points to the new_state_value
        else:
            state_values = new_state_values.copy()  # a full copy
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
        # draw_image(np.round(state_values, decimals=2))
        # plt.pause(0.0001)

    return new_state_values, iteration


def figure_4_1():
    _, async_iteration = compute_state_value(in_place=True)
    values, sync_iteration = compute_state_value(in_place=False)
    draw_image(np.round(values, decimals=2))
    print('In-place: {} iterations'.format(async_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    plt.savefig('figure_4_1.png')

if __name__ == '__main__':
    # plt.ion()
    figure_4_1()
