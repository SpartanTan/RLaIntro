import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure as figure
import matplotlib.axes as axes
from matplotlib.table import Table

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']

ACTION_PROB = 0.25


def step(state: list, action: np.ndarray) -> (list, int):
    """
    Step change function
    Add the action nadrray onto the current state to get next state.
    Check the next state to give return value.


    :param state: current coordinate, [i, j]
    :param action: a 2D array for moving the coordinate, [i j]
    :return: (next_pose:[i,j], reward: int)
    """
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5
    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def draw_image(value_table: np.ndarray):
    fig, ax = plt.subplots()  # type: figure.Figure, axes.Axes
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = value_table.shape
    width, height = 1 / nrows, 1 / ncols

    for (i, j), val in np.ndenumerate(value_table):
        if [i, j] == A_POS:
            val = str(val) + "(A)"
        if [i, j] == B_POS:
            val = str(val) + "(B)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + "(A')"
        if [i, j] == B_PRIME_POS:
            val = str(val) + "(B')"

        tb.add_cell(i, j, width=width, height=height, text=val, loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(value_table)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)


def draw_policy(optimal_values: np.ndarray):
    fig, ax = plt.subplots()  # type: figure.Figure, axes.Axes
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1 / nrows, 1 / ncols

    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            # get the V*(s)
            next_vals.append(optimal_values[next_state[0], next_state[1]])
        best_action = np.where(next_vals == np.max(next_vals))[0]
        val = ''
        for ba in best_action:
            val += ACTIONS_FIGS[ba]

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

        # Row and column labels...
        for i in range(len(optimal_values)):
            tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                        edgecolor='none', facecolor='none')
            tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                        edgecolor='none', facecolor='none')

        ax.add_table(tb)


def figure3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    # get s' from s and action
                    (next_i, next_j), reward = step([i, j], action)
                    # V(s) = sigma(pi(a|s))*Sigma(p(s',r|s,a)*[r + discount*V(s')]
                    # here pi(a|s) = ACTION_PROB for each action
                    # p(s',r|s,a) = 1
                    # V(s') = value[next_i, next_j]
                    new_value[i, j] += ACTION_PROB * 1 * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.show()
            break
        value = new_value


def figure_3_2_linear_system():
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)  # 25*25
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)  # 25*1

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r
    x = np.linalg.solve(A, b)

    draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.show()


def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                # v* = max(a) Sigma(p(s',r|s,a)[r + V*(s')]
                # state value under optimal policy
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            draw_policy(new_value)
            plt.show()
            break
        value = new_value


if __name__ == '__main__':
    # figure_3_2_linear_system()
    # figure3_2()
    figure_3_5()
