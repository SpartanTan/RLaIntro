import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# # of states except for terminal states
N_STATES = 1000

# all states
# [1... 1000]
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100


def compute_true_value():
    # true state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0
    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    # -1*step or 1*step
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]

        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value


def step(state, action):
    """
    take an @action at @state, return new state and reward for this transition
    @param state:
    @param action:
    @return:
    """
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward


def get_action():
    """
    get an action, following random policy
    @return: -1 or 1, 50%
    """
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1


class ValueFunction:
    def __init__(self, num_of_groups: int):
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups

        # thetas, estimation of the value
        self.params = np.zeros(num_of_groups)

    def value(self, state: int):
        """
        find the estimated value of the state
        @param state: int
        @return: estimate value int
        """
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size
        return self.params[group_index]

    def update(self, delta, state):
        """
        update parameters
        @param delta: step size * (target - old estimation)
        @param state:
        """
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta


def gradient_monte_carlo(value_function: ValueFunction, alpha: float, distribution=None):
    """
    gradient Monte Carlo algorithm
    @param value_function:
    @param alpha:
    @param distribution: a list stores how many times each state has been visited
    """
    state = START_STATE
    trajectory = [state]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state

    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1


def figure_9_1(true_value):
    episodes = int(1e5)
    # step size
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    value_function = ValueFunction(num_of_groups=10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    file_name = "true_value.npy"

    # load the true_value from .npy
    if os.path.exists(file_name):
        true_value = np.load(file_name)
        print("true value loaded from file")
    else:
        true_value = compute_true_value()
        np.save(file_name, true_value)
        print("true value calculated and saved")
    figure_9_1(true_value)
