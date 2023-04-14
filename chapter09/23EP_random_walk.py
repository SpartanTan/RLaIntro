import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Union

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


def compute_true_value() -> np.ndarray:
    """
    compute the true state values for 1000 state random walk
    @return: ndarray, [1002 * 1]
    """
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


# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
FOURIER_BASES = 1


class BasesValueFunction:
    def __init__(self, order: int, type: int):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == FOURIER_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    def value(self, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        """
        w<-w + delta * derivative_value
        @param delta:
        @param state:
        """
        # map the state spce into [0, 1]
        state /= float(N_STATES)
        # get derivative value
        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivative_value


class TilingsValueFunction:
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        """

        @param numOfTilings: # of tilings
        @param tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
        @param tilingOffset: specifies how tilings are put together
        """
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling
        self.tilingSize = N_STATES // tileWidth + 1

        # weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    def update(self, delta, state):
        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings

        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta


def gradient_monte_carlo(value_function: Union[ValueFunction, BasesValueFunction], alpha: float, distribution=None):
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


def semi_gradient_temporal_difference(value_function: ValueFunction, n, alpha: float):
    state = START_STATE
    states = [state]
    rewards = [0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')

    while True:
        # go to next time step
        time += 1

        if time < T:
            # choose an action randomly
            action = get_action()
            next_state, reward = step(state, action)

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            # add state value to the return
            if update_time + n < T:
                returns += value_function.value(states[update_time + n])
            state_to_update = states[update_time]
            # update the value function
            if not state_to_update in END_STATES:
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
        if update_time == T - 1:
            break
        state = next_state


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


def figure_9_2_left(true_value):
    """
    semi-gradient TD on 1000-state random walk
    @param true_value:
    """
    episodes = int(1e5)
    alpha = 2e-4
    value_function = ValueFunction(10)
    for ep in tqdm(range(episodes)):
        semi_gradient_temporal_difference(value_function, 1, alpha)

    stateValues = [value_function.value(i) for i in STATES]
    plt.plot(STATES, stateValues, label='Approximate TD value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()


def figure_9_2_right(true_value):
    """
    different alphas and steps for semi-gradient TD
    @param true_value:
    """
    # all possible steps
    steps = np.power(2, np.arange(0, 10))

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1)

    # each run has 10 episodes
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))

    for run in tqdm(range(runs)):
        for step_ind, step in zip(range(len(steps)), steps):
            for alpha_ind, alpha in zip(range(len(alphas)), alphas):
                # we have 20 aggregations in this example
                value_function = ValueFunction(20)
                for ep in range(0, episodes):
                    semi_gradient_temporal_difference(value_function, step, alpha)
                    # calculate the RMS error
                    state_value = np.asarray([value_function.value(i) for i in STATES])
                    errors[step_ind, alpha_ind] += np.sqrt(
                        np.sum(np.power(state_value - true_value[1: -1], 2)) / N_STATES)

    # take average
    errors /= episodes * runs
    # truncate the error
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()


def figure_9_2(true_value):
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    figure_9_2_left(true_value)
    plt.subplot(2, 1, 2)
    figure_9_2_right(true_value)

    plt.show()


def figure_9_5(true_value):
    runs = 1
    episodes = 5000
    # # of bases
    orders = [5, 10, 20]

    alphas = [1e-4, 5e-5]
    labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]

    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in range(runs):
        for i in range(len(orders)):
            value_functions = [BasesValueFunction(orders[i], POLYNOMIAL_BASES),
                               BasesValueFunction(orders[i], FOURIER_BASES)]
            for j in range(len(value_functions)):
                for episode in tqdm(range(episodes)):
                    # gradient Monte Carlo algorithm
                    gradient_monte_carlo(value_functions[j], alphas[j])
                    # get state values under current value function
                    state_values = [value_functions[j].value(state) for state in STATES]

                    # get the root-mean-squared error
                    errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))
    # average over independent runs
    errors /= runs
    for i in range(len(alphas)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()


def figure_9_10(true_value):
    runs = 1
    # number of episodes
    episodes = 5000
    num_of_tilings = 50

    # each tile will cover 200 states
    tile_width = 200

    # how to put so many tilings
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']

    # track errors for each episode
    errors = np.zeros((len(labels), episodes))

    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [TilingsValueFunction(num_of_tilings, tile_width, tiling_offset),
                           ValueFunction(N_STATES // tile_width)]
        for i in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 1.0 / (episode + 1)

                # gradient Monte Carlo algorithm
                gradient_monte_carlo(value_functions[i], alpha)

                # get state values under current value function
                state_values = [value_functions[i].value(state) for state in STATES]

                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

        # average over independent runs
    errors /= runs

    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
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
    # figure_9_1(true_value)
    # figure_9_2(true_value)
    # figure_9_5(true_value)
    figure_9_10(true_value)
