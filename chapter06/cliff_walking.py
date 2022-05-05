import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability of exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state-action pair values
START = [3, 0]
GOAL = [3, 11]


def step(state: list, action: int) -> (list, int):
    """
    one step to the next state
    :param state: current position, [row, col]
    :param action: integer number, represent the action
    :return: next state position, [row, col]
    """
    i, j = state
    if action == ACTION_UP:
        next_state =  [max(i - 1, 0), j]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i ==2 and 1 <= j <= 10) or (action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START
    return next_state, reward


def choose_action(state: list, q_value: np.ndarray) -> int:
    """
    choose an action based on epsilon-greedy algorithm
    :param state: current position, [row, col]
    :param q_value: action-value function array
    """
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1]]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


def q_learning(q_value: np.ndarray, step_size: float = ALPHA) -> float:
    """
    an episode with Q-Learning
    :param q_value: action-value function array, WORLD_HEIGHT * WORLD_WIDTH * 4
    :param step_size: Alpha
    """
    # initialize the state
    state = START
    rewards = 0.0

    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward

        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], action])
        state = next_state
    return rewards


def figure_6_4():
    # episodes of each run
    episodes = 500

    # independent runs to perform
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            # cut off the value by -100 to draw the figure more elegantly
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)


