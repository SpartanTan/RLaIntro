import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability for exploration
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

# initial state action pair values
START = [3, 0]
GOAL = [3, 11]


def step(state: list, action: int):
    """
    Move one step to the next step
    @param state: list, current state
    @param action: int, action number
    @return: (next_state, reward)
    """
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (action == ACTION_RIGHT and state == START):
        next_state = START
        reward -= 100

    return next_state, reward


def choose_action(state: list, q_value: np.ndarray):
    """
    choose the action under policy epsilon-greedy
    @param state: list, coordinate of the point, index in the matrix
    @param q_value: ndarray, state-action pair value table
    @return: int, action
    """
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


def sarsa(q_value: np.ndarray, expected=False, step_size=ALPHA):
    # Initialize S
    state = START

    # Choose A from S using policy epsilon-greedy
    action = choose_action(state, q_value)
    rewards = 0.0
    while state != GOAL:
        # take action A, observe R, S'
        next_state, reward = step(state, action)
        # Choose A' from S' using policy
        next_action = choose_action(next_state, q_value)
        rewards += reward

        if not expected:
            target = reward + GAMMA * q_value[next_state[0], next_state[1], next_action]
        else:
            target = 0.0
            best_actions = np.argmax(q_value[next_state[0], next_state[1], :])
            # expected value of new state
            # Q(S,A) <- Q(S,A) + alpha * (R + gamma * Sigma(probability * Q(S',a)) - Q(S,A))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    # target += (1 - EPSILON) * q_value[next_state[0], next_state[1], action_]
                    target += (1 - EPSILON + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    # target += EPSILON * q_value[next_state[0], next_state[1], action_]
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        q_value[state[0], state[1], action] += step_size * (reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards


def q_learning(q_value: np.ndarray, step_size=ALPHA):
    """
    One episode using q-learning
    @param q_value:
    @param step_size:
    @return: sumed rewards
    """
    # Initialize S
    state = START
    rewards = 0.0

    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # q-learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards


def figure_6_4():
    # episodes of each run
    episodes = 500

    # perform 50 independent runs
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)

    for r in tqdm(range(runs)):
        # Initialize Q(S,A)
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)

        # Loop for each episode
        for i in range(0, episodes):
            # run one episode
            # sum up the rewards in each episode
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)

    # averaging over independent runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.show()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)


def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)


if __name__ == "__main__":
    figure_6_4()
    figure_6_3()
