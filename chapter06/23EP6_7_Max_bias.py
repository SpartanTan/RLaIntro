import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# state a
STATE_A = 0

# state b
STATE_B = 1

# use one terminal state
STATE_TERMINAL = 2

# state from state a
STATE_START = STATE_A

# possible actions in A
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1

# possible actions in B, maybe 10 actions
ACTIONS_B = range(0, 10)

#  All possible actions
# action[0] returns a list of two possible actions
# action[1] returns a range typre object, range(0, 10); The range object can be accessed by range_obj[0]
STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.1

# discount for max value
GAMMA = 1.0

# state action pair values, if a state is a terminal state, the value is always 0
# [vales of actions in state A, values in state B, values in state terminal]
# q[0] represents getting all the action values from state A
# q[1]... from state B
# q[0][0] means the action value of taking right in state A
INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]

# set up destination for each state and each action
TRANSITION = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTIONS_B)]


def choose_action(state: int, q_value: list):
    """
    choose an action based on current state and the action value function
    @param state: an int, represents the current state, 0 is A, 1 is B
    @param q_value: a list of ndarrays, q[0] gets all action values in A
    @return: an int, representing the action to take, 0 is right, 1 is left in A; 0~9 in B
    """
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(STATE_ACTIONS[state])
    else:
        # getting all action values from current state
        values_ = q_value[state]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


def take_action(state: int, action: int):
    """
    return a reward by applying this action in this state
    @param state: state, int
    @param action: aciton, int
    @return: reward, 0 if state A, mean -0.1, variance 1 if state B
    """
    if state == STATE_A:
        return 0
    return np.random.normal(-0.1, 1)


def q_learning(q1: list, q2: list = None):
    state = STATE_START

    # track the numbers of action left in state A
    left_count = 0
    while state != STATE_TERMINAL:
        if q2 is None:
            action = choose_action(state, q1)
        else:
            # derive a action from Q1 and Q2
            action = choose_action(state, [item1 + item2 for item1, item2 in zip(q1, q2)])

        if state == STATE_A and action == ACTION_A_LEFT:
            left_count += 1

        reward = take_action(state, action)
        next_state = TRANSITION[state][action]

        if q2 is None:
            active_q = q1
            target = np.max(active_q[next_state])
        else:
            if np.random.binomial(1, 0.5) == 1:
                active_q = q1
                target_q = q2
            else:
                active_q = q2
                target_q = q1

            best_action = np.random.choice(
                [action_ for action_, value_ in enumerate(active_q[next_state]) if
                 value_ == np.max(active_q[next_state])])
            target = target_q[next_state][best_action]

        # Q-learning update
        active_q[state][action] += ALPHA * (reward + GAMMA * target - active_q[state][action])
        state = next_state
    return left_count


def figure_6_5():
    episodes = 300
    runs = 1000
    left_counts_q = np.zeros((runs, episodes))
    left_counts_double_q = np.zeros((runs, episodes))

    for run in tqdm(range(runs)):
        q = copy.deepcopy(INITIAL_Q)
        q1 = copy.deepcopy(INITIAL_Q)
        q2 = copy.deepcopy(INITIAL_Q)
        for ep in range(0, episodes):
            left_counts_q[run, ep] = q_learning(q)
            left_counts_double_q[run, ep] = q_learning(q1, q2)

    left_counts_q = left_counts_q.mean(axis=0)
    left_counts_double_q = left_counts_double_q.mean(axis=0)

    plt.plot(left_counts_q, label='Q-Learning')
    plt.plot(left_counts_double_q, label='Double Q-Learning')
    plt.plot(np.ones(episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    figure_6_5()
