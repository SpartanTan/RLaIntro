import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1-5 represents A...E

VALUES = np.zeros(7)
VALUES[1:6] = 0.5
VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1  # TRUE_VALUE = [0 1/6 2/6 3/6 4/6 5/6 1]

ACTION_LEFT = 0
ACTION_RIGHT = 1


def temporal_difference(values: np.ndarray, alpha=0.1, batch=False) -> (list, list):
    """
    TD(0) method

    :param values: current states value, will be updated if @batch is false
    :param alpha: step size
    :param batch: whether to update values under a batch of data
    :returns trajectory: a trajectory of states, e.g. [3, 2, 3, 4, 3, 2 ...]
             rewards: a list of reward, e.g.
    """
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])  # V(s) = V(s) + α[R + γV(s+1) - V(s)]
        if state == 6 or state == 0:
            break
        rewards.append(reward)
    return trajectory, rewards


def monte_carlo(values: np.ndarray, alpha=0.1, batch=False):
    """
    MC method
    :param values:
    :param alpha:
    :param batch:
    """
    state = 3  # initialize s
    trajectory = [state]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0
            break
    if not batch:  # not batch, don't need the whole episode?
        for state_ in trajectory[:-1]:
            # MC update
            values[state_] += alpha * (returns - values[state_])  # V(St) = V(St) + α[Gt - V(St)]
    return trajectory, [returns] * (len(trajectory) - 1)


# example 6.2 left
def compute_state_value():
    """
    calculate and print the estimated value function using TD(0)
    """
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)  # initial value
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(("A", "B", "C", "D", "E"), current_values[1:6], label=str(i) + ' episodes')
        temporal_difference(current_values)
    plt.plot(("A", "B", "C", "D", "E"), TRUE_VALUE[1:6], label='true values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()


# example 6.2 right
def rms_error():
    """
    Root mean squared error between the value function learned and the true value function
    """

    # same alpha values can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):  # 7 α, 100 episodes,
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))  # root mean square error
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', $\\alpha$ = %.02f' % alpha)
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, average over states')
    plt.legend()


def example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()

    plt.savefig('example_6_2.png')

def figure_6_2():
    episodes = 100 + 1
    td_errors = batch_updating('TD', episodes)
    mc_errors = batch_updating('MC', episodes)

if __name__ == "__main__":
    example_6_2()
    plt.show()

