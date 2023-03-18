import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple

##
# [T] A B C D E [T]
#  0  1 2 3 4 5  6

# 0 is the left terminal state
# 6 is the right terminal state
# 1...5 represents A..E
VALUES = np.zeros(7)

# Assume all rewards are 0
VALUES[1:6] = 0.3
# Terminates are right, reward 1
VALUES[6] = 1

# setup true state value
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

# define actions
ACTION_LEFT = 0
ACTION_RIGHT = 1


# TD(0)
# V(St) <- V(St)+ alpha* [Rt+1 + gamma *V(St+1) - V(St)]
def temporal_difference(values: np.ndarray, alpha=0.1, batch=False) -> (list, list):
    """
    One episode of a random walk.
    Agent perform one step, update the value function using TD(0).
    Abort until it terminates.

    @param values: value function, list of the values of each state
    @param alpha: step-size parameter
    @param batch:
    @return: trajectory:  list of states at each step, [S0, S1, S2, ...]
    @return: rewards:  list of reward at each step, [R1, R2, R3, ...]
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
        # assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break
        rewards.append(reward)
    return trajectory, rewards


# Monte-Carlo incremental update
# V(St) <- V(St) + alpha + [Gt - V(St)]
def monte_carlo(values: np.ndarray, alpha=0.1, batch=False) -> Tuple[list, list]:
    """
    MC method

    @param values: value function, list of the values of each state
    @param alpha: step size
    @param batch: (bool) choose True to enable batch training
    @return: trajectory (list): stores the state at each step
            return (list): [returns] * (len(trajectory) - 1)
                        e.g. [return, return, return, ...]

    """
    state = 3
    trajectory = [state]

    # loop until reaches the terminals
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
            returns = 0.0
            break
    # update the value function immediately after each episode if not batch
    if not batch:
        for state_ in trajectory[:-1]:
            values[state_] += alpha * (returns - values[state_])
    return trajectory, [returns] * (len(trajectory) - 1)


def compute_state_value():
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)

    plt.figure(1)
    # loop through 100 episodes
    for i in range(episodes[-1] + 1):
        # if the episode is the chosen one, plot the value of exactly this episode
        if i in episodes:
            plt.plot(("A", "B", "C", "D", "E"), current_values[1:6], label=str(i) + ' episodes')
        temporal_difference(current_values)

    plt.plot(("A", "B", "C", "D", "E"), TRUE_VALUE[1:6], label='true values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()


def rms_error():
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = "TD"
            linestyle = "solid"
        else:
            method = "MC"
            linestyle = "dashdot"
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                if method == "TD":
                    temporal_difference(current_values, alpha)
                else:
                    monte_carlo(current_values, alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', $\\alpha$ = %.02f' % (alpha))
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.legend()


def batch_update(method, episodes: int, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    total_errors = np.zeros(episodes)

    # loop through each independent run
    for r in tqdm(range(0, runs)):
        # perform one run
        # reset all the parameters to origin
        current_values = np.copy(VALUES)
        current_values[1:6] = -1
        # rms error of each episode in this run
        errors = []
        # track show trajectories and reward/return sequences
        trajectories = []
        rewards = []
        # loop in episodes
        for ep in range(episodes):
            # perform one episode
            if method == 'TD':
                trajectory_, rewards_ = temporal_difference(current_values, batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, batch=True)
            # Add the trajectories from this episode
            trajectories.append(trajectory_)  # [ [1,2,3,4,5], [1,3,4,5,]....]
            rewards.append(rewards_)

            # Looping through the training data from existing episodes
            # repeating the update rule until it converges
            while True:
                # keep feeding our algorithm with trajectories seen so far until state value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    # retrieve one trajectory and rewards from one episode
                    for i in range(0, len(trajectory_) - 1):
                        # loop through the time steps in this episode
                        if method == 'TD':
                            # V(St) <- V(St)+ alpha* [Rt+1 + gamma *V(St+1) - V(St)]
                            # (error) update = Rt+1 + gamma * V(St+1) -V(St)
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - \
                                                       current_values[trajectory_[i]]
                        else:
                            # MC V(St) <- V(St) + alpha + [Gt - V(St)]
                            # update = Gt - V(St)
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                # end of walking through all the episodes
                # after running through all the episodes
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    # if converge
                    break
                # if not converge, perform batch updating
                # change the value function
                current_values += updates
            # calculate rms error for this loop of the episodes
            # [ep1, ep2, ep3, .... ep100]
            # [100, 80, 60, 20, ...]
            errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0))

        # accumulate the errors of each run
        total_errors += np.asarray(errors)

    # average the total errors from the runs
    total_errors /= runs
    return total_errors


def batch_updating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    total_errors = np.zeros(episodes)
    for r in tqdm(range(0, runs)):
        current_values = np.copy(VALUES)
        current_values[1:6] = -1
        errors = []
        # track shown trajectories and reward/return sequences
        trajectories = []
        rewards = []
        for ep in range(episodes):
            if method == 'TD':
                trajectory_, rewards_ = temporal_difference(current_values, batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, batch=True)
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            while True:
                # keep feeding our algorithm with trajectories seen so far until state value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(0, len(trajectory_) - 1):
                        if method == 'TD':
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - \
                                                       current_values[trajectory_[i]]
                        else:
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # perform batch updating
                current_values += updates
            # calculate rms error
            errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0))
        total_errors += np.asarray(errors)
    total_errors /= runs
    return total_errors


def example_6_2():
    # plotting the value function
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    # plotting the root-mean-square error
    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()

    plt.show()


def figure_6_2():
    episodes = 100 + 1
    td_errors = batch_update('TD', episodes)
    mc_errors = batch_update('MC', episodes)

    plt.figure(2)
    plt.plot(td_errors, label='TD')
    plt.plot(mc_errors, label='MC')
    plt.title("Batch Training")
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS error, averaged over states')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # example_6_2()
    figure_6_2()
