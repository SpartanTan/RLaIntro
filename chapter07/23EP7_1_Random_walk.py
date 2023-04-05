import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# 19 states, -1 outcome on the left, all values initialzed to 0

# all states
N_STATES = 19

# discount
GAMMA = 1

# all states but terminal states
STATES = np.arange(1, N_STATES + 1)  # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state value from bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0


def temporal_difference(value: np.ndarray, n: int, alpha: float):
    state = START_STATE

    # store the states and rewards experienced
    states = [state]
    rewards = [0]

    # track the time
    time = 0

    T = float('inf')
    while True:
        # go to next time step
        time += 1
        next_state = 0
        if time < T:
            # choose a random action
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1

            # Observe the next reward
            if next_state == 0:
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time  # setting the total time when terminates

        # tau = t-n+1. the time whose state's estimate is being updated
        update_time = time - n
        if update_time >= 0:

            # initialize returns, G
            # G = Rt+1 + gammaRt+2 + ...gamma^n Vt+n-1
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(update_time + n, T) + 1):
                returns += pow(GAMMA, t - update_time - 1) * rewards[t]
            if update_time + n <= T:
                returns += pow(GAMMA, n) * value[states[update_time + n]]
            state_to_update = states[update_time]
            # update the state value
            value[state_to_update] += alpha * (returns - value[state_to_update])
        if update_time == T - 1:
            break
        state = next_state


# 100 runs for each (step, alpha), 10 episodes in each run
def figure7_2():
    # produce the "n" in n-step TD
    steps = np.power(2, np.arange(0, 10))  # [  1   2   4   8  16  32  64 128 256 512]
    alphas = np.arange(0, 1.1, 0.1)
    episodes = 10
    runs = 100

    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                # print('run: ', run, 'step: ', step, 'alpha: ', alpha)
                value = np.zeros(N_STATES + 2)
                for ep in range(0, episodes):
                    temporal_difference(value, step, alpha)
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / N_STATES)

    # take average
    errors /= episodes * runs

    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    figure7_2()
