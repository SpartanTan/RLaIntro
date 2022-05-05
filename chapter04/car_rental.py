from matplotlib import pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import numpy as np
import seaborn as sns
from scipy.stats import poisson
import time

# maximum number of cars in each location
MAX_CARS = 20

# maximum number of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 3

# expectation for number of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for number of cars returned in second location
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a car
MOVE_CAR_COST = 2

# all possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


def expected_return(state: list, action: int, state_value: np.ndarray, constant_returned_cars):
    """calculate the income from rental and return

    :param state: [i, j]
    :param action: n: int
    :param state_value: np.ndarray
    :param constant_returned_cars:
    """
    returns = 0.0

    # initialize total return
    returns -= MOVE_CAR_COST * abs(action)

    # moving cars
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # go through all possible rental requests
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                   poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # valid rental requests should be less than actual number of cars
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:
                # get returned cars, those cars can be used for renting tomorrow
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
                # V(s) = p(s',r|s, pi(s))[r + discount * V(s')]

            else:
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(returned_cars_first_loc, RETURNS_FIRST_LOC) * \
                                      poisson_probability(returned_cars_second_loc, RETURNS_FIRST_LOC)
                        num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        returns += prob * prob_return * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
    return returns


def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)  # move from the first place to second

    iterations = 0
    # 40 20
    ffigure, ax = plt.subplots(2, 3, figsize=(40, 20))  # type: figure.Figure, axes.Axes
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    ax = ax.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap='YlGnBu', ax=ax[iterations])  # type: axes.Axes
        fig.set_ylabel('# cars at second location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max() # delta = max(delta, |v- V(s)|)
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1): # for each s
                old_action = policy[i, j]
                action_returns = []
                for action in actions: # [-5, 5]
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap='YlGnBu', ax=ax[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cats at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1
        # plotting

        # re-drawing the figure
        # ffigure.canvas.draw()
        # # to flush the GUI events
        # ffigure.canvas.flush_events()
        # time.sleep(0.1)
        # end of interaction

        # plt.draw()
        # plt.pause(0.001)
        # plt.clf()
    plt.savefig('figure_4_2.png')
        # plt.close()


if __name__ == '__main__':
    # plt.ion()
    figure_4_2()