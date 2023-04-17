import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor
from tile import IHT, tiles

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1

# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0


class ValueFunction:
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    def get_active_tiles(self, position, velocity, action):
        """
        get the indices of the current state in all the tilings
        @param position:
        @param velocity:
        @param action:
        @return:
        """
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                             [self.position_scale * position, self.velocity_scale * velocity], [action])
        return active_tiles

    def value(self, position, velocity, action):
        """
        get the current state-action pair value
        @param position:
        @param velocity:
        @param action:
        @return:
        """
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    def learn(self, position, velocity, action, target):
        """
        update weights
        @param position:
        @param velocity:
        @param action:
        @param target:
        """
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    def cost_to_go(self, position, velocity):
        """
        get # of steps to reach the goal under current state value function
        @param position:
        @param velocity:
        @return:
        """
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)


def get_action(position, velocity, value_function):
    """
    epsilon-greedy policy.
    @param position:
    @param velocity:
    @param value_function:
    @return:
    """
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(value_function.value(position, velocity, action))
    # action is -1 or 0 or 1, index from enumerate starts from 0, thus -1
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)]) - 1


def step(position, velocity, action):
    """
    execute one step
    @param position:
    @param velocity:
    @param action:
    @return: (new_position, new_velocity, reward)
    """
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward


def semi_gradient_n_step_sarsa(value_function: ValueFunction, n=1):
    """
    n-step Sarsa using tiling coding
    @param value_function:
    @param n: number of steps, default 1
    """
    current_position = np.random.uniform(-0.6, -0.4)
    # initial velocity
    current_velocity = 0.0
    # get initial action
    current_action = get_action(current_position, current_velocity, value_function)

    # track previous position, velocity, action and reward
    positions = [current_position]
    velocities = [current_velocity]
    actions = [current_action]
    rewards = [0.0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')

    while True:
        # go to next time step
        time += 1

        if time < T:
            # take current action and go to the new state
            new_position, new_velocity, reward = step(current_position, current_velocity, current_action)
            new_action = get_action(new_position, new_velocity, value_function)

            # track new state and action
            positions.append(new_position)
            velocities.append(new_velocity)
            actions.append(new_action)
            rewards.append(reward)

            if new_position == POSITION_MAX:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]

            # add estimated state action value to the return
            if update_time + n <= T:
                returns += value_function.value(positions[update_time + n],
                                                velocities[update_time + n],
                                                actions[update_time + n])
            # update the state value function
            if positions[update_time] != POSITION_MAX:
                value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)
        if update_time == T - 1:
            break
        current_position = new_position
        current_velocity = new_velocity
        current_action = new_action
    return time


def print_cost(value_function, episode, ax):
    grid_size = 40
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    # positionStep = (POSITION_MAX - POSITION_MIN) / grid_size
    # positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    # velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / grid_size
    # velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
    axis_x = []
    axis_y = []
    axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))

    ax.scatter(axis_x, axis_y, axis_z, s=10)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))


def figure_10_1():
    episodes = 9000
    plot_episodes = [0, 99, episodes - 1]
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i + 1, projection='3d') for i in range(len(plot_episodes))]
    num_of_tilings = 8
    alpha = 0.3

    value_function = ValueFunction(alpha, num_of_tilings)
    for ep in tqdm(range(episodes)):
        semi_gradient_n_step_sarsa(value_function)
        if ep in plot_episodes:
            print_cost(value_function, ep, axes[plot_episodes.index(ep)])

    plt.show()


def figure_10_2():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.1, 0.2, 0.5]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index])
                steps[index, episode] += step
    steps /= runs
    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='alpha = ' + str(alphas[i]) + '/' + str(num_of_tilings))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.show()


def figure_10_3():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.5, 0.3]
    n_steps = [1, 8]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index], n_steps[index])
                steps[index, episode] += step

    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = %.01f' % (n_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.show()


def figure_10_4():
    alphas = np.arange(0.25, 1.75, 0.25)
    n_steps = np.power(2, np.arange(0, 5))
    episodes = 50
    runs = 5

    max_steps = 300
    steps = np.zeros((len(n_steps), len(alphas)))
    for run in range(runs):
        for n_step_index, n_step in enumerate(n_steps):
            for alpha_index, alpha in enumerate(alphas):
                if (n_step == 8 and alpha > 1) or \
                        (n_step == 16 and alpha > 0.75):
                    # In these cases it won't converge, so ignore them
                    steps[n_step_index, alpha_index] += max_steps * episodes
                    continue
                value_function = ValueFunction(alpha)
                for episode in tqdm(range(episodes)):
                    step = semi_gradient_n_step_sarsa(value_function, n_step)
                    steps[n_step_index, alpha_index] += step

    # average over independent runs and episodes
    steps /= runs * episodes

    for i in range(0, len(n_steps)):
        plt.plot(alphas, steps[i, :], label='n = ' + str(n_steps[i]))
    plt.xlabel('alpha * number of tilings(8)')
    plt.ylabel('Steps per episode')
    plt.ylim([220, max_steps])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # figure_10_1()
    figure_10_2()
