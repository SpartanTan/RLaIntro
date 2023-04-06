import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy

import Maze
import trivialModel
import timeModel
from priorityModel import PriorityModel, PriorityQueue

from typing import Union

# Parameters
GRID_DIMS = (6, 9)
POS_START = (5, 3)
POS_GOAL = (0, 8)
POS_OBSTACLES = [(3, j) for j in range(8)]

# [(-1, 0), (0, -1), (0, 1), (1, 0)]
ALL_4_ACTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]

# TD step size
ALPHA = .1  # 0.1
# Discount factor
GAMMA = 0.95
# Exploration ratio
EPSILON = 0.1
# Temporal reward coefficient # time weight
TEMPORAL_RWD_COEF = 1e-5
# Number of step from which the map is modified
N_STEP_CHANGE_MAP = 1000

FIGURE_SIZE = (16, 12)

RANDOM_SEED = 2


class Grid:
    def __init__(self, dims=GRID_DIMS, pos_start=POS_START, pos_goal=POS_GOAL, pos_obstacles=POS_OBSTACLES,
                 fig_size=FIGURE_SIZE, vis_mode='single', modifying_mode='blocking'):
        # grid height and width
        self._h, self._w = dims
        self._pos_start = pos_start
        self._pos_goal = pos_goal
        self._pos_obstacles = pos_obstacles
        assert modifying_mode in ['blocking', 'shortcut']
        self._modifying_mode = modifying_mode
        self._is_map_modified = False

        if vis_mode == 'single':
            self._fig_size = fig_size
            self.fig, self.ax = plt.subplots(1, 1, figsize=self._fig_size)
        elif vis_mode == 'parallel':
            self.fig = plt.figure(figsize=(22, 16), constrained_layout=False)
            self.gs = self.fig.add_gridspec(2, 2)
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[0, 1])
            self.ax3 = self.fig.add_subplot(self.gs[1, :])


class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


def changing_maze(maze: Maze.Maze, dyna_params: DynaParams):
    # set up max steps
    max_steps = maze.max_steps

    # track the cumulative rewards
    rewards = np.zeros((dyna_params.runs, 2, max_steps))

    for run in tqdm(range(dyna_params.runs)):
        models = [trivialModel.TrivialModel(), timeModel.TimeModel(maze, time_weight=dyna_params.time_weight)]

        # initialize state action values
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]

        for i in range(len(dyna_params.methods)):

            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles

            steps = 0
            last_steps = steps
            while steps < maze.max_steps:
                # play for one episode
                steps += dyna_q(q_values[i], models[i], maze, dyna_params)
                # print(steps)
                # update cumulative rewards
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                last_steps = steps

                if steps > maze.obstacle_switch_time:
                    maze.obstacles = maze.new_obstacles

    # averaging over runs
    rewards = rewards.mean(axis=0)

    return rewards


def choose_action(state: list, q_value: np.ndarray, maze: Maze.Maze, dyna_params: DynaParams):
    """
    choose an action following epsilon-greedy policy
    @param state: current state
    @param q_value: state-action value
    @param maze: maze object
    @param dyna_params: parameters
    @rtype: an int, representing one of four actions [0, 1, 2, 3]->[up, down, left, right]
    """
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


def figure_8_4():
    blocking_maze = Maze.Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]

    # step limit
    blocking_maze.max_steps = 3000

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blocking_maze.obstacle_switch_time = 1000

    # set up parameters
    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4

    # play
    rewards = changing_maze(blocking_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.show()


def test():
    grid = Grid(vis_mode='parallel')
    plt.show()


if __name__ == "__main__":
    test()
