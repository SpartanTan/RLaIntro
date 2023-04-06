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


def dyna_q(q_value: np.ndarray, model: trivialModel.TrivialModel, maze: Maze.Maze, dyna_params: DynaParams):
    """
    Dyna-Q loop. Start from S and end at G or exceeds the max steps. Return the number of steps the agent experienced in this episode.
    @param q_value: state-action function
    @param model: the model to learn
    @param maze: the maze object
    @param dyna_params: parameters
    @return: steps, number of steps the agent experienced in this episode
    @rtype: int
    """
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-learning update
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :])
                                 - q_value[state[0], state[1], action])

        # feed the model with the experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :])
                                     - q_value[state_[0], state_[1], action_])

        state = next_state

        if steps > maze.max_steps:
            break
    return steps


def prioritized_sweeping(q_value: np.ndarray, model: Union[trivialModel.TrivialModel, PriorityModel], maze: Maze.Maze,
                         dyna_params: DynaParams):
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # get the priority for current state action pair
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :])
                          - q_value[state[0], state[1], action])

        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < dyna_params.planning_steps and not model.empty():
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state-action value for the sample pair
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample states
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the number of backups
        backups += planning_step + 1

    return backups


def check_path(q_value: np.ndarray, maze: Maze.Maze):
    """
    check if q_value is optimal by letting a new agent follow the optimal path from q_value. If the
    agent reach GAOL in max_steps then it's optimal.
    @param q_value:
    @param maze:
    @return: True if optimal, False if not optimal
    """
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_value[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True


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


def figure_8_2():
    dyna_maze = Maze.Maze()
    dyna_params = DynaParams()

    runs = 10
    episodes = 50
    planning_steps = [0, 5, 50]

    # a table records the number of steps each agent experienced in each episode
    #   1 2 3 ... 50
    # 0 ......
    # 5 ......
    # 50 .. ....
    steps = np.zeros((len(planning_steps), episodes))  # (3x50)

    for run in tqdm(range(0, runs)):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            model = trivialModel.TrivialModel()
            for ep in range(episodes):
                # save the steps experienced in one episode
                steps[i, ep] += dyna_q(q_value, model, dyna_maze, dyna_params)

    # averaging between runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))

    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()


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


def figure_8_5():
    # setup a shorcut maze
    shortcut_maze = Maze.Maze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcut_maze.max_steps = 6000

    shortcut_maze.obstacle_switch_time = 3000

    # set up parameters
    dyna_params = DynaParams()

    # 50-step planning
    dyna_params.planning_steps = 50
    dyna_params.runs = 5
    dyna_params.time_weight = 1e-3
    dyna_params.alpha = 1.0

    # play
    rewards = changing_maze(shortcut_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.show()


def example_8_4():
    original_maze = Maze.Maze()  # 9x6

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna]

    # set up models for planning
    models = [PriorityModel, trivialModel.TrivialModel]

    # setup methods names
    method_names = ['Prioritized Sweeping', 'Dyna-Q']

    # number of different mazes
    num_of_mazes = 5

    # mazes
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    # A list of two different algorithm functions
    methods = [prioritized_sweeping, dyna_q]

    # run times
    runs = 5

    # track the number of backups
    # which_run, which_method, which_maze
    backups = np.zeros((runs, 2, num_of_mazes))
    for run in tqdm(range(runs)):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in enumerate(mazes):
                print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    steps.append(methods[i](q_value, model, maze, params[i]))

                    # check whether the optimal path is found
                    if check_path(q_value, maze):
                        break
                backups[run, i, mazeIndex] = np.sum(steps)
    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    # because the steps in Dyna-Q are not logged
    backups[1, :] *= params_dyna.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # figure_8_2()  # maze
    # figure_8_4()  # blocking maze
    # figure_8_5()  # shortcut maze
    example_8_4()
