import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy

import maze
import trivialModel


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


def choose_action(state: list, q_value: np.ndarray, maze: maze.Maze, dyna_params: DynaParams):
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


def dyna_q(q_value: np.ndarray, model: trivialModel.TrivialModel, maze: maze.Maze, dyna_params: DynaParams):
    """
    @param q_value: state-action function
    @param model: the model to learn
    @param dyna_maze: the maze object
    @param dyna_params: parameters
    @rtype: a float, number of steps the agent experienced in this episode
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


def figure_8_2():
    dyna_maze = maze.Maze()
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


if __name__ == "__main__":
    figure_8_2()
