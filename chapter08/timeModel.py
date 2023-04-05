import numpy as np
from copy import deepcopy
import Maze


class TimeModel:
    def __init__(self, maze: Maze.Maze, time_weight=1e-4, rand=np.random):
        self.model = dict()
        self.rand = rand

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    def feed(self, state: list, action: int, next_state: list, reward):
        """
        Feed the model with previous experience.
        Will be called every time after an action is taken at a state, and receive next state and reward from env.
        @param state:
        @param action:
        @param next_state:
        @param reward:
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1  # time step of a real interaction with env
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    # Add all possible actions into model, so that these will be considered in planning step
                    self.model[tuple(state)][action_] = [list(state), 0, 1]
        self.model[tuple(state)][action] = [list(next_state), reward, self.time]

    def sample(self) -> tuple:
        """
        randomly sample from preivous experience
        @return: tuple, (list(state), action, list(next_state), reward)
        """
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]  # it is a tuple instead of a list
        action_index = self.rand.choice(range(len(self.model[tuple(state)].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last visit
        # self.time: the current time step
        # time: the time step of this action taken in the real env last time
        # This line gives small credits to those actions that has not been tried for a long time
        # So they become slightly more important when adjusting the policy
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward
