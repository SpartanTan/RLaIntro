import numpy as np
from copy import deepcopy


class TrivialModel:
    """
    A base class to describe a model
    """

    def __init__(self, rand=np.random):
        # {tuple(state): {[action]: [list(next_state), reward]}, ...}
        self.model = dict()
        self.rand = rand

    def feed(self, state: list, action: int, next_state: list, reward: int):
        """
        register the next state and reward to the model.
        self.model is a dictionary, maintaining the state transition for each state-action pair
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    def sample(self):
        """
        sample one state-action pair and get the corresponding next state and reward from model
        @rtype: (list(state), action, list(next_state), reward)
        """
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]  # turn the model dictionary into a list of only states, then chose
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward
