import heapq

import numpy as np
from trivialModel import TrivialModel
from copy import deepcopy


class PriorityQueue:
    def __init__(self):
        self.pq = []  # the queue stores a lot of [priority, self.counter, item], prioritized by the priority
        self.entry_finder = {}  # not know what for
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item: tuple, priority=0):
        """

        @param item: a tuple of state-action pair, (tuple(state), action)
        @param priority: inverse of priority(negative number)
        """
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1  # track the number of items added into the entry
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        """
        Find this state-action pair from entry_finder: dict, then mark the item with REMOVED
        @param item: (tuple(state), action)
        """
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        """
        return one state-action pair and its priority
        @return: ((tuple(state), action)), priority)
        """
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        """
        return 0 if entry_finder is not empty
        return 1 if it is empty
        @return:
        """
        return not self.entry_finder


class PriorityModel(TrivialModel):
    def __init__(self, rand=np.random):
        TrivialModel.__init__(self, rand)
        self.priority_queue = PriorityQueue()
        # {tuple(next_state1): {SApair1, SApira2}, tuple(next_state2): {SApair3, ...}}
        # SApair = (tuple(state), action)
        self.predecessors = dict()

    def insert(self, priority, state, action):
        """
        Add a @state-@action pair into the priority queue with priority @priority
        @param priority:
        @param state:
        @param action:
        """
        # the item added into the queue is a big tuple of state-action pair: (tuple(state), action)
        # since the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((tuple(state), action), -priority)

    def feed(self, state: list, action: int, next_state: list, reward: int):
        """
        Feed the model with previous experience.
        Also log the current state-action pair as the predecessor of the next state
        @param state:
        @param action:
        @param next_state:
        @param reward:
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        TrivialModel.feed(self, state, action, next_state, reward)

        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        # add a state-action pair into the set
        # {tuple(next_state): {SApair1, SApira2}}
        self.predecessors[tuple(next_state)].add((tuple(state), action))

    def sample(self):
        """
        Get the first item in the priority queue.
        Sample from the priority queue. The state-action pair with highest priority will be simulated first
        @return: (-priority, list(state), action, list(next_state), reward)
        """
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    def predecessor(self, state):
        """
        Find and return all predecessors of the current state.
        S,A->S',R;
        Get S,A,R and stack into a list then return.
        @param state:
        @return: a list of (tuple(state_pre), action_pre, reward)
        """
        # check if this state has predecessors
        # if this state has been visited before, it will have at least one predecessor
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors

    def empty(self):
        """
        return 0 if queue is not empty
        return 1 if it is empty
        @return:
        """
        return self.priority_queue.empty()
