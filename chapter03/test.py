import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.axes as axes
from matplotlib.table import Table
from typing import List, Optional
from scipy.stats import poisson

import heapq
from copy import deepcopy


class PriorityQueue:
    def __init__(self):
        self.pq = []  # [[priority, self.counter, item], [priority, self.counter, item],...]
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


if __name__ == "__main__":
    priority_queue = PriorityQueue()

    item = (tuple([1, 1]), 1)
    # priority 1: 1
    # priority 2: 2
    priority_queue.add_item(item, -1)
    print(priority_queue.pq)
    print(priority_queue.entry_finder)

    print("cg")
    priority_queue.add_item(item, -2)
    print(priority_queue.pq)
    print(priority_queue.entry_finder)

    print("cg")
    item = (tuple([2, 2]), 1)
    priority_queue.add_item(item, -3)
    print(priority_queue.pq)
    print(priority_queue.entry_finder)

    print(priority_queue.pop_item())
    print(priority_queue.pop_item())
