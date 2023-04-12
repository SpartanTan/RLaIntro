import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# two actions
ACTIONS = [0, 1]

# each transition has a probability to terminate with 0
TERMINATION_PROB = 0.1

# maximum expected updates
MAX_STEPS = 20000

# epsilon greedy
EPSILON = 0.1


class Task:
    def __init__(self, n_states: int, b: int):
        """

        @param n_states: number of non-terminal states
        @param b:
        """
        self.n_states = n_states
        self.b = b

        # all possible transitions
        # size is n_states * 2 * b
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))
        self.reward = np.random.randn(n_states, len(ACTIONS), b)

    def step(self, state, action):
        """
        perform one step by given state and action
        @param state:
        @param action:
        @return: next_state, reward. two integers. When terminate, will return (1000, 0)
        """
        if np.random.rand() < TERMINATION_PROB:
            return self.n_states, 0

        next_ = np.random.randint(self.b)
        return self.transition[state, action, next_], self.reward[state, action, next_]


def argmax(value):
    """
    For calculating true value. Greedy policy
    @param value: [2x1] ndarray, two values of two actions in one state
    @return: index of the chosen action that is greedy, 0 or 1
    """
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


def evaluate_pi(q, task):
    """
    Evaluate the value of the start state for the greedy policy derived from @q under the MDP @task

    @param q: [1000x2] ndarray
    @param task:
    """
    runs = 1000
    returns = []
    for r in range(runs):
        rewards = 0
        state = 0
        while state < task.n_states:
            # when terminate, step() will return (1000, 0) thus the condition will not be fulfilled
            action = argmax(q[state])
            state, r = task.step(state, action)
            rewards += r
        returns.append(rewards)
    return np.mean(returns)


def uniform(task: Task, eval_interval):
    """
    perform expected update from a uniform state-action distribution of the MDP task
    evaluate the learned q value every eval_interval steps

    @param task:
    @param eval_interval:
    @return: (steps, values) like (1,2,3,4), (3214, 1234,5125..)
    """
    performance = []
    q = np.zeros((task.n_states, 2))
    for step in tqdm(range(MAX_STEPS)):
        # cycle through all possible state-action pairs
        state = step // len(ACTIONS) % task.n_states  # 0, 0, 1, 1, 2, 2, ...
        action = step % len(ACTIONS)  # 0, 1, 0, 1, 0, 1,...

        next_states = task.transition[state, action]  # [bx2] ndarray

        # expected return
        q[state, action] = (1 - TERMINATION_PROB) * (
                task.reward[state, action] + np.max(q[next_states, :], axis=1))

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])

    return zip(*performance)


def on_policy(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    # always start from start state
    state = 0
    for step in tqdm(range(MAX_STEPS)):
        # epsilon-greedy policy
        # on-policy
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q[state])

        next_state, _ = task.step(state, action)

        next_states = task.transition[state, action]

        q[state, action] = (1 - TERMINATION_PROB) * np.mean(
            task.reward[state, action] + np.max(q[next_states, :], axis=1))

        if next_state == task.n_states:
            next_state = 0
        state = next_state

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])

    return zip(*performance)


def figure_8_8():
    # number of total states
    num_states = [1000, 10000]

    # number of branches
    branch = [1, 3, 10]
    methods = [on_policy, uniform]

    # average across 30 tasks
    n_tasks = 30

    # number of evaluation points
    x_ticks = 100

    for i, n in enumerate(num_states):
        plt.subplot(2, 1, i + 1)
        for b in branch:
            tasks = [Task(n, b) for _ in range(n_tasks)]
            for method in methods:
                steps = None
                value = []
                for task in tasks:
                    steps, v = method(task, MAX_STEPS / x_ticks)
                    value.append(v)
                value = np.mean(np.asarray(value), axis=0)
                plt.plot(steps, value, label=f'b = {b}, {method.__name__}')
        plt.title(f'{n} states')
        plt.ylabel('value of start state')
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('computation time, in expected updates')
    plt.show()


if __name__ == "__main__":
    figure_8_8()
