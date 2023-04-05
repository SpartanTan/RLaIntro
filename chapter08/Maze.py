class Maze:
    """
    a class to describe the properties of maze
    """

    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9
        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))  # (6,9,4)

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    def step(self, state: list, action: int):
        """
        execute one step in the env, return next state and reward
        @param state: current state
        @param action: action input
        @return: (next_state, reward), ([x, y], reward)
        """
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)

        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0

        return [x, y], reward

    def extend_state(self, state: list, factor: int) -> list:
        """
        Extend a state into factor^2 states.
        For example, given [3,3] with factor 2, output 4 states: [[2, 2], [2, 3], [3, 2], [3, 3]]
        @param state:
        @param factor:
        @return:
        """
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

    def extend_maze(self, factor):
        """
        Extend the original maze into a higher resolution by the factor.
        @param factor: resolution parameter
        @return: return a new maze
        """
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        new_maze.resolution = factor
        return new_maze


if __name__ == "__main__":
    maze = Maze()
    print(maze.q_size)
