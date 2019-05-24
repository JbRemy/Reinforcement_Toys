from Q import Q

import matplotlib.pyplot as plt
import numpy as np

class  GridWorld(object):
    def __init__(self, grid=None):
        """
        Args:
            > grid: (np.array) defines the labyrinth.
                in binary:
                    000000 (0) blank cell
                    001000 (8) wall up
                    000100 (4) wall right
                    000010 (2) wall down 
                    000001 (1) wall left 
                    010000 (16) start
                    100000 (32) end
                Any combination of those basics cells are possible
                Hence 100101=37 is an end cell with a wall right and a wall left
        """
        assert isinstance(grid, str), "Random generation of a grid not\
                implemented yet, please define a grid."
        if isinstance(grid, str):
            with open(grid, "r") as f:
                self.grid = np.array([[int(_) for _ in line.split(";")] for\
                                      line in f.readlines()])

        self.action_space = [8,4,2,1]
        self.action_dict = {"up": 8, "right":4, "down": 2, "left": 1}
        self.start_coordinates = np.where([[self.count_ones_binary(_+16) ==\
                                            self.count_ones_binary(_) for _ in\
                                            line] for line in self.grid])
        self.Q = Q(list(self.action_dict.keys()), state_shape=self.grid.shape)
        self.reset()

    def reset(self, random_init=False):
        """
        Resets the agent to the start position and time to 0.
        """
        if not random_init:
            self.x = self.start_coordinates[1][0]
            self.y = self.start_coordinates[0][0]
            self.current_cell = self.grid[self.y, self.x]

        else:
            self.current_cell = 0
            while self.current_cell == 0:
                self.x = np.random.randint(low=0, high=self.grid.shape[1])
                self.y = np.random.randint(low=0, high=self.grid.shape[0])
                self.current_cell = self.grid[self.y, self.x]

        self.t = 0
        self.path = []

        return [self.y, self.x]

    def step(self, action):
        """
        Perfomrs the asked action
        Args:
            > action: (int) The action to perdorm.
                1000 up, 0100 right, 0010 down, 0001 left
                      (str) "up", "right", "down", "left"
                will be converted to int
        """
        if isinstance(action, str):
            action = self.action_dict[action]

        if not self.check_action_(action):
            pass

        else:
            x_moove, y_moove = self.action_results_(action)
            self.x += x_moove
            self.y += y_moove

        self.path.append([self.x, self.y])
        self.t += 1
        self.current_cell = self.grid[self.y, self.x]
        done = self.check_terminate_()

        return [self.y, self.x], -1, done
        
    def check_action_(self, action):
        """
        Checks if the proposed action is possible to execute
        Args: see self.step
        """
        return self.count_ones_binary(self.current_cell + action) ==\
                self.count_ones_binary(self.current_cell) + 1
    
    def check_terminate_(self):
        """
        If the state is terminal returns True, False otherwise
        """
        return self.current_cell >= 32

    def render_path_and_V(self, scale=1.5):
        shape = tuple([_*0.5 for _ in self.grid.shape])
        plt.figure(figsize=(shape[1]*scale*2+1, shape[0]*scale))
        plt.subplot(1,2,1)
        self.render_path(fig=True, show=False)
        plt.subplot(1,2,2)
        self.render_V(fig=True, show=False)
        plt.show()

    def render_board(self, scale=1.5, show=True, fig=False):
        """
        displays a figure with the labirynth.
        Args
        """
        if not fig:
            shape = tuple([_*0.5 for _ in self.grid.shape])
            plt.figure(figsize=(shape[1]*scale, shape[0]*scale))
        
        plt.ylim(self.grid.shape[0]+0.1, -0.1)
        plt.xlim(-0.1,self.grid.shape[1]+0.1)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                for ind,_ in enumerate(bin(256+self.grid[i,j])[-4:]):
                    if _ == "1": 
                        x_modif, y_modif = self.get_trace_(ind)
                        x = [j+x_modif[0], j+x_modif[1]]
                        y = [i+y_modif[0], i+y_modif[1]]
                        plt.plot(x,y,"k-", linewidth=5)

        if show:
            plt.show()

    def render_path(self, fig=False, show=True):
        """
        Plots the path taken by the agent up to now
        """
        self.render_board(show=False, fig=fig)
        current_pos = (self.start_coordinates[1][0], self.start_coordinates[0][0])
        for next_pos in self.path:
            x = [current_pos[0]+1/2, next_pos[0]+1/2]
            y = [current_pos[1]+1/2, next_pos[1]+1/2]
            plt.plot(x,y,"b-", linewidth=5)
            current_pos = next_pos

        plt.title("Path taken by the agent, %i steps taken." % len(self.path))

        if show:
            plt.show()

    def render_V(self, fig=False, show=True):
        """
        plots the Q map
        """
        self.Q.get_V()
        self.render_board(show=False, fig=fig)
        plt.pcolormesh(self.Q.V, cmap="hot")
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("State of the value function")

        if show:
            plt.show()


    @classmethod
    def get_trace_(cls, n):
        """
        returns the path to trace the line [x_start, x_end], [y_start, y_end]
        Args:
            > n: (int) a number between 0 and 3
        """
        if n==0:
            return [0,1], [0,0]

        elif n==1:
            return [1,1], [0,1]

        elif n==2:
            return [0,1], [1,1]

        elif n==3:
            return [0,0], [0,1]


    @classmethod
    def action_results_(cls, action):
        """
        returns the state modification of an action
        Args: see self.step
        """
        if action == 8:
            return 0, -1

        elif action == 4:
            return 1, 0

        elif action == 2:
            return 0, 1

        elif action == 1:
            return -1, 0

    @classmethod
    def count_ones_binary(cls, number):
        return len([one for one in bin(number)[2:] if one=="1"])






        

