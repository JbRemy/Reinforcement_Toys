from .Q import Q
from .V import V

from typing import Union, Type, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import operator

class  GridWorld(object):
    """A class to generate and interact with a gridworld environement

    Attributes:
        Q (Q): The empirical Q function of the environement.
        V (V): The empirical Value function of the environement.
        x, y (int): Coordinates of the agent in the grid.
        t (int): The time stamp in the current run.
        path (list): The path taken during the current run

    Methods:
        reset(self, random_init: bool=False, state: Optional[Tuple[int]]=None)
              -> None:
            Resets the agent to the start/random/required position and time to 0.
        step(self, action: Union[str, int]) 
             -> Tuple[Union[Tuple[int], int, bool, None]]:
            Perfomrs the asked action
        render_path_and_V(self, scale: int=1.5) -> None:
            Plots the path taken by the agent and the Value function.
        render_board(self, scale= float:1.5, show: bool=True,
                     fig: bool=False) -> None:
            displays a figure with the labirynth.
        render_path(self, fig: bool=False, show: bool=True) -> None:
            Plots the path taken by the agent up to now
        render_V(self, fig: bool=False, show: bool=True,from: str='Q') -> None:
            Plots the Value function

    Constants:
        ACTION_SPACE (list): The possible actions in int
        ACTION_DICT (dict): The possible actions in strings
            "up", "right", "down", "left"
    """
    ACTION_SPACE = [8,4,2,1]
    ACTION_DICT = {"up": 8, "right":4, "down": 2, "left": 1}
    _TRACE = [[[0,1], [0,0]], [[1,1], [0,1]], [[0,1], [1,1]], [[0,0], [0,1]]]
    def __init__(self, grid: Union[str, tuple]) -> None:
        """Initializes the grid

        If a filename is passed then the grid is parsed, otherwise, if a tuple
        is provided, then a random grid is generated.
        Also initialises a Q and a Value function for the grid

        Args:
            grid (str, tuple): A file name or the size of the requested grid.
                The grid is defined with the following synthax:
                    000000 (0) blank cell
                    001000 (8) wall up
                    000100 (4) wall right
                    000010 (2) wall down 
                    000001 (1) wall left 
                    010000 (16) start
                    100000 (32) end
                Any combination of those basics cells are possible
                Hence 100101=37 is an end cell with a wall to the right and 
                a wall to the left

        Returns:
            None
        """
        assert isinstance(grid, str) or isinstance(grid, tuple), \
            "Grid must be a file or tuple to generate a random grid"
        if isinstance(grid, str):
            with open(grid, "r") as f:
                self._grid = np.array([[int(_) for _ in line.split(";")] for\
                                      line in f.readlines()])
        
        elif isinstance(grid, tuple):
            self._grid = self._generate_grid_(grid)

        self._start_coordinates = np.where([[self.count_ones_binary(_+16) ==\
                                            self.count_ones_binary(_) for _ in\
                                            line] for line in self._grid])
        self._end_coordinates = np.where([[_>=32 for _ in line] for line in self._grid])
        self.Q = Q(list(self.ACTION_DICT.keys()), state_shape=self._grid.shape)
        self.V = V(state_shape=self._grid.shape)
        self.reset()

        return None

    def reset(self, random_init: bool=False, 
              state: Optional[Tuple[int]]=None) -> Tuple[int]:
        """Resets the agent to the start/random/required position and time to 0.

        Args:
            random_init (bool): If True inits to a random position, otherwise
                to the start coordinates.
            state (tuple): The state to reinitialise

        Returns:
            state (tuple): posiiton of the agent
        """
        if not random_init:
            if not state is None: 
                self.y, self.x = state
            else:
                self.x = self._start_coordinates[1][0]
                self.y = self._start_coordinates[0][0]

        else:
            self._current_cell = 0
            while self._current_cell == 0:
                self.x = np.random.randint(low=0, high=self._grid.shape[1])
                self.y = np.random.randint(low=0, high=self._grid.shape[0])

        self._current_cell = self._grid[self.y, self.x]

        self.t = 0
        self.path = []

        return (self.y, self.x)

    def step(self, action: Union[str, int])\
            -> Tuple[Union[Tuple[int], int, bool, None]]:
        """Perfomrs the asked action

        Checks if the action is possible, and perfoms it, the agent stays in
        place. Also checks the reward. The reward function is: 1 if done, 0
        otherwise.

        Args:
            action (str, int): The action to perform. 

        Return:
            state (tuple): The new position of the agent.
            reward (int): The reward of the action
            done (True): If True, the episode is over.
            None
        """
        if isinstance(action, str):
            action = self.ACTION_DICT[action]

        if not self._check_action_(action):
            pass

        else:
            x_moove, y_moove = self.action_results_(action)
            self.x += x_moove
            self.y += y_moove

        self.path.append([self.x, self.y])

        self.t += 1
        self._current_cell = self._grid[self.y, self.x]
        done = self._check_terminate_()
        if done:
            reward = 1 
        else :
            reward = 0

        return (self.y, self.x), reward, done, None
        
    def render_path_and_V(self, scale: int=1.5) -> None:
        """Plots the path taken by the agent and the Value function.

        Args:
            scale (float): size of the figures
        """
        shape = tuple([_*0.5 for _ in self._grid.shape])
        plt.figure(figsize=(shape[1]*scale*2+1, shape[0]*scale))
        plt.subplot(1,2,1)
        self.render_path(fig=True, show=False)
        plt.subplot(1,2,2)
        self.render_V(fig=True, show=False)
        plt.show()

        return None

    def render_board(self, scale: float=1.5, show: bool=True,
                     fig: bool=False) -> None:
        """displays a figure with the labirynth.

        Args:
            scale (float): size of the figures
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.

        return:
            None
        """
        if not fig:
            shape = tuple([_*0.5 for _ in self._grid.shape])
            plt.figure(figsize=(shape[1]*scale, shape[0]*scale))
        
        plt.ylim(self._grid.shape[0]+0.1, -0.1)
        plt.xlim(-0.1,self._grid.shape[1]+0.1)

        for i in range(self._grid.shape[0]):
            for j in range(self._grid.shape[1]):
                for ind,_ in enumerate(bin(256+self._grid[i,j])[-4:]):
                    if _ == "1": 
                        x_modif, y_modif = self._TRACE[ind]
                        x = [j+x_modif[0], j+x_modif[1]]
                        y = [i+y_modif[0], i+y_modif[1]]
                        plt.plot(x,y,"k-", linewidth=5)

        plt.text(x=self._start_coordinates[1]+0.5,
                 y=self._start_coordinates[0]+0.5, s="START",
                    bbox={'facecolor':'purple','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='white')
        plt.text(x=self._end_coordinates[1]+0.5,
                 y=self._end_coordinates[0]+0.5, s="END",
                    bbox={'facecolor':'purple','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='white')

        if show:
            plt.show()

        return None

    def render_path(self, fig: bool=False, show: bool=True) -> None:
        """Plots the path taken by the agent up to now

        Args:
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.

        return:
            None
        """
        self.render_board(show=False, fig=fig)
        current_pos = (self._start_coordinates[1][0], self._start_coordinates[0][0])
        for next_pos in self.path:
            x = [current_pos[0]+1/2, next_pos[0]+1/2]
            y = [current_pos[1]+1/2, next_pos[1]+1/2]
            plt.plot(x,y,"b-", linewidth=5)
            current_pos = next_pos

        plt.title("Path taken by the agent, %i steps taken." % len(self.path))

        if show:
            plt.show()

        return None

    def render_V(self, fig: bool=False, show: bool=True, f: str='Q') -> None:
        """Plots the Value function

        Args:
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.
            f (str): from which source to plot the Value function

        return:
            None
        """
        self.Q.get_V()
        self.render_board(show=False, fig=fig)
        if f == 'Q':
            plt.pcolormesh(self.Q.V, cmap="hot")

        if f == 'V':
            plt.pcolormesh(self.V.W, cmap="hot")

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("State of the value function")

        if show:
            plt.show()

        return None

    def _check_action_(self, action: int) -> bool:
        """Checks if the proposed action is possible to execute

        Args:
            action (int): The action to perform

        Return:
            None
        """
        return self.count_ones_binary(self._current_cell + action) ==\
                self.count_ones_binary(self._current_cell) + 1
    
    def _check_terminate_(self) -> bool:
        """If the state is terminal returns True, False otherwise

        Returns:
            terminate (bool)
        """
        return self._current_cell >= 32

    @staticmethod
    def action_results_(action: int) -> Tuple[int]:
        """ returns the state modification of an action

        Args:
            action (str, int): The action to perform. 

        Returns:
            state modif (tuple)
        """
        if action == 8:
            return 0, -1

        elif action == 4:
            return 1, 0

        elif action == 2:
            return 0, 1

        elif action == 1:
            return -1, 0

    @staticmethod
    def count_ones_binary(number: int) -> int:
        """Counts the number of ones in the binary writting of a number

        Args:
            number (int)

        Return:
            result (int)
        """
        return len([one for one in bin(number)[2:] if one=="1"])

    @staticmethod
    def _generate_grid_(size: Tuple[int]) -> np.array:
        """Generates a random grid
        """
        grid = np.zeros(shape=size, dtype=int)
        visited = grid != 0

        start_x = np.random.randint(0, size[1])
        grid[:,:] += 15
        grid[-1, start_x] += 16

        visited[-1,start_x] = True 

        cell = (size[0]-1, start_x)
        stack = [cell]
        while len(stack) > 0:
            unvisited_neighbors = [pos for pos in
                                   GridWorld._get_neighbors_(cell, clip=size) if not
                                   visited[pos]]
            if len(unvisited_neighbors) > 0:
                next_cell = unvisited_neighbors[np.random.randint(len(unvisited_neighbors))]
                cell_modif, next_cell_modif = GridWorld._get_wall_(cell, next_cell)
                grid[cell] -= cell_modif
                grid[next_cell] -= next_cell_modif
                stack.append(next_cell)
                cell = next_cell
                visited[cell] = True
                last_visited = cell


            else:
                cell = stack.pop(np.random.randint(len(stack)))

        grid[last_visited] += 32
        return grid
                
    @staticmethod
    def _get_wall_(state_1: Tuple[int], state_2: Tuple[int]) -> Tuple[int]:
        """ returns the wall corresponding to the transition from state one to state two
        """
        transition = tuple(map(operator.sub, state_2, state_1))
        if transition == (0, -1):
            return 1, 4

        elif transition == (1,0):
            return 2, 8

        elif transition == (0, 1):
            return 4, 1

        elif transition == (-1, 0):
            return 8, 2

    @staticmethod
    def _get_neighbors_(cell, clip):
        return [pos for pos in [tuple(map(operator.add, cell, (1,0))), 
                tuple(map(operator.add, cell, (-1,0))), 
                tuple(map(operator.add, cell, (0,1))),
                tuple(map(operator.add, cell, (0,-1)))] if\
                    all([0 <= _ < s for _,s in zip(pos, clip)])]






        

