
from typing import Tuple, List, Optional, Union

import os 
import numpy as np
from math import floor

class Q(object):
    """A Q-table for discrete or continous state space

    Attributes:
        W (np.array): The Q table

    Methods:
        __init__(self, actions: List, discrete: bool=True, 
                 state_shape: Optional[Tuple]=None,
                 segmentation: Optional[List]=None, 
                 init_range: List[int]=[0,1]) -> None
            Initialises the Q table.
        __call__(self, observation: Tuple[int], 
                 action: Union[str, int]) -> float:
            Returns the value of the Q table for the state and action provided.
        update(self, value: float, observation: Tuple[int], 
               action: Union[str, int], lr: float) -> None:
            Updates the Q-table value for the oservation-action to the new value
            with provided lr.

    """
    def __init__(self, actions: List, discrete: bool=True, 
                 state_shape: Optional[Tuple]=None,
                 segmentation: Optional[List]=None, 
                 init_range: List[int]=[0,1]) -> None:
        """Initialises the Q table.

        If the sate is not discrete it is segmented according to the
        segmentation provided.

        Args:
            actions (list): The list of possible actions.
            discrete (bool): If True, the action space is discrete. Otherwise,
                it has to be discretized
            state_shape (tuple): The shape of the state space
            segmentation (list): The boundaries of the boxes for the
                discretization of the continuous state space.
            init_range (list): The range for the initialization of the weight.
            
        Returns:
            None
        """
        self._discrete = discrete
        if self._discrete:
            if isinstance(actions, list):
                self._action_dict = {_:ind for ind,_ in enumerate(actions)}

            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(list(state_shape)+[len(actions)]))

        else:
            shape = [len(_)+2 for _ in segmentation]
            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(shape + [actions]))
            self._segmentation = segmentation

        return None
    
    def __call__(self, observation: Tuple[int], 
                 action: Union[str, int]) -> float:
        """Returns the value of the Q table for the state and action provided.

        Args:
            observaation (tuple): The observation/state
            action (str, int): The action 

        Returns:
            Q-value (float)
        """
        query = self._get_query_(observation, action)

        return self.W[query]
    
    def update(self, value: float, observation: Tuple[int], 
               action: Union[str, int], lr: float) -> None:
        """Updates the Q-table value for the oservation-action to the new value
        with provided lr.

        Args:
            value (float): The new value.
            observaation (tuple): The observation/state.
            action (str, int): The action.
            lr (float): Learning rate.

        Returns:
            None
        """
        query = self._get_query_(observation, action)
        self.W[query] -= lr*(self.W[query]-value)

        return None
    
    def _get_V(self, observation: Optional[Tuple[int]]=None)\
            -> Union[None, float]:
        """If no observation is provided, updates the Value function, else returns the
        required value function for the provided observation.

        Args:
            observaation (tuple): The observation/state.

        Returns:
            (float, None)
        """
        if not observation:
            self.V = np.max(self.W, -1)

        else:
            query = self._get_query_(observation)
            return np.max(self.W[query], -1)

    def _segment(self, observation: Tuple[float], 
                 action: Optional[Union[str, int]]) -> Tuple[int]:
        """Finds the position in the discret Q table associated with the oservation

        Args:
            observation (float): raw observation
            action (int, str)

        Return:
            position (tuple)
        """
        query = [] 
        for _, segm in enumerate(self._segmentation):
            if len(segm) > 0:
                query.append(sum([s < observation[_] for s in segm]))

            else:
                query.append(0)
            
        if not action is None:
            query.append(action)

        return tuple(query)
    
    def _get_query_(self, observation: Tuple[int], 
                 action: Optional[Union[str, int]]=None) -> Tuple[int]:
        """Returns the position in the Q-table for any input

        Args:
            observation (float): raw observation
            action (int, str)

        Return:
            position (tuple)
        """
        if self._discrete:
            if hasattr(self, "_action_dict") and not action is None:
                action = self._action_dict[action]

            if action:
                return tuple(observation + [action])

            else:
                return tuple(observation)

        else:
            return self._segment(observation, action)

