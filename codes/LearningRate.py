
from typing import Union, Tuple, List, Optional

from math import floor
import numpy as np

class LearningRate(object):
    """Implementation of a learning rate with indepandent decay for every
    elements

    Methods:
        __init__(self, lr0: float, decay: float, actions_size: int, 
                     discrete: bool=True, state_shape: Optional[Tuple]=None,
                     segmentation: Optional[List]=None, 
                     min_lr: float=0.1) -> None:
            Initializes the learning rate
        __call__(self, observation: Union[int, np.array], 
                     action: Union[str, int]) -> float:
            Returns the current value of the learning rate for the requiered
            entry in the Q-table
    """
    def __init__(self, lr0: float, decay: float, actions_size: int, 
                 discrete: bool=True, state_shape: Optional[Tuple]=None,
                 segmentation: Optional[List]=None, 
                 min_lr: float=0.1) -> None:
        """Initializes the learning rate

        Args:
            lr0 (float): Initialisation value of the learning rate
            decay (float): Decay of the learning rate for every visit
            actions_size (int): size of the action space
            discrete (bool): If True, the action space is discrete. Otherwise,
                it has to be discretized
            state_shape (tuple): The shape of the state space
            segmentation (list): The boundaries of the boxes for the
                discretization of the continuous state space.
            min_lr (float): The minimum value of the learning rate

        Returns:
            None
        """
        self._discrete = discrete
        self._decay = decay
        self._lr0 = lr0
        self._min_lr = min_lr
        if isinstance(actions_size, list):
            self._action_dict = {_:ind for ind,_ in enumerate(actions_size)}

        if self._discrete:
            self._lr = np.ones(shape=tuple(list(state_shape)+[len(actions_size)]))
            self._t = np.zeros(shape=tuple(list(state_shape)+[len(actions_size)]))
            self._lr *= lr0

        else:
            shape = [len(segm)+2 for segm in segmentation]
            self._lr = np.ones(shape=tuple(shape + [actions_size]), dtype=np.float)
            self._lr *= lr0
            self._t = np.zeros(shape=tuple(shape + [actions_size]))
            self._segmentation = segmentation
    
    def __call__(self, observation: Union[int, np.array], 
                 action: Union[str, int]) -> float:
        """Returns the current value of the learning rate for the requiered
        entry in the Q-table

        Args:
            observation (int, np.array): A state
            action (str, int): an element of the action space

        Returns:
            learning_rate (float)
        """
        query = self._get_query(observation, action)
        self._lr[query] = max(self._lr0/(1+self._t[query]*self._decay), self._min_lr)
        self._t[query] += 1

        return self._lr[query]
    
    def _get_query(self, observation: Union[Tuple, List], 
                   action: Optional[Union[str, int]]=None):
        """See Q._get_query
        """
        if not isinstance(observation, list):
            observation = list(observation)

        if self._discrete:
            if hasattr(self, "_action_dict") and not action is None:
                action = self._action_dict[action]

            if not action is None:
                return tuple(observation + [action])

        else:
            return self._segment(observation, action)

    def _segment(self, observation: np.array, 
                 action: Union[str, int]) -> Tuple[int]:
        """See Q.segment
        """
        query = [] 
        for _, segm in enumerate(self._segmentation):
            if len(segm) > 0:
                query.append(sum([s<observation[_] for s in segm]))

            else:
                query.append(0)
            
        query.append(action)

        return tuple(query)

