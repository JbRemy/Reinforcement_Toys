from typing import List, Union, Tuple, Type
import numpy as np
from .Q import Q

Q_type = Type[Q]

class PolicyEpsilonGreedy(object):
    """This is an implementation of the epsilon greedy policy with respect to a
    provided Q-table

    Methods:
        __init__(self, Q: Q, action_space: List[Union[int, str]], 
                 epsilon: float, decay: float, lower: float, 
                 decay_every: int) -> None:
            Initializes the Policy
        __call__(self, state: Union[np.array, List, int, str],
                 be_greedy: bool=False) -> Union[int, str]:
            Returns an action with respect to the policy
    """
    def __init__(self, Q: Q_type, action_space: List[Union[int, str]], 
                 epsilon: float, decay: float, lower: float, 
                 decay_every: int) -> None:
        """Initializes the Policy

        Args:
            Q (Q.Q): The Q-table
            action_space (list): All the possible actions
            epsilon (float): Exploration proportion
            decay (float): exploration decay.
            lower (float): smallest possible value for epsilon
            decay_every (int): when to decay epsilon

            $\epsilon = \frac{\epsilon_0}{decay^{\left \lfloor{t/rate}\right \rfloor}}$

        Returns:
            None
        """
        self._Q = Q
        if isinstance(action_space, int):
            self._action_space = list(range(action_space))

        else:
            self._action_space = action_space

        self._epsilon = epsilon
        self._probs = [self._epsilon/(len(self._action_space)-1)]*len(self._action_space)
        self._decay = decay
        self._lower = lower
        self._decay_every = decay_every
        self._t = 0

        return None
        
    def __call__(self, state: Union[np.array, List, int, str],
                 be_greedy: bool=False) -> Union[int, str]:
        """Returns an action with respect to the policy

        Args:
            state (np.array, list, int, str): an element of the state_space
            be_greedy (bool): If True returns the greedy policy

        Returns:
            action (int, str)
        """
        vals = [self._Q(state, action) for action in self._action_space]
        if be_greedy:
            return self._action_space[np.argmax(vals)]

        else:
            probs = self._probs.copy()
            probs[np.argmax(vals)] = 1 - self._epsilon
            if self._t % self._decay_every == 0 and self._t > 0:
                self._do_decay()
            
            self._t += 1
            return np.random.choice(self._action_space, p=probs)
    
    def _do_decay(self) -> None:
        """Decays the learning rate

        Returns: 
            None
        """
        self._epsilon = max(self._lower, self._epsilon*self._decay)
        self._probs = [self._epsilon/(len(self._action_space)-1)]*len(self._action_space)

        return None
