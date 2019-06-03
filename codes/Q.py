import os 
import numpy as np
from math import floor

class Q(object):
    def __init__(self, actions_size, discrete=True, state_shape=[],
                 segmentation=[], init_range=[0,1]):
        self.discrete = discrete
        self.actions_size = actions_size
        if self.discrete:
            if isinstance(actions_size, list):
                self.action_dict = {_:ind for ind,_ in enumerate(actions_size)}

            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(list(state_shape)+[len(actions_size)]))

        else:
            shape = [len(_)+2 for _ in segmentation]
            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(shape + [actions_size]))
            self.segmentation = segmentation
    
    def __call__(self, observation, action):
        query = self.get_query_(observation, action)
        return self.W[query]
    
    def update(self, value, observation, action, lr):
        query = self.get_query_(observation, action)
        self.W[query] -= lr*(self.W[query]-value)
    
    def segment_(self, observation, action=None):
        query = [] 
        for _, segm in enumerate(self.segmentation):
            if len(segm) > 0:
                query.append(sum([s < observation[_] for s in segm]))

            else:
                query.append(0)
            
        if not action is None:
            query.append(action)

        return tuple(query)
    
    def get_V(self, observation=None):
        if observation is None:
            self.V = np.max(self.W, -1)

        else:
            query = self.get_query_(observation)
            return np.max(self.W[query], -1)

    def get_query_(self, observation, action=None):
        if self.discrete:
            if hasattr(self, "action_dict") and not action is None:
                action = self.action_dict[action]
                
            if not action is None:
                return tuple(observation + [action])

            else:
                return tuple(observation)

        else:
            return self.segment_(observation, action)

    #def save(self, path):
    #    if not os.path.exists(path):
    #        os.mkdir(path)
    #        
    #    np.save(os.path.join(path, "W.npy"), self.W)
    #    if not discrete:
    #        np.save(os.path.join(path, "segmentation.npy"), self.segmentation)

    #@classmethod
    #def Load(cls, path):
    #    Q_ = cls.__new__(cls)
    #    Q_.W = np.load(os.path.join(path, "W.npy"))
    #    if os.path.exists(os.path.join(path, "high.npy")):
    #        Q_.discrete = False
    #        Q_.segmentation = np.load(os.path.join(path, "segmentation.npy"))

    #    else:
    #        Q_.discrete = True
    #    
    #    return Q_k
