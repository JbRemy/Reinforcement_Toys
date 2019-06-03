from math import floor
import numpy as np

class LearningRate(object):
    def __init__(self, lr0, decay, actions_size, discrete=True, state_shape=[], segmentation=[], min_lr=0.1):
        self.discrete = discrete
        self.decay = decay
        self.lr0 = lr0
        self.min_lr = min_lr
        if self.discrete:
            if isinstance(actions_size, list):
                self.action_dict = {_:ind for ind,_ in enumerate(actions_size)}

            self.lr = np.ones(shape=tuple(list(state_shape)+[len(actions_size)]))
            self.t = np.zeros(shape=tuple(list(state_shape)+[len(actions_size)]))
            self.lr *= lr0

        else:
            shape = [len(segm)+2 for segm in segmentation]
            self.lr = np.ones(shape=tuple(shape + [actions_size]), dtype=np.float)
            self.lr *= lr0
            self.t = np.zeros(shape=tuple(shape + [actions_size]))
            self.segmentation = segmentation
    
    def __call__(self, observation, action):
        query = self.get_query_(observation, action)
        self.lr[query] = max(self.lr0/(1+self.t[query]*self.decay), self.min_lr)
        self.t[query] += 1
        return self.lr[query]
    
    def segment_(self, observation, action):
        query = [] 
        for _, segm in enumerate(self.segmentation):
            if len(segm) > 0:
                query.append(sum([s<observation[_] for s in segm]))

            else:
                query.append(0)
            
        query.append(action)
        return tuple(query)
    
    def get_query_(self, observation, action=None):
        if self.discrete:
            if hasattr(self, "action_dict") and not action is None:
                action = self.action_dict[action]
                
            return tuple(observation + [action])

        else:
            return self.segment_(observation, action)

