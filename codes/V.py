import os 
import numpy as np
from math import floor

class V(object):
    def __init__(self, discrete=True, state_shape=[], segmentation=[], init_range=[0,1]):
        self.discrete = discrete
        if self.discrete:
            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(list(state_shape)))

        else:
            shape = [len(_)+2 for _ in segmentation]
            self.W = np.random.uniform(low=init_range[0], high=init_range[1],
                                       size=tuple(shape))
            self.segmentation = segmentation
    
    def __call__(self, observation):
        query = self.get_query_(observation)
        return self.W[query]
    
    def update(self, value, observation, lr):
        query = self.get_query_(observation)
        self.W[query] -= lr*(self.W[query]-value)
    
    def segment_(self, observation):
        query = [] 
        for _, segm in enumerate(self.segmentation):
            if len(segm) > 0:
                query.append(sum([s < observation[_] for s in segm]))

            else:
                query.append(0)
            
        return tuple(query)
    
    def get_query_(self, observation):
        if self.discrete:
            return tuple(observation)

        else:
            return self.segment_(observation)

