import numpy as np

class policy_ietration(Object):
    def __init__(self, env):
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            self.actions = [_ for _ in range(env.action_space.n)]

        self.states = []
        self.V = []
        self.rewards_ = env.compute_reward

    def train(self, K):

        for k in range(K):
            V = self.evaluate_()
            for 

    def policy_(self, x):
        
        values = []
        for a in self.actions:
            env.state = self.states(x)
            observation, reward, done, info = env.step(action)
            values.append(reward + self.gamma*self.V[observation]
            
        max = [self.rewards_(x,a) + self.gamma*self.env.
        
