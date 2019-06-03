import numpy as np

class PolicyEpsilonGreedy(object):
    def __init__(self, Q, action_space, epsilon, decay, lower, decay_every):
        self.Q = Q
        if isinstance(action_space, int):
            self.action_space = list(range(action_space))

        else:
            self.action_space = action_space

        self.epsilon = epsilon
        self.probs = [self.epsilon/(len(self.action_space)-1)]*len(self.action_space)
        self.decay = decay
        self.lower = lower
        self.decay_every = decay_every
        self.t = 0
        
    def __call__(self, state, be_greedy=False):
        vals = [self.Q(state, action) for action in self.action_space]
        if be_greedy:
            return self.action_space[np.argmax(vals)]
        else:
            probs = self.probs.copy()
            probs[np.argmax(vals)] = 1 - self.epsilon
            if self.t % self.decay_every == 0 and self.t > 0:
                self.do_decay()
            
            self.t += 1
            return np.random.choice(self.action_space, p=probs)
    
    def do_decay(self):
        self.epsilon = max(self.lower, self.epsilon*self.decay)
        self.probs = [self.epsilon/(len(self.action_space)-1)]*len(self.action_space)
