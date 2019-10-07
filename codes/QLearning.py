from .utils import ipython_info
if ipython_info() == "notebook":
    from tqdm import tnrange as trange
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm, trange

from .PolicyEpsilonGreedy import PolicyEpsilonGreedy
from .LearningRate import LearningRate

import numpy as np
import time
from copy import deepcopy, copy

import multiprocessing
from copy import deepcopy

class QLearner(object):
    def __init__(self, Q_, env, gamma, epsilon, epsilon_decay, epsilon_min,
                 epsilon_decay_every, lr0, lr_decay, min_lr, max_steps,
                 verbose=1, logg_every_episode=25):
        self.verbose = verbose
        self.env = env
        self.gamma = gamma
        self.max_steps = max_steps
        self.Q_ = Q_
        self.policy = PolicyEpsilonGreedy(Q_, len(Q_._action_dict), epsilon, 
                                          epsilon_decay, epsilon_min, epsilon_decay_every)
        self.lr = LearningRate(lr0=lr0, decay=lr_decay, min_lr=min_lr, 
                               discrete=Q_._discrete,
                               actions_size=len(Q_._action_dict),
                               state_shape=Q_.W.shape[:-1],
                               segmentation=Q_._segmentation)
    
    def fit(self, n_episodes, logg_every_episode=50):
        rewards = [0]*n_episodes
        iterator = self.get_iterator_(n_episodes, 0, "Q-Learning")
        for episode in iterator:
            rewards[episode] += self.run_episode()
            if self.verbose and episode % logg_every_episode == 0:
                iterator.set_description("Q-Learning (reward: %.2f)" %\
                                         np.mean(rewards[max(0,episode-logg_every_episode):episode+1]))

            self.update_best_(rewards[episode])
                
        return self.best_policy, rewards, self.best_reward

    def run_episode(self):
        episode_reward = 0
        state = self.env.reset()
        for t in range(self.max_steps):
            action = self.policy(state)
            res = self.env.step(action)
            new_state, reward, done = res[:3]
            episode_reward += reward
            if done:  
                self.Q_.update(reward, state, action, self.lr(state, action))
                break
                
            else:
                self.Q_.update(reward + self.gamma*self.Q_.get_V(new_state),  state, action,
                          self.lr(state, action))
                
            state = new_state 

        return episode_reward

    def get_iterator_(self, steps, position, title, leave=True):
        if self.verbose > 0:
            iterator = trange(steps, desc=title, position=position, leave=leave)

        else:
            iterator = range(steps)

        return iterator

    def update_best_(self, reward):
         if not hasattr(self, "best_reward"):
             self.best_reward = reward
             self.best_policy = deepcopy(self.policy)

         elif reward > self.best_reward:
             self.best_reward = reward
             self.best_policy = deepcopy(self.policy)



