from IPython.display import HTML
import base64
import os
import gym
from gym import wrappers
import io

class Env(object):
    def __init__(self, exercise, wrapper_path=None):
        '''
            Initialises the environement
            Args: exercise: (str) The type of environement (ex:CartPole-v1)
                  wrapper_path: (str) Path where to store the video of the run
        '''
        self.env = gym.make(exercise) # Setting up a spave invaders game
        self.wrapper_path = wrapper_path
        if not wrapper_path is None: # wrapper for video saving
            self.env = wrappers.Monitor(self.env, self.wrapper_path, force=True)

    def run(self, steps=100, policy="random", **policy_kwargs):
        '''
            runs the agent for a certain number of time steps
            Args: steps: (int) The total numbr of steps to take
                  policy: (str) or (callable) must return an action
                  **policy_kwargs: supplementary arguments of the policy if needed
        '''
        rewards = [] # We will return the rewards

        # Setting the env to its original state
        state = self.env.reset() 

        for _ in range(steps):
            if policy=="random":# random selection of an action
                action = self.env.action_space.sample() 

            elif callable(policy):# if we passed a policy
                action = policy(state, **policy_kwargs)

            state, reward, done, info = self.env.step(action) # executing the action
            rewards.append(reward)
            if done: break # done declares the end of the run (ie: Game Over)

        # Closing the env
        self.env.close() 

        return rewards

    def display_run(self):
        '''
            This function displays the last saved run to the notebook
        '''
        video_path = list(filter(lambda x: x.endswith(".mp4"),
                            os.listdir(self.wrapper_path)))[0]
        video = io.open(os.path.join(self.wrapper_path, video_path), 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''
            <center><video width="360" height="auto" alt="test"
                         controls><source src="data:video/mp4;base64,{0}"
                         type="video/mp4" /></video></center>'''
             .format(encoded.decode('ascii')))
