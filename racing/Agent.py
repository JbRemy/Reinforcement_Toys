import pygame
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras import optimizers
from keras.initializers import VarianceScaling
from tensorflow.losses import huber_loss
import numpy as np
from itertools import cycle

def check_nans(x):
    try:
        return np.isnan(x)
    except ValueError:
        return any([check_nans(_) for _ in x])
    except TypeError:
        return False

class Agent(object):
    def __init__(self, game):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=12, activation='relu',
                             kernel_initializer=VarianceScaling(scale=2.0)))
        self.model.add(Dense(32, input_dim=12, activation='relu',
                             kernel_initializer=VarianceScaling(scale=2.0)))
        self.model.add(Dense(9, activation='linear',
                             kernel_initializer=VarianceScaling(scale=2.0)))
        self.game=game
        self.game.agent = self
        self.counter = 0
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def train(self, lr=0.01, n_episodes=500, max_t=1000, epsilon=0.9, 
              min_epsilon=0.05, batch_size=128, discount=0.999, max_memory_size=10000,
             action_per_seconds=2, epsilon_rate=5e-2, C=10):
        sgd = optimizers.SGD(lr=lr)
        self.model.compile(loss=huber_loss, optimizer=sgd)
        self.trainning = True
        self.epsilon =epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_rate = epsilon_rate
        self.X = []
        self.previous_state = self.build_state()
        self.previous_action = 8
        self.batch_size = batch_size
        self.discount = discount
        self.max_t = max_t
        self.max_memory_size = max_memory_size
        self.iteration = 0
        self.game.max_t = self.max_t
        self.reapeat_action = int(self.game.fps/action_per_seconds)
        self.cycle = 0
        self.running_reward = 0
        self.running_score = 0
        self.C=C
        self.game.game_loop()
        
    def act(self, reward):
        self.running_score += reward
        self.running_reward += reward
        state = self.build_state()
        if self.counter == 0:
            if np.random.uniform() < self.epsilon and self.trainning:
                action, keys = self.return_action('random')

            else:
                actions_values = self.model.predict(np.array([state]))[0]
                action, keys = self.return_action(actions_values)

            self.counter = (self.counter + 1) % self.reapeat_action

        else:
            self.counter = (self.counter + 1) % self.reapeat_action
            action, keys = self.previous_action_keys

        if self.trainning:
            if self.counter == 0 or self.game.car.has_reset:
                self.append_X(self.running_reward, state)
                self.previous_state = state
                self.previous_action = action
                self.running_reward = 0
                self.counter = 0
                self.iteration += 1

            if self.iteration > self.batch_size and self.iteration % 1 ==0: 
                x,y = self.build_batch()
                self.model.fit(x,y,epochs=1,batch_size=self.batch_size,
                               verbose=0)

            if self.cycle % self.C == 0:
                self.target_model.set_weights(self.model.get_weights())

            if self.game.car.has_reset :
                if self.iteration > self.batch_size:
                    self.epsilon = max(self.epsilon + (self.min_epsilon -
                                                       self.epsilon)*self.epsilon_rate,
                                       self.min_epsilon)
                self.cycle += 1
                if self.cycle % self.C  == 0:
                    if self.iteration > self.batch_size:
                        print("Finished cycle %i, score: %.2f, previous_loss: %.2f" %
                              (self.cycle, self.running_score,
                               self.model.history.history['loss'][-1]))
                    else:
                        print("Finished cycle %i, score: %.2f" %
                              (self.cycle, self.running_score))

                self.running_score = 0

        self.previous_action_keys = (action, keys)
        return keys

    def build_batch(self):
        X = []
        Y = []
        for _ in range(self.batch_size):
            i = np.random.randint(0, len(self.X))
            x = self.X[i]
            X.append(x['state'])
            y = self.model.predict(np.array([x['state']]))[0]
            y[x['action']] = x['reward']
            if not x['terminal']:
                y[x['action']] += self.discount*np.max(self.target_model.predict(np.array([x['next']]))[0])

            Y.append(y)

        return np.array(X),np.array(Y)


    def append_X(self, reward, state):
        x = {}
        x['state'] = self.previous_state
        x['next'] = state
        x['action'] = self.previous_action
        x['reward'] = reward
        x['terminal'] = self.game.car.has_reset
        if not check_nans([_ for k,_ in x.items()]):
            if len(self.X) >= self.max_memory_size:
                self.X = self.X[1:]

            self.X.append(x)
            
    def build_state(self):
        state = []
        for (x,_) in self.game.car.collision_points:
            state.append(x)

        state.append(self.game.car.speed)
        state.append(self.game.car.acc)
        state.append(self.game.car.theta_wheels)
        state.append(self.game.car.lat_speed)

        return state
    
    def return_action(self, values):
        action_space = {
            pygame.K_UP:False,
            pygame.K_DOWN:False,
            pygame.K_LEFT:False,
            pygame.K_RIGHT:False,
            pygame.K_r:False
        }
        if values == 'random':
            values = [0]*9
            values[np.random.randint(0,9)] = 1

        action = np.argmax(values)
        if action in [0, 4, 6]:
            action_space[pygame.K_UP] = True

        if action in [3, 5, 7]:
            action_space[pygame.K_DOWN] = True

        if action in [1, 4, 5]:
            action_space[pygame.K_LEFT] = True

        if action in [2, 6, 7]:
            action_space[pygame.K_RIGHT] = True

        return action, action_space
            


