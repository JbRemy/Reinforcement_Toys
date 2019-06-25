import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pygame
from pygame.locals import *

from Game import Game
from Agent import Agent
 
def main():
    game = Game(caption='racing', render=False)
    agent = Agent(game)
    agent.train()

if __name__ == '__main__': main()
