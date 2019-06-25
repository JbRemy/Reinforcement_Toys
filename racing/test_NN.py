import pygame
from pygame.locals import *

from Game import Game
from Agent import Agent
 
def main():
    game = Game(caption='racing')
    agent = Agent(game, weights='agents/best_weights')
    agent.train(trainning=False)

 
if __name__ == '__main__': main()
