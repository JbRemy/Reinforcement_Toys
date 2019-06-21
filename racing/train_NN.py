import pygame
from pygame.locals import *

from Game import Game
from Agent import Agent
 
def main():
    game = Game(caption='racing')
    agent = Agent(game)
    agent.train()

 
if __name__ == '__main__': main()
