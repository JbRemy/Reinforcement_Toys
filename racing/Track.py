import pygame
from pygame.sprite import Sprite
import math
import numpy as np

n_checkpoints = 10
track_center = [400,300]

class Track(Sprite):
    def __init__(self, game, f, grid_file=None, grid_size=None):
        super(Sprite, self).__init__()
        self.game = game
        self.image = pygame.image.load(f).convert_alpha()
        self.image = pygame.transform.scale(self.image, self.game.window_size)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        if not grid_file is None:
            self.grid_file = grid_file
            self.grid = np.load(self.grid_file)
            self.grid_size = self.grid.shape[0]

        elif not grid_size is None:
            self.grid_size = grid_size
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.display_grid = False

        self.checkpoints_lines = []
        for theta in [2*math.pi*_ /n_checkpoints for _ in range(n_checkpoints)]:
            self.checkpoints_lines.append([
               track_center,
                (
                    track_center[0] + math.cos(theta)*500,
                    track_center[1] + math.sin(theta)*500
                )])

        self.display_checkpoints = False


    def update(self, key):
        if key[pygame.K_c]:
            self.display_checkpoints = not self.display_checkpoints

        if key[pygame.K_g]:
            self.display_grid = not self.display_grid

    def display(self):
        self.game.screen.blit(self.image, (0,0))
        if self.display_checkpoints:
            for line in self.checkpoints_lines:
                pygame.draw.lines(self.game.screen, pygame.Color('blue'), False, line, 3)

        if self.display_grid:
            self.draw_grid()

    def draw_grid(self):
        for (x,y), flag in np.ndenumerate(self.grid):
            if flag == 1 or flag == 2:
                case = pygame.Surface((self.game.window_size[0]/self.grid_size,
                                      self.game.window_size[1]/self.grid_size))
                case.set_alpha(100)
                case = case.convert()
                case.fill([pygame.Color('blue'),pygame.Color('purple')][flag-1])
                self.game.screen.blit(case, (x*self.game.window_size[0]/self.grid_size,
                                             y*self.game.window_size[1]/self.grid_size))

        for x in [_*self.game.window_size[0]/self.grid_size for _ in range(self.grid_size)]:
            pygame.draw.lines(self.game.screen, pygame.Color('white'), False,
                              [(x,0), (x, self.game.window_size[1])], 1)
        for y in [_*self.game.window_size[1]/self.grid_size for _ in range(self.grid_size)]:
            pygame.draw.lines( self.game.screen, pygame.Color('white'), False, 
                [(0,y), (self.game.window_size[0], y)], 1)
        
    def fill_grid(self, pos):
        x = int(pos[0] / self.game.window_size[0] * self.grid_size)
        y = int(pos[1] / self.game.window_size[1] * self.grid_size)
        self.grid[x,y] = (self.grid[x,y]+1) % 3
        np.save(self.grid_file, self.grid)





