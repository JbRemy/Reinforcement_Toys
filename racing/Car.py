from pygame.sprite import Sprite
import pygame
import math
import numpy as np
from itertools import cycle

# Parameters of the dynamics of the car, are hand adjusted.
car_weight = 733
engine_F = 2000
C_drag = 0.4257
C_pi = 12.8
C_alpha = 50
max_theta_wheels = math.pi/ 2
theta_increment = (math.pi / 2) 
max_speed = 10
vision_range = 100

rotate = pygame.transform.rotate

class Car(Sprite):
    def __init__(self, game):
        '''
        '''
        super(Car, self).__init__()
        self.game = game
        self.car_length = 20 
        # Sprite loading
        self.image = pygame.image.load('sprites/car.png').convert_alpha()
        #self.image.set_colorkey(0, RLEACCEL)
        self.image = pygame.transform.scale(self.image, (self.car_length,
                                                        int(self.car_length*0.5)))
        self.image = pygame.transform.flip(self.image, True, False)
        self.mask = pygame.mask.from_surface(self.image)
        self.image_original = self.image
        # Caracteristics
        self.pos = np.array([550,480], 'float32')
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.speed = 0
        self.acc = 0
        self.theta = 0
        self.theta_wheels = 0 
        self.F=0
        self.lat_speed = 0
        self.lat_speed_front = 0
        self.lat_speed_rear = 0
        self.lat_speed = 0
        self.update_lines()
        self.display_lines = False
        self.checkpoint_value = 0
        self.has_reset = False

    def reset(self):
        self.__init__(self.game)
        self.has_reset = True

    def display(self):
        self.game.screen.blit(self.image, self.rect)
        if self.display_lines:
            self.draw_lines()

    def move(self, key):
        if key[pygame.K_r]:
            self.reset()

        else:
            dt = 1/self.game.get_fps()
            self.update_acc(key, dt)
            self.update_pos(dt)
            self.rotate()

        self.has_reset = False

    def update_acc(self, key, dt):
        '''
        '''
        if key[pygame.K_UP]:
            self.F = engine_F
        elif key[pygame.K_DOWN]:
            self.F = -engine_F*5
        else:
            self.F = -engine_F

        if key[pygame.K_LEFT]:
            self.theta_wheels = min(max_theta_wheels,
                                   self.theta_wheels + dt*theta_increment)
        elif key[pygame.K_RIGHT]:
            self.theta_wheels = max(-max_theta_wheels, 
                                   self.theta_wheels - dt*theta_increment)
        else:
            self.theta_wheels = 0

    def update(self, key):
        if key[pygame.K_l]:
            self.display_lines = not self.display_lines

    def update_pos(self, dt):
        '''
        '''
        self.acc = (self.F - C_drag*self.speed**2 - C_pi * self.speed)/car_weight
        self.speed = np.clip(self.speed+self.acc*dt, 0, max_speed)
        dz = self.speed * dt * self.car_length
        self.pos_update = np.array([math.cos(self.theta),
                                    math.sin(self.theta)], 'float32')*dz
        self.pos += self.pos_update
        if dz<self.car_length:
            dtheta = math.asin(dz*math.sin(self.theta_wheels)/self.car_length)
        else:
            dtheta = self.theta_wheels

        self.theta -= dtheta 
        self.theta = self.theta % (math.pi*2)

        self.update_lines()

    def drift(self, dz, dt):
        omega = self.speed*math.sin(self.theta_wheels)/dz
        front = math.atan2(self.lat_speed - omega*self.car_length/2, self.speed)\
                - self.theta_wheels
        rear = math.atan2(self.lat_speed + omega*self.car_length/2,self.speed)
        self.lat_acc_front = (C_alpha * front - C_drag*self.lat_speed**2 - C_pi * self.lat_speed) / car_weight
        self.lat_acc_rear = (C_alpha * rear- C_drag*self.lat_speed**2 - C_pi * self.lat_speed) / car_weight
        self.lat_speed_front += self.lat_acc_front * dt
        self.lat_speed_rear += self.lat_acc_rear * dt
        self.lat_speed += self.lat_speed_rear + self.lat_speed_front
        pos_mod_front = self.lat_speed_front * dt
        pos_mod_rear = self.lat_speed_rear * dt
        pos_mod = np.array([math.sin(self.theta),
                            -math.cos(self.theta)],
                           'float32')*(pos_mod_rear + pos_mod_front/2)*self.car_length
        theta_mod = math.asin((pos_mod_front-pos_mod_rear)/self.car_length)

        return pos_mod, theta_mod

    def rotate(self):
        '''
        '''
        rot = (-self.theta/(2*math.pi)*360) % 360
        new_image = pygame.transform.rotate(self.image_original, rot)
        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def update_lines(self):
        self.lines = []
        for theta_mod in [0, math.pi/4, math.pi/2, 3*math.pi/4,
                         math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
            line = [
                self.rect.center, 
                (self.rect.centerx + math.cos(self.theta+theta_mod)*vision_range,
                 self.rect.centery + math.sin(self.theta+theta_mod)*vision_range)
            ]
            self.lines.append(line)

        self.get_line_collision_points()

    def draw_lines(self):
        for line, (_, point) in zip(self.lines, self.collision_points):
            pygame.draw.lines(self.game.screen, pygame.Color('white'), False, line, 3)
            pygame.draw.circle(self.game.screen, pygame.Color('white'), point, 5, 1)

        pygame.draw.rect(self.game.screen, pygame.Color('white'),  self.rect, 3)

    def get_line_collision_points(self):
        self.collision_points = []
        for line in self.lines:
            for x in range(21):
                point = (line[0][0] + int((line[1][0] - line[0][0]) *  x/20),
                         line[0][1] + int((line[1][1] - line[0][1]) *  x/20))
                if not self.game.track.mask.get_at(point):
                    self.collision_points.append([x/20, tuple(map(int,point))])
                    break

                elif x == 20:
                    self.collision_points.append([x/20, tuple(map(int,line[1]))])


    def check_status(self):
        still_on_track = True
        if any([x == 0  for (x, _) in self.collision_points]):
            still_on_track = False

        x, y = self.check_grid_position()
        grid_value = self.game.track.grid[min(x, self.game.track.grid_size-1),
                                          min(y, self.game.track.grid_size-1)]
        reward = -10/self.game.fps
        if grid_value == (self.checkpoint_value + 1) % 3:
            reward = 100

        if grid_value == (self.checkpoint_value + 2) % 3:
            reward = 200

        self.checkpoint_value = grid_value 

        return still_on_track, reward

    def check_grid_position(self):
        x = int(self.pos[0] / self.game.window_size[0] * self.game.track.grid_size)
        y = int(self.pos[1] / self.game.window_size[1] * self.game.track.grid_size)

        return x, y
