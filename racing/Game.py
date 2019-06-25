import pygame
from itertools import cycle

from Track import Track
from Car import Car

BLINK_EVENT = pygame.USEREVENT + 0
EDIT_MODE_GRID = False
EDIT_MODE_STARTS = False

class Game(object):
    def __init__(self, window_size=(860,544), caption='Random Program', fps=60,
                agent = 'human', render=True, random_start=True):
        self.random_start = random_start
        self.max_t = 10000000
        self.agent = agent
        self.RENDER = render
        self.fps = fps
        self.window_size = window_size
        # Initialization of the window
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.screen_rect = self.screen.get_rect()
        pygame.display.set_caption(caption)
        # Background filling
        background = pygame.Surface(self.screen.get_size())
        self.background = background.convert()
        self.background_color = pygame.Color('white')
        self.background.fill(self.background_color)
        # Creating a clock
        self.clock = pygame.time.Clock()
        # Displaying welcome text
        self.display_text_('Welcome to the Racing Game', 'center')
        blink_surfaces, blink_rect = self.display_text_('(Press Enter)',
                                          (self.background.get_rect().centerx,
                                           self.background.get_rect().centery+20),
                                          size=28,
                                          blinking=self.fps)
        blink_surface = next(blink_surfaces)
        # Display
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        # Game loop
        if not self.RENDER:
            return

        while 1:
            for event in pygame.event.get():
                if self.check_QUIT(event):
                    return 

                if event.type == BLINK_EVENT:
                    blink_surface = next(blink_surfaces)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.game_loop()
                        return

                self.screen.blit(blink_surface, blink_rect)
                pygame.display.update()
                self.clock.tick(self.fps)

    def game_loop(self):
        self.background_color = pygame.Color('skyblue')
        self.background.fill(self.background_color)
        self.screen.blit(self.background, (0,0))
        self.track = Track(self,
                           f='tracks/1/PinClipart.com_race-clipart_41252.png',
                           grid_file='tracks/1/grid.npy',
                           start_points_file='tracks/1/start_points.npy',
                           grid_size=25)
        self.car = Car(self)
        score = 0
        reward = 0
        t = 0
        while 1:
            for event in pygame.event.get():
                if self.check_QUIT(event):
                    return 

                if event.type == pygame.MOUSEBUTTONUP and EDIT_MODE_GRID:
                    self.track.fill_grid(pygame.mouse.get_pos())

                if event.type == pygame.MOUSEBUTTONUP and EDIT_MODE_STARTS:
                    pressed = pygame.key.get_pressed()
                    pressed_key = None
                    for _ in [pygame.K_UP,pygame.K_RIGHT,pygame.K_DOWN,pygame.K_LEFT]:
                        if pressed[_]:
                            pressed_key = _

                    if not pressed_key is None:
                        self.track.build_starting_points(
                            pygame.mouse.get_pos(),
                            pressed_key
                        )

                if event.type == pygame.KEYDOWN:
                    pressed = pygame.key.get_pressed()
                    self.car.update(pressed) 
                    self.track.update(pressed) 
                    if pressed[pygame.K_n]:
                        self.RENDER = not self.RENDER
                    if hasattr(self.agent, 'trainning'):
                        if pressed[pygame.K_p]:
                            self.agent.trainning = not self.agent.trainning


            self.car.move(self.get_agent_action_(reward))

            still_on_track, reward = self.car.check_status()
            if not still_on_track or t > self.max_t:
                self.car.reset()
                t = 0
                reward = -100
                score = 100

            score += reward

            if self.RENDER:
                self.screen.blit(self.background, (0,0))
                self.display_text_('Score : %i' % score, (80,20))
                self.display_text_('Time : %i' % int(t/self.fps), (80,40))
                self.track.display()
                self.car.display()
                pygame.display.update()
                self.clock.tick(self.fps)

            t += 1

    def get_agent_action_(self, reward):
        if self.agent == 'human':
            return pygame.key.get_pressed()
        
        else:
            return self.agent.act(reward)

    def display_text_(self, string, pos, font=None, size=36, blinking=None):
        assert isinstance(pos, str) or isinstance(pos, tuple),\
            'Text position as to be str or tuple'
        if isinstance(pos, str):
            assert pos in ['center'], 'Text position as to be in "center"'
        if isinstance(pos, tuple):
            assert len(pos) == 2, 'Text position as to be of length 2'

        font = pygame.font.Font(font, size)
        text = font.render(string, 1, pygame.Color('black'))
        textpos = text.get_rect()
        if isinstance(pos, str):
            textpos.centerx = self.background.get_rect().centerx
            textpos.centery = self.background.get_rect().centery

        if isinstance(pos, tuple):
            textpos.centerx, textpos.centery = pos

        blank_surface = pygame.Surface(textpos.size)
        blank_surface.fill(self.background_color)
        if not blinking is None:
            blink_rect = textpos
            blink_surfaces = cycle([text, blank_surface])
            pygame.time.set_timer(BLINK_EVENT, blinking)
            return blink_surfaces, blink_rect

        self.screen.blit(text, textpos)

    def get_fps(self):
        return self.fps
        fps = self.clock.get_fps()
        if fps > 0 and self.RENDER:
            return fps
        else:
            return self.fps

    def check_QUIT(self, event):
        if event.type == pygame.QUIT:
            return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return True
        else:
            return False



