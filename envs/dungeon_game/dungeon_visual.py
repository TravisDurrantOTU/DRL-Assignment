# Visual rendering logic so that a graphical window pops up
# This is honestly probably completely pointless but whatever

import pygame

class dungeonVisual():

    def __init__(self):
        # Placeholder size for now
        w,h = 1080, 720

        self.screen = pygame.display.set_mode((w,h))
        pygame.display.set_caption("Dungeon Explorer")

    def getScreen(self):
        return self.screen
    
    def primeScreen(self):
        self.screen.fill((255,255,255))

    def graphicsUpdate(self):
        pygame.display.flip()

            