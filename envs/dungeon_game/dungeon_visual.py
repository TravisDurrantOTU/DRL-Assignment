# Visual rendering logic so that a graphical window pops up
# TODO: Separate core game logic into a DungeonGame class in dungeon_core
# right now this is holding some of that and is incorrectly functioning as a main for testing
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

            