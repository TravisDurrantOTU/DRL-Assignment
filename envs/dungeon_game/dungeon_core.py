# Core classes and logic for the dungeon exploration game
# Visual stuff will probably be stuffed in here while I test it

# TODO: Add basic world logic
# TODO: Add ability for player to move in world
# TODO: Decide if discrete (easy but lame) or continuous (cooler but harder) movement

from abc import ABC
from abc import abstractmethod
from dungeon_visual import dungeonVisual
import pygame
import math
import os

# Global variables are bad practice, sure
# But I might need to change some of these values later
# So they're global constants
playerRadius = 15
coinRadius = 10

class GameObject(ABC):
    @abstractmethod
    def draw():
        pass

class Moveable(GameObject):
    @abstractmethod
    def move():
        pass

class Interactable(GameObject):
    # what should happen upon collision with the player
    @abstractmethod
    def collidePlayer():
        pass

    # should just return the size and shape of the interactable
    @abstractmethod
    def detectCollision():
        pass

class DungeonController():
    def __init__(self, visuals=False, human=False):
        pygame.init()
        self.running = True
        self.visuals = visuals
        self.renderer = dungeonVisual()
        if human:
            self.player = HumanPlayer()
        self.room = Room()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.player.move()
            self.room.checkCollisions(self.player)

            if self.visuals:
                self.renderer.primeScreen()
                self.room.draw(self.renderer.getScreen())
                self.player.draw(self.renderer.getScreen())
                self.renderer.graphicsUpdate()


class Coin(Interactable):
    # Defaults to first tile center
    def __init__(self, initial_pos=(72, 72)):
        # offset because circles are drawn based on center not top-left
        self.x = initial_pos[0] + 36
        self.y = initial_pos[1] + 36 
        self.collected = False

    def collidePlayer(self):
        # TODO: Actual coin logic
        print("coin collected")
        self.collected = True

    def detectCollision(self, player):
        if self.collected:
            return

        dx = self.x - player.x
        dy = self.y - player.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < abs(coinRadius + playerRadius):
            self.collidePlayer()


    def draw(self, screen):
        # TODO: Add texture
        if self.collected:
            return
        pygame.draw.circle(screen, (200, 200, 0), (self.x, self.y), coinRadius)

class PitHazard(Interactable):
        # Defaults to occupying second tile
    def __init__(self, initial_pos=(144, 144)):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        

    def collidePlayer(self, player):
        # TODO: Actual health logic
        # right now just sends player back up to top right
        player.x = 72
        player.y = 72

    def detectCollision(self, player):
        def clamp(value, min_value, max_value):
            return max(min_value, min(value, max_value))
        closest_x = clamp(player.x, self.x, self.x + 72)
        closest_y = clamp(player.y, self.y, self.y + 72)

        dx = player.x - closest_x
        dy = player.y - closest_y
        distance_squared = dx * dx + dy * dy
        
        if distance_squared <= playerRadius*playerRadius:
            self.collidePlayer(player)


    def draw(self, screen):
        # TODO: Add texture
        pygame.draw.rect(screen, (0, 0, 255), (self.x, self.y, 72, 72))


# TODO: Implement drawing player health and other stats on screen
class HUD(GameObject):
    def __init__(self):
        return

    def draw(self, player):
        return

# Room gets its own class so that they are revisitable
class Room(GameObject):
    @staticmethod
    def read_tileset(filename='defaultroom.txt'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'rooms', filename)
        tile_mapping = {
            '#': 'wall',
            '.': 'empty',
            'c': 'coin',
            'p': 'pitfall'
        }

        tileset = []

        try:
            with open(file_path, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    row = []
                    for col_number, char in enumerate(line, start=1):
                        if char in tile_mapping:
                            row.append(tile_mapping[char])
                        else:
                            raise ValueError(f"Unknown tile character '{char}' at line {line_number}, column {col_number}")
                    tileset.append(row)

            return tileset

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __init__(self):
        self.size = 10
        self.interactables = []
        #TODO: Add actual logic instead of this hardcoded box trash
        self.tiles = self.read_tileset()


        for x in range(self.size):
            for y in range(self.size):
                if self.tiles[x][y] == 'pitfall':
                    self.interactables.append(PitHazard(initial_pos=(x*72,y*72)))
                elif self.tiles[x][y] == 'coin':
                    self.interactables.append(Coin(initial_pos=(x*72,y*72)))
    def checkCollisions(self, player):
        for item in self.interactables:
            item.detectCollision(player)
    
    def draw(self, screen):
        #TODO: Make tileset for rooms
        for i in range(self.size):
            for j in range(self.size):
                pos_x = 0 + i*72
                pos_y = 0 + j*72
                if self.tiles[i][j] == 'wall':
                    colour = (0,0,0)
                elif self.tiles[i][j] == 'empty' or self.tiles[i][j] == 'coin':
                    colour = (100,100,100)
                # This should have the updated tile image drawn instead
                pygame.draw.rect(screen, colour, (pos_x, pos_y, 72, 72))
                pygame.draw.rect(screen, (0, 0, 0), (pos_x, pos_y, 72, 72), 1)
        for item in self.interactables:
            item.draw(screen)

        

class HumanPlayer(Moveable):
    def __init__(self, health=10, initial_pos=(500, 400)):
        self.health = health
        self.gold = 0
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.x_speed = 0.0
        self.y_speed = 0.0

    def draw(self, screen):
        #TODO: Add default player skin, remove placeholder circle
        pygame.draw.circle(screen, (255, 50, 0), (self.x, self.y), playerRadius)

    def update_velocity(self, dx=0, dy=0):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            dx += 0.1
        if keys[pygame.K_LEFT]:
            dx -= 0.1
        if keys[pygame.K_DOWN]:
            dy += 0.1
        if keys[pygame.K_UP]:
            dy -= 0.1
        self.x_speed += dx
        self.y_speed += dy

    def move(self):
        self.update_velocity()

        self.x += self.x_speed
        self.y += self.y_speed

        # hardlocking inside the boundary
        # 0 + 72 (wall tile size) + 15 (player radius)
        # 720 - above
        if self.x <= 87:
            self.x = 87
        elif self.x >= 633:
            self.x = 633
        if self.y <= 87:
            self.y = 87
        elif self.y >= 633:
            self.y = 633

        # so that speed decays
        self.x_speed *= 0.9
        self.y_speed *= 0.9
        if abs(self.x_speed) < 0.05:
            self.x_speed = 0
        if abs(self.y_speed) < 0.05:
            self.y_speed = 0

    

if __name__ == "__main__":
    dc = DungeonController(visuals=True, human=True)
    dc.run()