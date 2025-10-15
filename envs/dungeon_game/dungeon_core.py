# Core classes and logic for the dungeon exploration game
# Visual stuff will probably be stuffed in here while I test it

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
        self.rooms = DungeonConstructor.constructDungeon()
        self.activeRoom = 0
        self.hud = HUD()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.player.move()
            self.rooms[self.activeRoom].checkCollisions(self.player)

            if self.visuals:
                self.renderer.primeScreen()
                self.hud.draw(self.player, self.renderer.getScreen())
                self.room.draw(self.renderer.getScreen())
                self.player.draw(self.renderer.getScreen())
                self.renderer.graphicsUpdate()

class DungeonConstructor():
    @staticmethod
    def constructDungeon():
        rooms = []
        # smooth brain garbage but i dont care
        for i in range(16):
            index = i+1
            if index == 10:
                index = 'A'
            elif index == 11:
                index == 'B'
            elif index == 12:
                index == 'C'
            elif index == 13:
                index = 'D'
            elif index == 14:
                index = 'E'
            elif index == 15:
                index = 'F'
            elif index == 16:
                index = 'G'
            newroom = Room(f'room{index}.txt')
            rooms.append(newroom)
        return rooms


class Coin(Interactable):
    # Defaults to first tile center
    def __init__(self, initial_pos=(72, 72)):
        # offset because circles are drawn based on center not top-left
        self.x = initial_pos[0] + 36
        self.y = initial_pos[1] + 36 
        self.collected = False

    def collidePlayer(self, player):
        player.gold += 1
        self.collected = True

    def detectCollision(self, player):
        if self.collected:
            return

        dx = self.x - player.x
        dy = self.y - player.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < abs(coinRadius + playerRadius):
            self.collidePlayer(player)


    def draw(self, screen):
        # TODO: Add texture
        if self.collected:
            return
        pygame.draw.circle(screen, (200, 200, 0), (self.x, self.y), coinRadius)

class Door(Interactable):
    def __init__(self, goes_to=0, facing='up', initial_pos=(0, 0)):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.nextroom = goes_to
        self.direction = facing

    def collidePlayer():
        return True
    
    def draw(self, screen):
        pygame.draw.circle(screen, (100, 0, 100), (self.x+36, self.y+36), 5)

    def detectCollision(self, player):
        if player.x >= self.x and player.x <= self.x + 72:
            if self.direction == 'up':
                if self.y + 20 <= player.x:
                    return self.nextroom
            if self.direction == 'down':
                if self.y - 20 >= player.x:
                    return self.nextroom
        if player.y >= self.y and player.y <= self.y + 72:
            if self.direction == 'left':
                if self.x - 20 >= player.x:
                    return self.nextroom
            if self.direction == 'right':
                if self.x + 20 <= player.x:
                    return self.nextroom



class PitHazard(Interactable):
        # Defaults to occupying second tile
    def __init__(self, initial_pos=(144, 144), room_spawn=(72, 72)):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.room_spawn = room_spawn

    def collidePlayer(self, player):
        player.x = self.room_spawn[0]
        player.y = self.room_spawn[1]
        player.health -= 1

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
        pygame.draw.circle(screen, (150, 150, 150), (self.x + 36, self.y + 36), 20)


# TODO: make this not look like garbage
class HUD(GameObject):
    def __init__(self):
        return

    def draw(self, player, screen):
        font = pygame.font.SysFont('Arial', 36)
        text1 = font.render(f"Gold: {player.gold}", True, (0,0,0))
        text2 = font.render(f"Health: {player.health}", True, (0,0,0))
        screen.blit(text1, (900, 300))
        screen.blit(text2, (900, 420))

        

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
            'p': 'pitfall',
            'i': 'initial'
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
                            row.append(char)
                    tileset.append(row)

            return tileset

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __init__(self, filename='defaultroom.txt'):
        self.size = 10
        self.collidables = []
        #TODO: Add actual logic instead of this hardcoded box trash
        self.tiles = self.read_tileset(filename=filename)
        self.spawnpoint = (72, 72)

        # grotesquely inefficient but this logic changes later based on transitions
        for x in range(self.size):
            for y in range(self.size):
                if self.tiles[y][x] == 'initial':
                    self.spawnpoint = (x*72+36,y*72+36)


        for x in range(self.size):
            for y in range(self.size):
                if self.tiles[y][x] == 'pitfall':
                    self.collidables.append(PitHazard(initial_pos=(x*72,y*72), room_spawn=self.spawnpoint))
                elif self.tiles[y][x] == 'coin':
                    self.collidables.append(Coin(initial_pos=(x*72,y*72)))
                elif self.tiles[y][x] != 'initial':
                    # This is door creation logic
                    facing = 'up'
                    initial_pos = (0,0)
                    if x == 0:
                        facing = 'right'
                        initial_pos=(100,y*72+36)
                    if x == 8:
                        facing = 'left'
                        initial_pos=(100,y*72+36)
                    if y == 0:
                        facing = 'up'
                        initial_pos=(x*72+36, 100)
                    if y == 8:
                        facing = 'down'
                        initial_pos=(x*72+36, 100)
                    self.collidables.append(Door(goes_to=self.tiles[y][x], facing=facing, initial_pos=initial_pos))

    
    def checkCollisions(self, player):
        for item in self.collidables:
            item.detectCollision(player)
    
    def draw(self, screen):
        #TODO: Make tileset for rooms
        for i in range(self.size):
            for j in range(self.size):
                pos_x = 0 + i*72
                pos_y = 0 + j*72
                if self.tiles[j][i] == 'wall':
                    colour = (0,0,0)
                elif self.tiles[j][i] == 'door':
                    colour = (0,255,0)
                elif self.tiles[j][i] == 'pitfall':
                    colour = (200, 0, 0)
                else:
                    colour = (100,100,100)
                # This should have the updated tile image drawn instead
                pygame.draw.rect(screen, colour, (pos_x, pos_y, 72, 72))
                pygame.draw.rect(screen, (0, 0, 0), (pos_x, pos_y, 72, 72), 1)
        for item in self.collidables:
            item.draw(screen)

class Player(Moveable):
    @abstractmethod
    def draw():
        pass
    @abstractmethod
    def move():
        pass
    @abstractmethod
    def isInteracting():
        pass

class HumanPlayer(Player):
    def __init__(self, health=3, initial_pos=(500, 400)):
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

    def isInteracting(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            return True

    

if __name__ == "__main__":
    dc = DungeonController(visuals=True, human=True)
    dc.run()