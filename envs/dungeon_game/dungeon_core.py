from abc import ABC, abstractmethod
import pygame
import math
import os

# Global constants
playerRadius = 15
coinRadius = 10
gamewin = False

class GameObject(ABC):
    @abstractmethod
    def get_render_data(self):
        """Return data needed for rendering this object"""
        pass

class Moveable(GameObject):
    @abstractmethod
    def move(self):
        pass

class Interactable(GameObject):
    @abstractmethod
    def collidePlayer(self, player):
        pass

    @abstractmethod
    def detectCollision(self, player):
        pass


class DungeonController:
    def __init__(self, visuals=False, human=False):
        pygame.init()
        self.running = True
        self.visuals = visuals
        
        if self.visuals:
            self.renderer = dungeonVisual()
        else:
            self.renderer = None
            
        if human:
            self.player = HumanPlayer()
        self.rooms = DungeonConstructor.constructDungeon()
        self.activeRoom = 0

    def run(self):
        global gamewin
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if gamewin:
                    self.running = False
                if self.player.health <= 0:
                    self.running = False
            
            self.player.move()
            door_result = self.rooms[self.activeRoom].checkCollisions(self.player)
            
            # Handle room transitions
            if door_result is not None:
                next_room_index = self.convert_label_to_index(door_result['next_room'])
                if next_room_index is not None:
                    from_direction = door_result['from_direction']
                    self.activeRoom = next_room_index
                    self.update_room_spawn(from_direction)
                    self.player.x = self.rooms[self.activeRoom].spawnpoint[0]
                    self.player.y = self.rooms[self.activeRoom].spawnpoint[1]
                    self.player.x_speed = 0
                    self.player.y_speed = 0

            if self.visuals and self.renderer:
                self.renderer.render_frame(
                    self.player,
                    self.rooms[self.activeRoom]
                )
    
    def update_room_spawn(self, from_direction):
        """Update room's spawn point based on which door was entered"""
        opposite = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }
        
        spawn_direction = opposite[from_direction]
        
        for item in self.rooms[self.activeRoom].collidables:
            if isinstance(item, Door) and item.direction == spawn_direction:
                if spawn_direction == 'up':
                    new_spawn = (item.x + 36, 105)
                elif spawn_direction == 'down':
                    new_spawn = (item.x + 36, 615)
                elif spawn_direction == 'left':
                    new_spawn = (105, item.y + 36)
                elif spawn_direction == 'right':
                    new_spawn = (615, item.y + 36)
                
                self.rooms[self.activeRoom].spawnpoint = new_spawn
                for obj in self.rooms[self.activeRoom].collidables:
                    if isinstance(obj, PitHazard):
                        obj.updateSpawn(new_spawn)
                return
    
    def convert_label_to_index(self, label):
        """Convert room labels (1-9, A-G) to list indices (0-15)"""
        if label.isdigit():
            index = int(label) - 1
            if 0 <= index < 9:
                return index
        elif label.isalpha():
            index = 9 + (ord(label.upper()) - ord('A'))
            if 9 <= index < 16:
                return index
        return None


class DungeonConstructor:
    @staticmethod
    def constructDungeon():
        rooms = []
        for i in range(16):
            if i < 9:
                index = str(i + 1)
            else:
                index = chr(ord('A') + (i - 9))
            
            filename = f'room{index}.txt'
            new_room = Room(filename)
            rooms.append(new_room)
        
        return rooms


class Coin(Interactable):
    def __init__(self, initial_pos=(72, 72)):
        self.x = initial_pos[0] + 36
        self.y = initial_pos[1] + 36 
        self.collected = False

    def get_render_data(self):
        """Return coin rendering data"""
        return {
            'type': 'coin',
            'x': self.x,
            'y': self.y,
            'radius': coinRadius,
            'collected': self.collected
        }

    def collidePlayer(self, player):
        player.gold += 1
        self.collected = True

    def detectCollision(self, player):
        if self.collected:
            return None

        dx = self.x - player.x
        dy = self.y - player.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < abs(coinRadius + playerRadius):
            self.collidePlayer(player)
        
        return None

class Exit(Interactable):
    def __init__(self, initial_pos=(72, 72)):
        self.x = initial_pos[0] + 36
        self.y = initial_pos[1] + 36 
        self.collected = False

    def get_render_data(self):
        """Return exit rendering data"""
        return {
            'type': 'exit',
            'x': self.x,
            'y': self.y,
            'radius': 10,
            'collected': self.collected
        }

    def collidePlayer(self, player):
        global gamewin
        gamewin = True
        self.collected = True

    def detectCollision(self, player):
        if self.collected:
            return None

        dx = self.x - player.x
        dy = self.y - player.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < abs(coinRadius + playerRadius):
            self.collidePlayer(player)
            return {'exit_reached': True}
        
        return None


class Door(Interactable):
    def __init__(self, goes_to=0, facing='up', initial_pos=(0, 0)):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.nextroom = goes_to
        self.direction = facing

    def get_render_data(self):
        """Return door rendering data"""
        return {
            'type': 'door',
            'x': self.x,
            'y': self.y,
            'direction': self.direction
        }

    def collidePlayer(self, player):
        return True

    def detectCollision(self, player):
        if self.direction == 'up':
            if player.y <= 100 and player.x >= self.x and player.x <= self.x + 72:
                return {'next_room': self.nextroom, 'from_direction': 'up'}
                
        elif self.direction == 'down':
            if player.y >= 620 and player.x >= self.x and player.x <= self.x + 72:
                return {'next_room': self.nextroom, 'from_direction': 'down'}
                
        elif self.direction == 'left':
            if player.x <= 100 and player.y >= self.y and player.y <= self.y + 72:
                return {'next_room': self.nextroom, 'from_direction': 'left'}
                
        elif self.direction == 'right':
            if player.x >= 620 and player.y >= self.y and player.y <= self.y + 72:
                return {'next_room': self.nextroom, 'from_direction': 'right'}
        
        return None


class PitHazard(Interactable):
    def __init__(self, initial_pos=(144, 144), room_spawn=(72, 72)):
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.room_spawn = room_spawn

    def get_render_data(self):
        """Return pit hazard rendering data"""
        return {
            'type': 'pit',
            'x': self.x,
            'y': self.y
        }

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
        
        if distance_squared <= playerRadius * playerRadius:
            self.collidePlayer(player)
        
        return None

    def updateSpawn(self, new_spawn):
        self.room_spawn = new_spawn


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
            'i': 'initial',
            ',': 'exit'  # Added exit tile
        }

        tileset = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    row = []
                    for ch in line.strip():
                        if ch in tile_mapping:
                            row.append(tile_mapping[ch])
                        else:
                            row.append({'door_to': ch})
                    tileset.append(row)
            return tileset
        except FileNotFoundError:
            print(f"Error: Missing {file_path}, using blank room.")
            return [['empty' for _ in range(10)] for _ in range(10)]

    def __init__(self, filename='defaultroom.txt'):
        self.size = 10
        self.tiles = self.read_tileset(filename)
        self.collidables = []
        self.spawnpoint = (72, 72)

        # Find spawnpoint
        for y in range(self.size):
            for x in range(self.size):
                if self.tiles[y][x] == 'initial':
                    self.spawnpoint = (x * 72 + 36, y * 72 + 36)

        # Build interactables
        for y in range(self.size):
            for x in range(self.size):
                tile = self.tiles[y][x]
                tile_pos = (x * 72, y * 72)

                if tile == 'pitfall':
                    self.collidables.append(PitHazard(initial_pos=tile_pos, room_spawn=self.spawnpoint))

                elif tile == 'coin':
                    self.collidables.append(Coin(initial_pos=tile_pos))
                
                elif tile == 'exit':
                    self.collidables.append(Exit(initial_pos=tile_pos))

                elif isinstance(tile, dict) and 'door_to' in tile:
                    door_label = tile['door_to']
                    facing = None
                    if y == 0:
                        facing = 'up'
                    elif y == self.size - 1:
                        facing = 'down'
                    elif x == 0:
                        facing = 'left'
                    elif x == self.size - 1:
                        facing = 'right'

                    self.collidables.append(
                        Door(goes_to=door_label, facing=facing, initial_pos=tile_pos)
                    )

    def get_render_data(self):
        """Return room rendering data"""
        return {
            'type': 'room',
            'size': self.size,
            'tiles': self.tiles,
            'collidables': self.collidables
        }
    
    def checkCollisions(self, player):
        for item in self.collidables:
            result = item.detectCollision(player)
            if result is not None:
                return result
        return None


class Player(Moveable):
    @abstractmethod
    def move(self):
        pass
    
    @abstractmethod
    def isInteracting(self):
        pass


class HumanPlayer(Player):
    def __init__(self, health=3, initial_pos=(500, 400)):
        self.health = health
        self.gold = 0
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.x_speed = 0.0
        self.y_speed = 0.0

    def get_render_data(self):
        """Return player rendering data"""
        return {
            'type': 'player',
            'x': self.x,
            'y': self.y,
            'radius': playerRadius,
            'health': self.health,
            'gold': self.gold
        }

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

        # Hard locking inside the boundary
        if self.x <= 87:
            self.x = 87
        elif self.x >= 633:
            self.x = 633
        if self.y <= 87:
            self.y = 87
        elif self.y >= 633:
            self.y = 633

        # Speed decay
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


class dungeonVisual:
    """Handles all visual rendering for the dungeon game"""
    
    def __init__(self, width=1080, height=720):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dungeon Explorer")
        
        # Initialize font for HUD
        pygame.font.init()
        self.hud_font = pygame.font.SysFont('Arial', 36)
        
        # Cache for room surfaces
        self.room_cache = {}

    def getScreen(self):
        """Legacy method for compatibility"""
        return self.screen
    
    def render_frame(self, player, room):
        """Render a complete frame with room, player, and HUD"""
        self.screen.fill((255, 255, 255))
        self._render_room(room)
        self._render_player(player)
        self._render_hud(player)
        pygame.display.flip()
    
    def _render_room(self, room):
        """Render the room tiles and objects"""
        room_data = room.get_render_data()
        room_id = id(room)
        
        # Check cache for room base (tiles only)
        if room_id not in self.room_cache:
            self.room_cache[room_id] = self._create_room_surface(room_data)
        
        # Blit cached room tiles
        self.screen.blit(self.room_cache[room_id], (0, 0))
        
        # Render dynamic objects (coins, doors, pits)
        for obj in room_data['collidables']:
            self._render_object(obj)
    
    def _create_room_surface(self, room_data):
        """Create a cached surface for room tiles"""
        surface = pygame.Surface((720, 720))
        
        for j in range(room_data['size']):
            for i in range(room_data['size']):
                pos_x = i * 72
                pos_y = j * 72
                tile = room_data['tiles'][j][i]
                
                if tile == 'wall':
                    colour = (0, 0, 0)
                elif tile == 'pitfall':
                    colour = (200, 0, 0)
                elif tile == 'exit':
                    colour = (0, 200, 0)
                else:
                    colour = (100, 100, 100)
                
                pygame.draw.rect(surface, colour, (pos_x, pos_y, 72, 72))
                pygame.draw.rect(surface, (0, 0, 0), (pos_x, pos_y, 72, 72), 1)
        
        return surface
    
    def _render_object(self, obj):
        """Render individual game objects"""
        obj_data = obj.get_render_data()
        
        if obj_data['type'] == 'coin':
            if not obj_data['collected']:
                pygame.draw.circle(
                    self.screen,
                    (200, 200, 0),
                    (obj_data['x'], obj_data['y']),
                    obj_data['radius']
                )
        
        elif obj_data['type'] == 'door':
            pygame.draw.circle(
                self.screen,
                (100, 0, 100),
                (obj_data['x'] + 36, obj_data['y'] + 36),
                5
            )
        
        elif obj_data['type'] == 'pit':
            pygame.draw.circle(
                self.screen,
                (150, 150, 150),
                (obj_data['x'] + 36, obj_data['y'] + 36),
                20
            )
        elif obj_data['type'] == 'exit':
            if not obj_data['collected']:
                pygame.draw.circle(
                    self.screen,
                    (0, 255, 0),
                    (obj_data['x'], obj_data['y']),
                    15
                )
    
    def _render_player(self, player):
        """Render the player"""
        player_data = player.get_render_data()
        pygame.draw.circle(
            self.screen,
            (255, 50, 0),
            (player_data['x'], player_data['y']),
            player_data['radius']
        )
    
    def _render_hud(self, player):
        """Render the heads-up display"""
        player_data = player.get_render_data()
        
        text1 = self.hud_font.render(f"Gold: {player_data['gold']}", True, (0, 0, 0))
        text2 = self.hud_font.render(f"Health: {player_data['health']}", True, (0, 0, 0))
        
        self.screen.blit(text1, (900, 300))
        self.screen.blit(text2, (900, 420))
    
    def clear_cache(self):
        """Clear all cached surfaces"""
        self.room_cache.clear()


if __name__ == "__main__":
    dc = DungeonController(visuals=True, human=True)
    dc.run()