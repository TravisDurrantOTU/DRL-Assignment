import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any
import math

# Import your existing game classes
# Assuming the original code is in a file called 'dungeon_game.py'
# from dungeon_game import *

# For now, I'll include the necessary constants
PLAYER_RADIUS = 15
COIN_RADIUS = 10
ROOM_SIZE = 10
TILE_SIZE = 72
MAX_STEPS = 1000


class DungeonEnv(gym.Env):
    """
    Gymnasium environment for the dungeon exploration game.
    
    Observation Space:
        - Player position (x, y)
        - Player velocity (vx, vy)
        - Player stats (health, gold)
        - Room grid (flattened 10x10 one-hot encoded tiles)
        - Relative positions of nearby collectibles/hazards
        
    Action Space:
        - Discrete(5): [No-op, Up, Down, Left, Right]
        or
        - Box(2): Continuous control for dx, dy acceleration
    
    Reward:
        - +10 for collecting a coin
        - -5 for falling into a pit
        - -10 for losing all health (terminal)
        - +50 for entering a new room (exploration bonus)
        - -0.01 per step (time penalty)
        - +100 for reaching goal (if specified)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous_actions: bool = False,
        max_steps: int = MAX_STEPS,
        start_room: int = 0,
        goal_room: Optional[int] = None,
        observation_mode: str = 'vector'  # 'vector' or 'grid'
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.start_room = start_room
        self.goal_room = goal_room
        self.observation_mode = observation_mode
        
        # Action space
        if continuous_actions:
            # Continuous control: dx, dy acceleration
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        else:
            # Discrete actions: [no-op, up, down, left, right]
            self.action_space = spaces.Discrete(5)
        
        # Observation space
        if observation_mode == 'vector':
            # Vector observation
            obs_size = (
                2 +  # player position (normalized)
                2 +  # player velocity
                2 +  # player stats (health, gold)
                100 +  # room grid (10x10 flattened, simplified encoding)
                20   # nearby objects features
            )
            self.observation_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(obs_size,),
                dtype=np.float32
            )
        else:
            # Grid-based observation (for CNN policies)
            self.observation_space = spaces.Dict({
                'grid': spaces.Box(
                    low=0,
                    high=5,
                    shape=(ROOM_SIZE, ROOM_SIZE),
                    dtype=np.uint8
                ),
                'player_state': spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(6,),
                    dtype=np.float32
                )
            })
        
        # Initialize game components
        self._init_game()
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0
        self.visited_rooms = set()
        self.prev_gold = 0
        self.prev_health = 3
        
        # Rendering
        if self.render_mode == 'human':
            pygame.init()
            from dungeon_core import dungeonVisual
            self.renderer = dungeonVisual()
    
    def _init_game(self):
        """Initialize game components without rendering"""
        pygame.init()
        from dungeon_core import DungeonConstructor, HumanPlayer
        
        self.rooms = DungeonConstructor.constructDungeon()
        self.active_room = self.start_room
        
        # Set player to the correct room's spawn point
        spawn = self.rooms[self.active_room].spawnpoint
        self.player = HumanPlayer(health=3, initial_pos=spawn)
        self.player.x = spawn[0]
        self.player.y = spawn[1]
        self.player.x_speed = 0
        self.player.y_speed = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game state
        self._init_game()
        
        # Reset tracking variables
        self.current_step = 0
        self.total_reward = 0
        self.visited_rooms = {self.start_room}
        self.prev_gold = 0
        self.prev_health = 3
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Move player
        self.player.move()
        
        # Store previous state for reward calculation
        prev_room = self.active_room
        prev_gold = self.player.gold
        prev_health = self.player.health
        
        # Check collisions
        door_result = self.rooms[self.active_room].checkCollisions(self.player)
        
        # Handle room transitions
        if door_result is not None:
            next_room_index = self._convert_label_to_index(door_result['next_room'])
            if next_room_index is not None:
                from_direction = door_result['from_direction']
                self.active_room = next_room_index
                self._update_room_spawn(from_direction)
                self.player.x = self.rooms[self.active_room].spawnpoint[0]
                self.player.y = self.rooms[self.active_room].spawnpoint[1]
                self.player.x_speed = 0
                self.player.y_speed = 0
        
        # Calculate reward
        reward = self._calculate_reward(prev_room, prev_gold, prev_health)
        
        # Check terminal conditions
        terminated = self.player.health <= 0
        if self.goal_room is not None and self.active_room == self.goal_room:
            reward += 100.0
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Update tracking
        self.total_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply the action to the player"""
        if self.continuous_actions:
            # Continuous action
            dx = float(action[0]) * 0.1
            dy = float(action[1]) * 0.1
            self.player.update_velocity(dx=dx, dy=dy)
        else:
            # Discrete action
            action_map = {
                0: (0, 0),      # no-op
                1: (0, -0.1),   # up
                2: (0, 0.1),    # down
                3: (-0.1, 0),   # left
                4: (0.1, 0)     # right
            }
            dx, dy = action_map[action]
            self.player.update_velocity(dx=dx, dy=dy)
    
    def _calculate_reward(
        self,
        prev_room: int,
        prev_gold: int,
        prev_health: int
    ) -> float:
        """Calculate reward for the current step"""
        reward = -0.01  # Small time penalty
        
        # Coin collection
        if self.player.gold > prev_gold:
            reward += 10.0 * (self.player.gold - prev_gold)
        
        # Health loss (pit hazard)
        if self.player.health < prev_health:
            reward -= 5.0 * (prev_health - self.player.health)
        
        # Death penalty
        if self.player.health <= 0:
            reward -= 10.0
        
        # Room exploration bonus
        if self.active_room != prev_room:
            if self.active_room not in self.visited_rooms:
                reward += 50.0
                self.visited_rooms.add(self.active_room)
            else:
                reward += 5.0  # Small bonus for room transitions
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.observation_mode == 'vector':
            return self._get_vector_observation()
        else:
            return self._get_grid_observation()
    
    def _get_vector_observation(self) -> np.ndarray:
        """Get vector-based observation"""
        obs = []
        
        # Player position (normalized to [-1, 1])
        obs.extend([
            (self.player.x - 360) / 360,
            (self.player.y - 360) / 360
        ])
        
        # Player velocity
        obs.extend([
            self.player.x_speed,
            self.player.y_speed
        ])
        
        # Player stats (normalized)
        obs.extend([
            self.player.health / 3.0,
            self.player.gold / 10.0  # Assume max ~10 coins per episode
        ])
        
        # Room grid (simplified encoding)
        room = self.rooms[self.active_room]
        grid_encoding = self._encode_room_grid(room)
        obs.extend(grid_encoding)
        
        # Nearby objects
        nearby_features = self._get_nearby_objects()
        obs.extend(nearby_features)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_grid_observation(self) -> Dict[str, np.ndarray]:
        """Get grid-based observation for CNN policies"""
        room = self.rooms[self.active_room]
        
        # Create grid representation
        grid = np.zeros((ROOM_SIZE, ROOM_SIZE), dtype=np.uint8)
        tile_map = {
            'empty': 0,
            'wall': 1,
            'pitfall': 2,
            'coin': 3,
            'initial': 0
        }
        
        for j in range(ROOM_SIZE):
            for i in range(ROOM_SIZE):
                tile = room.tiles[j][i]
                if isinstance(tile, dict):
                    grid[j, i] = 4  # door
                else:
                    grid[j, i] = tile_map.get(tile, 0)
        
        # Player state
        player_state = np.array([
            (self.player.x - 360) / 360,
            (self.player.y - 360) / 360,
            self.player.x_speed,
            self.player.y_speed,
            self.player.health / 3.0,
            self.player.gold / 10.0
        ], dtype=np.float32)
        
        return {
            'grid': grid,
            'player_state': player_state
        }
    
    def _encode_room_grid(self, room) -> list:
        """Encode room grid as flattened feature vector"""
        encoding = []
        tile_map = {
            'empty': 0,
            'wall': 1,
            'pitfall': 2,
            'coin': 3,
            'initial': 0
        }
        
        for j in range(ROOM_SIZE):
            for i in range(ROOM_SIZE):
                tile = room.tiles[j][i]
                if isinstance(tile, dict):
                    encoding.append(4)  # door
                else:
                    encoding.append(tile_map.get(tile, 0))
        
        return encoding
    
    def _get_nearby_objects(self) -> list:
        """Get features of nearby objects (coins, pits, doors)"""
        features = []
        room = self.rooms[self.active_room]
        
        # Find nearest coin, pit, and door
        nearest_coin = None
        nearest_pit = None
        nearest_door = None
        
        min_coin_dist = float('inf')
        min_pit_dist = float('inf')
        min_door_dist = float('inf')
        
        for obj in room.collidables:
            from dungeon_core import Coin, PitHazard, Door
            
            if isinstance(obj, Coin) and not obj.collected:
                dist = math.sqrt((obj.x - self.player.x)**2 + (obj.y - self.player.y)**2)
                if dist < min_coin_dist:
                    min_coin_dist = dist
                    nearest_coin = (obj.x, obj.y)
            
            elif isinstance(obj, PitHazard):
                dist = math.sqrt((obj.x + 36 - self.player.x)**2 + (obj.y + 36 - self.player.y)**2)
                if dist < min_pit_dist:
                    min_pit_dist = dist
                    nearest_pit = (obj.x + 36, obj.y + 36)
            
            elif isinstance(obj, Door):
                dist = math.sqrt((obj.x + 36 - self.player.x)**2 + (obj.y + 36 - self.player.y)**2)
                if dist < min_door_dist:
                    min_door_dist = dist
                    nearest_door = (obj.x + 36, obj.y + 36)
        
        # Encode relative positions (normalized)
        if nearest_coin:
            features.extend([
                (nearest_coin[0] - self.player.x) / 360,
                (nearest_coin[1] - self.player.y) / 360,
                min_coin_dist / 500
            ])
        else:
            features.extend([0, 0, 1.0])
        
        if nearest_pit:
            features.extend([
                (nearest_pit[0] - self.player.x) / 360,
                (nearest_pit[1] - self.player.y) / 360,
                min_pit_dist / 500
            ])
        else:
            features.extend([0, 0, 1.0])
        
        if nearest_door:
            features.extend([
                (nearest_door[0] - self.player.x) / 360,
                (nearest_door[1] - self.player.y) / 360,
                min_door_dist / 500
            ])
        else:
            features.extend([0, 0, 1.0])
        
        # Direction to nearest objects (one-hot encoded roughly)
        # Add 11 more features to reach 20 total
        features.extend([0] * 11)
        
        return features
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary"""
        return {
            'player_x': self.player.x,
            'player_y': self.player.y,
            'health': self.player.health,
            'gold': self.player.gold,
            'active_room': self.active_room,
            'visited_rooms': len(self.visited_rooms),
            'total_reward': self.total_reward,
            'steps': self.current_step
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            self.renderer.render_frame(
                self.player,
                self.rooms[self.active_room]
            )
            pygame.time.Clock().tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render and return RGB array"""
        if not hasattr(self, 'rgb_renderer'):
            from dungeon_core import dungeonVisual
            self.rgb_renderer = dungeonVisual()
        
        self.rgb_renderer.render_frame(
            self.player,
            self.rooms[self.active_room]
        )
        
        # Convert pygame surface to numpy array
        surface = self.rgb_renderer.screen
        rgb_array = pygame.surfarray.array3d(surface)
        return np.transpose(rgb_array, (1, 0, 2))
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'renderer'):
            pygame.quit()
    
    def _update_room_spawn(self, from_direction: str):
        """Update room's spawn point based on which door was entered"""
        opposite = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }
        
        spawn_direction = opposite[from_direction]
        
        from dungeon_core import Door, PitHazard
        
        for item in self.rooms[self.active_room].collidables:
            if isinstance(item, Door) and item.direction == spawn_direction:
                if spawn_direction == 'up':
                    new_spawn = (item.x + 36, 105)
                elif spawn_direction == 'down':
                    new_spawn = (item.x + 36, 615)
                elif spawn_direction == 'left':
                    new_spawn = (105, item.y + 36)
                elif spawn_direction == 'right':
                    new_spawn = (615, item.y + 36)
                
                self.rooms[self.active_room].spawnpoint = new_spawn
                for obj in self.rooms[self.active_room].collidables:
                    if isinstance(obj, PitHazard):
                        obj.updateSpawn(new_spawn)
                return
    
    def _convert_label_to_index(self, label: str) -> Optional[int]:
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


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    env = DungeonEnv(render_mode='human', continuous_actions=False)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run random agent
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished: Gold={info['gold']}, Health={info['health']}")
            obs, info = env.reset()
    
    env.close()