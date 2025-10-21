import pygame
import math
import random
from abc import ABC, abstractmethod

# Game constants
MAX_SPEED = 5
ACCELERATION = 8
FRICTION = 3
TURN_SPEED = 300

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class GameObject(ABC):
    @abstractmethod
    def get_render_data(self):
        """Return data needed for rendering this object"""
        pass


class Moveable(GameObject):
    @abstractmethod
    def move(self):
        pass


class Collectible(GameObject):
    """Represents a target that can be collected"""
    
    def __init__(self, x, y, radius=15, value=1.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.value = value
        self.active = True
    
    def get_render_data(self):
        """Return target rendering data"""
        return {
            'type': 'target',
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'active': self.active
        }
    
    def check_collision(self, agent_x, agent_y, agent_radius):
        """Check if agent collides with this target"""
        if not self.active:
            return False
        
        dist = math.hypot(agent_x - self.x, agent_y - self.y)
        return dist < (self.radius + agent_radius)
    
    def collect(self):
        """Mark this target as collected"""
        self.active = False


class Obstacle(GameObject):
    """Represents a static obstacle"""
    
    def __init__(self, x, y, radius=25):
        self.x = x
        self.y = y
        self.radius = radius
    
    def get_render_data(self):
        """Return obstacle rendering data"""
        return {
            'type': 'obstacle',
            'x': self.x,
            'y': self.y,
            'radius': self.radius
        }
    
    def check_collision(self, agent_x, agent_y, agent_radius):
        """Check if agent collides with this obstacle"""
        dist = math.hypot(agent_x - self.x, agent_y - self.y)
        return dist < (self.radius + agent_radius)


class PlayArea:
    """Defines the play area boundaries"""
    
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, margin=30):
        self.width = width
        self.height = height
        self.margin = margin
        self.min_x = margin
        self.max_x = width - margin
        self.min_y = margin
        self.max_y = height - margin
    
    def get_render_data(self):
        """Return play area rendering data"""
        return {
            'type': 'play_area',
            'width': self.width,
            'height': self.height,
            'margin': self.margin
        }
    
    def check_out_of_bounds(self, x, y):
        """Check if position is out of bounds"""
        return (x < self.min_x or x > self.max_x or 
                y < self.min_y or y > self.max_y)
    
    def clamp_position(self, x, y):
        """Clamp position to stay within bounds"""
        x = max(self.min_x, min(x, self.max_x))
        y = max(self.min_y, min(y, self.max_y))
        return x, y


class Agent(Moveable):
    """The agent that navigates and collects targets"""
    
    START_X = WINDOW_WIDTH // 2
    START_Y = WINDOW_HEIGHT // 2
    START_ANGLE = 0
    START_SPEED = 0
    RADIUS = 12
    COLOR = (0, 120, 255)
    
    def __init__(self):
        self.x = self.START_X
        self.y = self.START_Y
        self.old_x = self.x
        self.old_y = self.y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED
        self.radius = self.RADIUS
        
        # Stats
        self.targets_collected = 0
        self.obstacles_hit = 0
    
    def get_render_data(self):
        """Return agent rendering data"""
        return {
            'type': 'agent',
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'radius': self.radius,
            'color': self.COLOR
        }
    
    def update_velocity(self, dt, acceleration=0, steering=0):
        """
        Update velocity based on continuous control inputs
        
        Args:
            dt: Time delta
            acceleration: Value in [-1, 1] for forward/backward
            steering: Value in [-1, 1] for left/right turning
        """
        # Apply acceleration
        if acceleration > 0:
            self.speed += ACCELERATION * dt * acceleration
        elif acceleration < 0:
            # Braking/reverse
            if self.speed > 0:
                self.speed = max(0, self.speed + ACCELERATION * dt * acceleration * 2)
            else:
                self.speed += ACCELERATION * dt * acceleration * 0.5
        else:
            # Apply friction when no input
            if abs(self.speed) > 0:
                friction_force = FRICTION * dt
                if self.speed > 0:
                    self.speed = max(0, self.speed - friction_force)
                else:
                    self.speed = min(0, self.speed + friction_force)
        
        # Clamp speed
        self.speed = max(-MAX_SPEED / 2, min(self.speed, MAX_SPEED))
        
        # Apply steering (speed-dependent)
        if abs(self.speed) > 0.1:
            turn_factor = abs(self.speed) / MAX_SPEED
            self.angle += TURN_SPEED * dt * turn_factor * steering
            
            # Normalize angle
            self.angle = self.angle % 360
    
    def move(self):
        """Move agent based on current speed and angle"""
        self.old_x = self.x
        self.old_y = self.y
        
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
    
    def reset(self):
        """Reset agent to starting position"""
        self.x = self.START_X
        self.y = self.START_Y
        self.old_x = self.x
        self.old_y = self.y
        self.angle = self.START_ANGLE
        self.speed = self.START_SPEED
        self.targets_collected = 0
        self.obstacles_hit = 0


class GameWorld:
    """Manages the game world with targets and obstacles"""
    
    def __init__(self, num_targets=8, num_obstacles=6, seed=None):
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.rng = random.Random(seed)
        
        self.play_area = PlayArea()
        self.targets = []
        self.obstacles = []
        
        self._generate_world()
    
    def _generate_world(self):
        """Generate targets and obstacles"""
        self.targets = []
        self.obstacles = []
        
        # Generate obstacles first
        for _ in range(self.num_obstacles):
            while True:
                x = self.rng.uniform(100, WINDOW_WIDTH - 100)
                y = self.rng.uniform(100, WINDOW_HEIGHT - 100)
                
                # Check not too close to center (spawn point)
                dist_to_center = math.hypot(x - WINDOW_WIDTH/2, y - WINDOW_HEIGHT/2)
                if dist_to_center > 80:
                    # Check not too close to other obstacles
                    valid = True
                    for obs in self.obstacles:
                        if math.hypot(x - obs.x, y - obs.y) < 70:
                            valid = False
                            break
                    if valid:
                        self.obstacles.append(Obstacle(x, y))
                        break
        
        # Generate targets
        for _ in range(self.num_targets):
            while True:
                x = self.rng.uniform(80, WINDOW_WIDTH - 80)
                y = self.rng.uniform(80, WINDOW_HEIGHT - 80)
                
                # Check not too close to obstacles
                valid = True
                for obs in self.obstacles:
                    if math.hypot(x - obs.x, y - obs.y) < 50:
                        valid = False
                        break
                
                # Check not too close to center
                dist_to_center = math.hypot(x - WINDOW_WIDTH/2, y - WINDOW_HEIGHT/2)
                if valid and dist_to_center > 60:
                    self.targets.append(Collectible(x, y))
                    break
    
    def get_render_data(self):
        """Return world rendering data"""
        return {
            'type': 'game_world',
            'play_area': self.play_area.get_render_data(),
            'targets': [t.get_render_data() for t in self.targets],
            'obstacles': [o.get_render_data() for o in self.obstacles]
        }
    
    def check_target_collection(self, agent):
        """Check if agent collected any targets"""
        collected = []
        for target in self.targets:
            if target.check_collision(agent.x, agent.y, agent.radius):
                target.collect()
                collected.append(target)
        return collected
    
    def check_obstacle_collision(self, agent):
        """Check if agent hit any obstacles"""
        for obstacle in self.obstacles:
            if obstacle.check_collision(agent.x, agent.y, agent.radius):
                return True
        return False
    
    def get_active_targets(self):
        """Return list of active (uncollected) targets"""
        return [t for t in self.targets if t.active]
    
    def all_targets_collected(self):
        """Check if all targets have been collected"""
        return all(not t.active for t in self.targets)
    
    def reset(self):
        """Reset world state"""
        self._generate_world()


class GameVisual:
    """Handles all visual rendering for the target collection game"""
    
    # Color constants
    BACKGROUND_COLOR = (240, 240, 240)
    BOUNDARY_COLOR = (100, 100, 100)
    TARGET_COLOR = (50, 200, 50)
    TARGET_INACTIVE_COLOR = (150, 150, 150)
    OBSTACLE_COLOR = (200, 50, 50)
    TEXT_COLOR = (0, 0, 0)
    
    def __init__(self, window_size=(WINDOW_WIDTH, WINDOW_HEIGHT)):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Target Collection Game")
        
        # HUD font
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)
    
    def render_frame(self, world, agent, fps):
        """Render a complete frame"""
        self.screen.fill(self.BACKGROUND_COLOR)
        self._render_play_area(world.play_area)
        self._render_obstacles(world.obstacles)
        self._render_targets(world.targets)
        self._render_agent(agent)
        self._render_hud(fps, agent, world)
        pygame.display.flip()
    
    def _render_play_area(self, play_area):
        """Render the play area boundaries"""
        data = play_area.get_render_data()
        margin = data['margin']
        
        # Draw boundary rectangle
        rect = pygame.Rect(margin, margin, 
                          data['width'] - 2*margin, 
                          data['height'] - 2*margin)
        pygame.draw.rect(self.screen, self.BOUNDARY_COLOR, rect, 2)
    
    def _render_targets(self, targets):
        """Render all targets"""
        for target in targets:
            data = target.get_render_data()
            color = self.TARGET_COLOR if data['active'] else self.TARGET_INACTIVE_COLOR
            
            pos = (int(data['x']), int(data['y']))
            radius = int(data['radius'])
            
            # Draw target with outline
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)
            
            # Draw inner circle for active targets
            if data['active']:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, radius // 3)
    
    def _render_obstacles(self, obstacles):
        """Render all obstacles"""
        for obstacle in obstacles:
            data = obstacle.get_render_data()
            pos = (int(data['x']), int(data['y']))
            radius = int(data['radius'])
            
            # Draw obstacle with gradient effect
            pygame.draw.circle(self.screen, self.OBSTACLE_COLOR, pos, radius)
            pygame.draw.circle(self.screen, (150, 30, 30), pos, radius - 3)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)
    
    def _render_agent(self, agent):
        """Render the agent"""
        data = agent.get_render_data()
        pos = (int(data['x']), int(data['y']))
        radius = int(data['radius'])
        
        # Draw agent body
        pygame.draw.circle(self.screen, data['color'], pos, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)
        
        # Draw direction indicator
        rad = math.radians(data['angle'])
        end_x = data['x'] + math.cos(rad) * (radius + 8)
        end_y = data['y'] + math.sin(rad) * (radius + 8)
        pygame.draw.line(self.screen, (255, 255, 255), pos, 
                        (int(end_x), int(end_y)), 3)
    
    def _render_hud(self, fps, agent, world):
        """Render the heads-up display"""
        active_targets = len(world.get_active_targets())
        total_targets = len(world.targets)
        
        lines = [
            f"FPS: {int(fps)}",
            f"Speed: {abs(agent.speed):.1f}",
            f"Targets: {total_targets - active_targets}/{total_targets}",
            f"Obstacles Hit: {agent.obstacles_hit}"
        ]
        
        y_offset = 10
        for line in lines:
            surface = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 30


class GameController:
    """Main controller for the target collection game"""
    
    def __init__(self, visuals=True):
        self.running = True
        self.visuals = visuals
        self.clock = pygame.time.Clock()
        
        if self.visuals:
            self.renderer = GameVisual()
        else:
            self.renderer = None
        
        self.agent = Agent()
        self.world = GameWorld(num_targets=8, num_obstacles=6)
    
    def run(self):
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.agent.reset()
                        self.world.reset()
            
            # Get keyboard input
            keys = pygame.key.get_pressed()
            acceleration = 0
            steering = 0
            
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                acceleration = 1.0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                acceleration = -1.0
            
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steering = -1.0
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steering = 1.0
            
            # Update
            self.agent.update_velocity(dt, acceleration, steering)
            self.agent.move()
            
            # Clamp position to play area
            self.agent.x, self.agent.y = self.world.play_area.clamp_position(
                self.agent.x, self.agent.y
            )
            
            # Check collisions
            collected = self.world.check_target_collection(self.agent)
            if collected:
                self.agent.targets_collected += len(collected)
            
            if self.world.check_obstacle_collision(self.agent):
                self.agent.obstacles_hit += 1
                # Push agent back
                self.agent.x = self.agent.old_x
                self.agent.y = self.agent.old_y
                self.agent.speed *= -0.5
            
            # Check win condition
            if self.world.all_targets_collected():
                print(f"All targets collected! Obstacles hit: {self.agent.obstacles_hit}")
                self.agent.reset()
                self.world.reset()
            
            # Render
            if self.visuals and self.renderer:
                self.renderer.render_frame(self.world, self.agent, 
                                          self.clock.get_fps())
        
        pygame.quit()


if __name__ == "__main__":
    controller = GameController(visuals=True)
    controller.run()
