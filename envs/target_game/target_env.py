import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional, Tuple, Dict, Any


class TargetEnv(gym.Env):
    """
    Gymnasium environment for the target collection game.
    
    Observation Space:
        - Agent speed (normalized)
        - Agent angle (sin and cos components)
        - Agent position (normalized x, y)
        - Distance to nearest active target (normalized)
        - Angle to nearest active target (sin and cos)
        - Distance sensors (8 rays to obstacles/boundaries)
        - Number of active targets remaining (normalized)
        - Distances to closest 3 targets (normalized)
        - Distances to closest 3 obstacles (normalized)
        
    Action Space:
        - Continuous: [acceleration, steering] in [-1, 1]
        
    Reward:
        - Positive reward for collecting targets
        - Small positive reward for moving toward targets
        - Penalty for hitting obstacles
        - Small penalty for staying still
        - Bonus for collecting all targets
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    # Game constants
    MAX_SPEED = 5
    ACCELERATION = 8
    FRICTION = 3
    TURN_SPEED = 300
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 2000,
        window_size: Tuple[int, int] = (800, 600),
        num_targets: int = 8,
        num_obstacles: int = 6,
    ):
        """
        Initialize the target collection environment.
        
        Args:
            render_mode: "human" for window display, "rgb_array" for pixels
            max_steps: Maximum steps per episode
            window_size: Window dimensions
            num_targets: Number of targets to collect
            num_obstacles: Number of obstacles to avoid
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.window_size = window_size
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        
        # Continuous action space: [acceleration, steering]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space
        # [speed, sin(angle), cos(angle), norm_x, norm_y,
        #  dist_to_nearest_target, sin(angle_to_target), cos(angle_to_target),
        #  8x distance_sensors, targets_remaining,
        #  3x closest_target_distances, 3x closest_obstacle_distances]
        obs_dim = 1 + 2 + 2 + 1 + 2 + 8 + 1 + 3 + 3
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Ray casting for distance sensing
        self.num_rays = 8
        self.ray_angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        self.max_ray_distance = 200.0
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_nearest_target_dist = 1.0
        
        # Game objects
        self.world = None
        self.agent = None
        
        # Pygame/rendering
        self.screen = None
        self.clock = None
        self.renderer = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Target Collection RL Environment")
            self.clock = pygame.time.Clock()
            from target_core import GameVisual
            self.renderer = GameVisual(window_size)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize world and agent
        from target_core import GameWorld, Agent
        
        self.world = GameWorld(
            num_targets=self.num_targets,
            num_obstacles=self.num_obstacles,
            seed=seed
        )
        
        self.agent = Agent()
        self.agent.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_nearest_target_dist = self._get_nearest_target_distance()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        dt = 1.0 / 60.0  # Fixed timestep
        
        # Apply action
        acceleration = np.clip(action[0], -1.0, 1.0)
        steering = np.clip(action[1], -1.0, 1.0)
        
        self.agent.update_velocity(dt, acceleration, steering)
        self.agent.move()
        
        # Clamp to play area
        self.agent.x, self.agent.y = self.world.play_area.clamp_position(
            self.agent.x, self.agent.y
        )
        
        # Check target collection
        collected_targets = self.world.check_target_collection(self.agent)
        targets_collected = len(collected_targets)
        if targets_collected > 0:
            self.agent.targets_collected += targets_collected
        
        # Check obstacle collision
        hit_obstacle = self.world.check_obstacle_collision(self.agent)
        if hit_obstacle:
            self.agent.obstacles_hit += 1
            # Push agent back slightly
            self.agent.x = self.agent.old_x
            self.agent.y = self.agent.old_y
            self.agent.speed *= -0.3
        
        # Check if all targets collected
        all_collected = self.world.all_targets_collected()
        
        # Calculate reward
        reward = self._calculate_reward(targets_collected, hit_obstacle, all_collected)
        
        # Check termination conditions
        self.current_step += 1
        terminated = all_collected  # Win condition
        truncated = self.current_step >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['targets_collected'] = targets_collected
        info['hit_obstacle'] = hit_obstacle
        info['all_collected'] = all_collected
        
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        obs = []
        
        # 1. Normalized speed
        obs.append(self.agent.speed / self.MAX_SPEED)
        
        # 2. Agent angle (sin and cos for continuity)
        angle_rad = math.radians(self.agent.angle)
        obs.append(math.sin(angle_rad))
        obs.append(math.cos(angle_rad))
        
        # 3. Normalized agent position
        obs.append((self.agent.x / self.window_size[0]) * 2 - 1)  # [-1, 1]
        obs.append((self.agent.y / self.window_size[1]) * 2 - 1)  # [-1, 1]
        
        # 4. Distance and angle to nearest active target
        active_targets = self.world.get_active_targets()
        if active_targets:
            nearest_target = min(active_targets, 
                               key=lambda t: math.hypot(t.x - self.agent.x, 
                                                       t.y - self.agent.y))
            
            dist = math.hypot(nearest_target.x - self.agent.x, 
                            nearest_target.y - self.agent.y)
            normalized_dist = min(dist / 400.0, 1.0)
            obs.append(normalized_dist)
            
            angle_to_target = math.atan2(nearest_target.y - self.agent.y,
                                        nearest_target.x - self.agent.x)
            relative_angle = angle_to_target - angle_rad
            obs.append(math.sin(relative_angle))
            obs.append(math.cos(relative_angle))
        else:
            obs.extend([1.0, 0.0, 0.0])  # No targets left
        
        # 5. Distance sensors (8 rays)
        ray_distances = self._cast_rays()
        obs.extend(ray_distances)
        
        # 6. Number of active targets remaining (normalized)
        targets_remaining = len(active_targets) / self.num_targets
        obs.append(targets_remaining)
        
        # 7. Distances to closest 3 targets
        target_distances = self._get_closest_target_distances(3)
        obs.extend(target_distances)
        
        # 8. Distances to closest 3 obstacles
        obstacle_distances = self._get_closest_obstacle_distances(3)
        obs.extend(obstacle_distances)
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self) -> list:
        """Cast rays from agent to detect distance to obstacles and boundaries."""
        distances = []
        agent_pos = (self.agent.x, self.agent.y)
        angle_rad = math.radians(self.agent.angle)
        
        for ray_offset in self.ray_angles:
            ray_angle = angle_rad + ray_offset
            ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
            
            min_dist = self.max_ray_distance
            
            # Check distance to boundaries
            if ray_dir[0] != 0:
                if ray_dir[0] > 0:
                    t = (self.world.play_area.max_x - agent_pos[0]) / ray_dir[0]
                else:
                    t = (self.world.play_area.min_x - agent_pos[0]) / ray_dir[0]
                if t > 0:
                    min_dist = min(min_dist, t)
            
            if ray_dir[1] != 0:
                if ray_dir[1] > 0:
                    t = (self.world.play_area.max_y - agent_pos[1]) / ray_dir[1]
                else:
                    t = (self.world.play_area.min_y - agent_pos[1]) / ray_dir[1]
                if t > 0:
                    min_dist = min(min_dist, t)
            
            # Check distance to obstacles
            for obstacle in self.world.obstacles:
                # Ray-circle intersection
                dx = obstacle.x - agent_pos[0]
                dy = obstacle.y - agent_pos[1]
                
                a = ray_dir[0]**2 + ray_dir[1]**2
                b = 2 * (ray_dir[0] * (-dx) + ray_dir[1] * (-dy))
                c = dx**2 + dy**2 - obstacle.radius**2
                
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    t = (-b - math.sqrt(discriminant)) / (2*a)
                    if t > 0:
                        min_dist = min(min_dist, t)
            
            # Normalize distance
            normalized_dist = min(min_dist / self.max_ray_distance, 1.0)
            distances.append(normalized_dist)
        
        return distances
    
    def _get_closest_target_distances(self, num_targets: int) -> list:
        """Get distances to the closest N targets."""
        active_targets = self.world.get_active_targets()
        
        if not active_targets:
            return [1.0] * num_targets
        
        distances = []
        for target in active_targets:
            dist = math.hypot(target.x - self.agent.x, target.y - self.agent.y)
            distances.append(dist)
        
        distances.sort()
        
        # Take closest N, pad if needed
        result = []
        for i in range(num_targets):
            if i < len(distances):
                result.append(min(distances[i] / 400.0, 1.0))
            else:
                result.append(1.0)
        
        return result
    
    def _get_closest_obstacle_distances(self, num_obstacles: int) -> list:
        """Get distances to the closest N obstacles."""
        if not self.world.obstacles:
            return [1.0] * num_obstacles
        
        distances = []
        for obstacle in self.world.obstacles:
            dist = math.hypot(obstacle.x - self.agent.x, obstacle.y - self.agent.y)
            distances.append(dist)
        
        distances.sort()
        
        # Take closest N, pad if needed
        result = []
        for i in range(num_obstacles):
            if i < len(distances):
                result.append(min(distances[i] / 300.0, 1.0))
            else:
                result.append(1.0)
        
        return result
    
    def _get_nearest_target_distance(self) -> float:
        """Get normalized distance to nearest active target."""
        active_targets = self.world.get_active_targets()
        
        if not active_targets:
            return 1.0
        
        nearest = min(active_targets,
                     key=lambda t: math.hypot(t.x - self.agent.x, t.y - self.agent.y))
        
        dist = math.hypot(nearest.x - self.agent.x, nearest.y - self.agent.y)
        return min(dist / 400.0, 1.0)
    
    def _calculate_reward(self, targets_collected: int, hit_obstacle: bool, 
                         all_collected: bool) -> float:
        """Calculate reward for current step."""
        reward = 0.0
        
        # Base survival reward (small)
        reward += 0.1
        
        # Speed incentive (encourage movement)
        speed_ratio = min(abs(self.agent.speed) / self.MAX_SPEED, 0.0)
        reward += speed_ratio * 0.5
        
        # Penalize being too slow
        if abs(self.agent.speed) < 0.5:
            reward -= 0.3
        
        # Target collection (major positive reward)
        if targets_collected > 0:
            reward += 50.0#* targets_collected
        
        # Obstacle collision (penalty)
        # Will likely multi-tick this so needs to be small
        if hit_obstacle:
            reward -= 15.0
        
        # Progress toward nearest target
        current_nearest_dist = self._get_nearest_target_distance()
        progress = self.prev_nearest_target_dist - current_nearest_dist
        reward += progress * 20.0  # Reward for getting closer
        self.prev_nearest_target_dist = current_nearest_dist
        
        # Bonus for collecting all targets
        if all_collected:
            reward += 200.0
            # Additional bonus for efficiency (fewer obstacles hit)
            reward += max(0, 50 - self.agent.obstacles_hit * 10)
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        active_targets = self.world.get_active_targets()
        
        return {
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'agent_position': (self.agent.x, self.agent.y),
            'agent_speed': self.agent.speed,
            'agent_angle': self.agent.angle,
            'targets_collected': self.agent.targets_collected,
            'targets_remaining': len(active_targets),
            'obstacles_hit': self.agent.obstacles_hit,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            if self.renderer is not None:
                self.renderer.render_frame(
                    self.world,
                    self.agent,
                    60  # FPS
                )
                self.clock.tick(self.metadata["render_fps"])
                
                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        
        elif self.render_mode == "rgb_array":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(self.window_size, pygame.HIDDEN)
                from target_core import GameVisual
                self.renderer = GameVisual(self.window_size)
            
            self.renderer.screen = self.screen
            self.renderer.render_frame(self.world, self.agent, 60)
            
            # Get pixel array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    
    def scripted_policy(self):
        """Simple target-seeking policy for warm-start."""
        active_targets = self.world.get_active_targets()
        
        if not active_targets:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Find nearest target
        nearest_target = min(active_targets,
                           key=lambda t: math.hypot(t.x - self.agent.x, 
                                                   t.y - self.agent.y))
        
        # Calculate angle to target
        angle_to_target = math.atan2(nearest_target.y - self.agent.y,
                                     nearest_target.x - self.agent.x)
        agent_angle_rad = math.radians(self.agent.angle)
        
        # Relative angle
        angle_diff = angle_to_target - agent_angle_rad
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Simple controller
        acceleration = 0.8  # Usually accelerate
        steering = np.clip(angle_diff * 2.0, -1.0, 1.0)
        
        # Slow down if obstacle is close ahead
        obs = self._get_observation()
        forward_sensors = [obs[8], obs[9], obs[10]]  # Front-facing sensors
        min_forward_dist = min(forward_sensors)
        
        if min_forward_dist < 0.3:
            acceleration = -0.5  # Brake
            steering *= 2.0  # Turn harder
        
        return np.array([acceleration, steering], dtype=np.float32)


# Wrapper for observation normalization
class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to zero mean and unit variance."""
    
    def __init__(self, env):
        super().__init__(env)
        self.obs_rms = None
        self.epsilon = 1e-8
    
    def observation(self, obs):
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=obs.shape)
        
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class RunningMeanStd:
    """Running mean and standard deviation tracker."""
    
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else 0
        batch_count = 1
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = TargetEnv(render_mode="human", num_targets=8, num_obstacles=6)
    
    # Test with scripted policy
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(2000):
        action = env.scripted_policy()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished!")
            print(f"  Reward: {info['episode_reward']:.2f}")
            print(f"  Steps: {info['current_step']}")
            print(f"  Targets collected: {info['targets_collected']}/{env.num_targets}")
            print(f"  Obstacles hit: {info['obstacles_hit']}")
            obs, info = env.reset()
    
    env.close()