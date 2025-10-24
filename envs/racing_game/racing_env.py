import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional, Tuple, Dict, Any


class RacingEnv(gym.Env):
    """
    Gymnasium environment for the racing game.
    
    Observation Space:
        - Car speed (normalized)
        - Car angle (sin and cos components)
        - Distance to track edges (8 rays)
        - Distance to next checkpoint (normalized)
        - Angle to next checkpoint (sin and cos)
        - Current checkpoint index (normalized)
        - Previous 3 checkpoint distances (for temporal info)
        
    Action Space:
        - Discrete(5): [No-op, Forward, Backward, Left, Right]
        - Or Continuous: [acceleration, steering] in [-1, 1] (the one I actually use)
        
    Reward:
        - Bonus for speed, penalty for being too slow
        - Bonus for progress
        - Small bonus for being on track, penalty for going off track
        - Big bonus for crossing a checkpoint, big bonus for completing a lap
        - Big bonus for making progress to next checkpoint
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    # Game constants
    MAX_SPEED = 8
    ACCELERATION = 4
    FRICTION = 2
    TURN_SPEED = 400
    TRACK_WIDTH = 100
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        action_type: str = "continuous",
        max_steps: int = 2500,
        window_size: Tuple[int, int] = (1080, 720),
        polygon_points: Optional[list] = None,
        track_width: int = 100,
        track_smoothing: int = 15,
        reward_mode = "sac"
    ):
        """
        Initialize the racing environment.
        
        Args:
            render_mode: "human" for window display, "rgb_array" for pixels
            action_type: "discrete" or "continuous"
            max_steps: Maximum steps per episode
            window_size: Window dimensions
            polygon_points: Custom track polygon (None for default)
            track_width: Width of the racing track
            track_smoothing: Smoothing parameter for track generation
            reward_mode: Unused but needed to keep code consistent across my envs
        """
        super().__init__()

        self.reward_mode = reward_mode
        
        self.render_mode = render_mode
        self.action_type = action_type
        self.max_steps = max_steps
        self.window_size = window_size
        
        # Track configuration
        if polygon_points is None:
            self.polygon_points = [
                (180, 280), (220, 200), (300, 160), (420, 160),
                (520, 200), (580, 280), (620, 360), (700, 400),
                (820, 420), (880, 480), (860, 560), (760, 600),
                (620, 600), (500, 580), (400, 540), (320, 480),
                (260, 420), (200, 360), (180, 320)
            ]
        else:
            self.polygon_points = polygon_points
            
        self.track_width = track_width
        self.track_smoothing = track_smoothing
        
        # Define action space
        if action_type == "discrete":
            # [No-op, Forward, Backward, Left, Right]
            self.action_space = spaces.Discrete(5)
        elif action_type == "continuous":
            # [acceleration, steering] in [-1, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
        
        # Define observation space
        # [speed, sin(angle), cos(angle), 8x distance_to_edge, 
        #  distance_to_checkpoint, sin(angle_to_cp), cos(angle_to_cp),
        #  checkpoint_progress, 3x prev_checkpoint_distances]
        obs_dim = 1 + 2 + 8 + 1 + 2 + 1 + 3 #17? I think?
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Ray casting for distance sensing
        self.num_rays = 8
        self.ray_angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        self.max_ray_distance = 300.0
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_checkpoint_distances = [1.0, 1.0, 1.0]
        
        # Game objects
        self.track = None
        self.car = None
        
        # Pygame/rendering
        self.screen = None
        self.clock = None
        self.renderer = None

        self.off_track_count = 0
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Racing RL Environment")
            self.clock = pygame.time.Clock()
            from racing_core import RacingVisual
            self.renderer = RacingVisual(window_size)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize track and car
        from racing_core import PolygonTrack, HumanPlayer
        
        self.track = PolygonTrack(
            polygon_points=self.polygon_points,
            track_width=self.track_width,
            smoothing=self.track_smoothing,
            window_size=self.window_size
        )
        
        self.car = HumanPlayer()
        self.car.reset()
        self.off_track_count = 0
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_checkpoint_distances = [1.0, 1.0, 1.0]
        self.last_checkpoint_index = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        dt = 1.0 / 60.0  # Fixed timestep
        
        # Apply action
        self._apply_action(action, dt)
        
        # Move car
        self.car.move()
        
        # Check checkpoint crossing
        prev_checkpoint = self.track.next_checkpoint_index
        self.track.check_checkpoint_crossing(self.car)
        checkpoint_crossed = self.track.next_checkpoint_index != prev_checkpoint
        
        # Check if off track
        off_track = self.track.check_off_track(self.car.x, self.car.y)
        
        # Calculate reward
        reward = self._calculate_reward(checkpoint_crossed, off_track)
        
        # Check termination conditions
        self.current_step += 1
        if off_track:
            self.off_track_count += 1
            self.car.reset()
            self.track.reset()
    
        # try 3 screwups before axing the episode
        terminated = (self.off_track_count >= 3 or 
                  self.track.laps_completed >= 1)
        truncated = self.current_step >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['checkpoint_crossed'] = checkpoint_crossed
        info['off_track'] = off_track
        
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray, dt: float):
        """Apply action to control the car."""
        if self.action_type == "discrete":
            # [No-op, Forward, Backward, Left, Right]
            if action == 1:  # Forward
                self.car.speed += self.ACCELERATION * dt
            elif action == 2:  # Backward/Brake
                if self.car.speed > 0:
                    self.car.speed = max(0, self.car.speed - self.ACCELERATION * dt * 2)
                else:
                    self.car.speed -= self.ACCELERATION * dt * 0.5
            elif action == 3:  # Left
                if abs(self.car.speed) > 0.1:
                    turn_factor = self.car.speed / self.MAX_SPEED
                    self.car.angle -= self.TURN_SPEED * dt * turn_factor
            elif action == 4:  # Right
                if abs(self.car.speed) > 0.1:
                    turn_factor = self.car.speed / self.MAX_SPEED
                    self.car.angle += self.TURN_SPEED * dt * turn_factor
            # action == 0 is no-op, handled by friction
            
        elif self.action_type == "continuous":
            acceleration = action[0]
            steering = action[1]
            
            # Apply acceleration
            if acceleration > 0:
                self.car.speed += self.ACCELERATION * dt * acceleration
            elif acceleration < 0:
                if self.car.speed > 0:
                    self.car.speed = max(0, self.car.speed + self.ACCELERATION * dt * acceleration * 2)
                else:
                    self.car.speed += self.ACCELERATION * dt * acceleration * 0.5
            
            # Apply steering (only when moving)
            if abs(self.car.speed) > 0.1:
                turn_factor = self.car.speed / self.MAX_SPEED
                self.car.angle += self.TURN_SPEED * dt * turn_factor * steering
        
        # Apply friction
        if abs(self.car.speed) > 0:
            friction_force = self.FRICTION * dt
            if self.car.speed > 0:
                self.car.speed = max(0, self.car.speed - friction_force)
            else:
                self.car.speed = min(0, self.car.speed + friction_force)
        
        # Clamp speed
        self.car.speed = np.clip(self.car.speed, -self.MAX_SPEED / 2, self.MAX_SPEED)
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        obs = []
        
        # 1. Normalized speed
        obs.append(self.car.speed / self.MAX_SPEED)
        
        # 2. Car angle (sin and cos for continuity)
        angle_rad = math.radians(self.car.angle)
        obs.append(math.sin(angle_rad))
        obs.append(math.cos(angle_rad))
        
        # 3. Distance to track edges (8 rays)
        ray_distances = self._cast_rays()
        obs.extend(ray_distances)
        
        # 4. Distance to next checkpoint (normalized)
        next_checkpoint = self.track.checkpoints[self.track.next_checkpoint_index]
        checkpoint_dist = next_checkpoint.distance_to_point((self.car.x, self.car.y))
        normalized_dist = min(checkpoint_dist / 500.0, 1.0)  # Normalize to [0, 1]
        obs.append(normalized_dist)
        
        # 5. Angle to next checkpoint (sin and cos)
        cp_center = (
            (next_checkpoint.p1[0] + next_checkpoint.p2[0]) / 2,
            (next_checkpoint.p1[1] + next_checkpoint.p2[1]) / 2
        )
        angle_to_cp = math.atan2(cp_center[1] - self.car.y, cp_center[0] - self.car.x)
        relative_angle = angle_to_cp - angle_rad
        obs.append(math.sin(relative_angle))
        obs.append(math.cos(relative_angle))
        
        # 6. Checkpoint progress (normalized)
        progress = self.track.next_checkpoint_index / len(self.track.checkpoints)
        obs.append(progress)
        
        # 7. Previous checkpoint distances (temporal information)
        obs.extend(self.prev_checkpoint_distances)
        
        return np.array(obs, dtype=np.float32)
    
    def _cast_rays(self) -> list:
        """Cast rays from car to detect distance to track edges."""
        distances = []
        car_pos = (self.car.x, self.car.y)
        angle_rad = math.radians(self.car.angle)
        
        for ray_offset in self.ray_angles:
            ray_angle = angle_rad + ray_offset
            ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
            
            # Binary search for intersection distance
            min_dist = 0
            max_dist = self.max_ray_distance
            
            for _ in range(10):  # 10 iterations for decent precision
                mid_dist = (min_dist + max_dist) / 2
                test_point = (
                    car_pos[0] + ray_dir[0] * mid_dist,
                    car_pos[1] + ray_dir[1] * mid_dist
                )
                
                if self.track.check_off_track(test_point[0], test_point[1]):
                    max_dist = mid_dist
                else:
                    min_dist = mid_dist
            
            # Normalize distance
            normalized_dist = min_dist / self.max_ray_distance
            distances.append(normalized_dist)
        
        return distances
    
    def _calculate_reward(self, checkpoint_crossed: bool, off_track: bool) -> float:
        reward = 0.0

        # Strong movement incentive
        speed_ratio = min(abs(self.car.speed) / self.MAX_SPEED, 1.0)
        
        if self.reward_mode == "sac":
            # SAC: Keep working configuration
            reward += speed_ratio * 1.0
        elif self.reward_mode == "ppo":
            # PPO: Slightly reduce to stabilize variance
            if self.car.speed > 0:
                reward += speed_ratio * 0.8
        elif self.reward_mode == "a2c":
            # A2C: Conservative speed reward
            reward += speed_ratio * 0.7
        elif self.reward_mode == "ddpg":
            # DDPG: Strong speed incentive with smooth gradient
            # Quadratic for smoother derivative at low speeds
            reward += (speed_ratio ** 0.8) * 1.2
        else:
            reward += speed_ratio * 1.0
        
        # Penalize being stationary
        if abs(self.car.speed) < 1.0:
            if self.reward_mode == "sac":
                # SAC: Keep your working configuration
                reward -= 2.0
            elif self.reward_mode == "ppo":
                # PPO: Reduce harsh penalty to avoid value spikes
                reward -= 1.2
            elif self.reward_mode == "a2c":
                # A2C: Even gentler penalty
                reward -= 1.0
            elif self.reward_mode == "ddpg":
                # DDPG: Smooth penalty curve
                speed_deficit = (1.0 - abs(self.car.speed))
                reward -= speed_deficit * speed_deficit * 1.8
            else:
                reward -= 2.0
        
        # Penalty for off track / reward for on track
        if off_track:
            if self.reward_mode == "sac":
                # SAC: Keep your working configuration
                reward -= 10.0
            elif self.reward_mode == "ppo":
                # PPO: Reduce penalty - large negative rewards cause instability
                reward -= 6.0
            elif self.reward_mode == "a2c":
                # A2C: Further reduced to minimize advantage variance
                reward -= 5.0
            elif self.reward_mode == "ddpg":
                # DDPG: Strong penalty for deterministic policy
                reward -= 12.0
            else:
                reward -= 10.0
        else:
            if self.reward_mode == "sac":
                # SAC: Keep your working configuration
                reward += 1.0
            elif self.reward_mode == "ppo":
                # PPO: Reduce on-track reward to lower variance
                reward += 0.5
            elif self.reward_mode == "a2c":
                # A2C: Small consistent reward
                reward += 0.4
            elif self.reward_mode == "ddpg":
                # DDPG: Moderate on-track reward
                reward += 0.8
            else:
                reward += 1.0
        
        # Bonus for crossing a checkpoint
        if checkpoint_crossed:
            if self.reward_mode == "sac":
                # SAC: Keep your working configuration
                reward += 100.0
                if self.track.next_checkpoint_index == 0:
                    reward += 500.0
            elif self.reward_mode == "ppo":
                # PPO: Reduce sparse rewards to stabilize advantage estimation
                # Use tanh to bound large rewards
                reward += 60.0
                if self.track.next_checkpoint_index == 0:
                    reward += 250.0
            elif self.reward_mode == "a2c":
                # A2C: More conservative sparse rewards
                reward += 50.0
                if self.track.next_checkpoint_index == 0:
                    reward += 200.0
            elif self.reward_mode == "ddpg":
                # DDPG: Can handle large sparse rewards well
                reward += 120.0
                if self.track.next_checkpoint_index == 0:
                    reward += 600.0
            else:
                reward += 100.0
                if self.track.next_checkpoint_index == 0:
                    reward += 500.0
        
        # Progress (scaled up significantly)
        next_checkpoint = self.track.checkpoints[self.track.next_checkpoint_index]
        current_dist = next_checkpoint.distance_to_point((self.car.x, self.car.y))
        normalized_current = min(current_dist / 500.0, 1.0)
        
        if len(self.prev_checkpoint_distances) > 0:
            prev_dist = self.prev_checkpoint_distances[-1]
            progress_reward = (prev_dist - normalized_current)
            
            if self.reward_mode == "sac":
                # SAC: Keep your working configuration
                reward += progress_reward
            elif self.reward_mode == "ppo":
                # PPO: Clip progress reward to reduce variance from sudden changes
                # This is often the main source of instability in racing environments
                clipped_progress = np.clip(progress_reward, -0.5, 0.5)
                reward += clipped_progress * 0.8
            elif self.reward_mode == "a2c":
                # A2C: More aggressive clipping and scaling
                clipped_progress = np.clip(progress_reward, -0.4, 0.4)
                reward += clipped_progress * 0.7
            elif self.reward_mode == "ddpg":
                # DDPG: Can use raw progress but scale up for stronger signal
                reward += progress_reward * 1.3
            else:
                reward += progress_reward
        
        self.prev_checkpoint_distances.pop(0)
        self.prev_checkpoint_distances.append(normalized_current)
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'car_position': (self.car.x, self.car.y),
            'car_speed': self.car.speed,
            'car_angle': self.car.angle,
            'laps_completed': self.track.laps_completed,
            'next_checkpoint': self.track.next_checkpoint_index,
            'checkpoint_progress': self.track.next_checkpoint_index / len(self.track.checkpoints),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            if self.renderer is not None:
                self.renderer.render_frame(
                    self.track,
                    self.car,
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
                from racing_core import RacingVisual
                self.renderer = RacingVisual(self.window_size)
            
            self.renderer.screen = self.screen
            self.renderer.render_frame(self.track, self.car, 60)
            
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
        """Simple forward-driving policy for jumpstarting useful progress into the SAC warmup."""
        if self.action_type == "discrete":
            return 1
        obs = self._get_observation()
        
        # Extract angle to checkpoint
        angle_to_cp_sin = obs[11]
        angle_to_cp_cos = obs[12]
        angle_to_cp = np.arctan2(angle_to_cp_sin, angle_to_cp_cos)
        
        # Always accelerate, steer toward checkpoint
        import random
        acceleration = 1 * random.randint(4, 10) / 10
        steering = np.clip(angle_to_cp * 2.0, -1.0, 1.0) * random.randint(-5, 10) / 10
        
        return np.array([acceleration, steering], dtype=np.float32)


# Wrapper for additional features
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
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
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
    env = RacingEnv(render_mode="human", action_type="continuous")
    
    # Test random policy
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(1000):
        action = env.scripted_policy()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished: Reward={info['episode_reward']:.2f}, Steps={info['current_step']}")
            obs, info = env.reset()
    
    env.close()