import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
import sys
import models_core as mc
import multilogger as ml

log = ml.MultiLogger()
log.add_output("console", sys.stdout, timestamps=True)
log.log("MultiLogger Initialized")

class PPOTrainer:
    """PPO trainer for any Gymnasium environment."""
    
    def __init__(
        self,
        env_name: str,
        hidden_sizes: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "auto",
        reward_mode: str = "sac"
    ):
        # Create environment
        self.env = gym.make(env_name)
        self.env.unwrapped.reward_mode = reward_mode
        self.env_name = env_name
        
        # Determine observation and action space dimensions
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError("Only Box observation spaces supported")
        
        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)
        if self.continuous:
            self.act_dim = self.env.action_space.shape[0]
        else:
            self.act_dim = self.env.action_space.n
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.network = mc.ActorCriticNetwork(
            self.obs_dim, self.act_dim, self.continuous, hidden_sizes
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ppo_debug.txt"))
        log.add_output("ppo_debug", log_path, timestamps=True)

        
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        return advantages, returns

    def collect_rollout(self, n_steps: int) -> dict:
        """Collect experience from environment."""
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        state, _ = self.env.reset()
        ep_reward = 0
        ep_length = 0
        
        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(state_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            if not self.continuous:
                action_np = int(action_np)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action.cpu().numpy().squeeze())
            log_probs.append(log_prob.cpu().item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().item())
            
            ep_reward += reward
            ep_length += 1
            
            if done:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                state, _ = self.env.reset()
                ep_reward = 0
                ep_length = 0
            else:
                state = next_state
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        log.log({
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'advantages': advantages,
            'returns': returns,
        }, 'ppo_debug')

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'advantages': advantages,
            'returns': returns,
        }
    
    def update(self, rollout: dict):
        """Update policy using PPO."""
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.FloatTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout['advantages']).to(self.device)
        returns = torch.FloatTensor(rollout['returns']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        dataset_size = states.shape[0]
        for _ in range(self.n_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                if self.continuous:
                    mean, std, values = self.network(batch_states)
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(batch_actions).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits, values = self.network(batch_states)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(batch_actions.long())
                    entropy = dist.entropy().mean()

                log.log(f"Entropy: {entropy}", "ppo_debug")
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.losses.append(loss.item())
    
    def train(self, total_timesteps: int, steps_per_rollout: int = 2048, eval_freq: int = 10000, save_path: Optional[str] = None):
        """Train the agent."""
        log.log(f"Training on {self.env_name}", ["ppo_debug", "console"])
        log.log(f"Device: {self.device}", ["ppo_debug", "console"])
        log.log(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}", ["ppo_debug", "console"])
        log.log(f"Continuous: {self.continuous}", ["ppo_debug", "console"])
        log.log("-" * 50, ["ppo_debug", "console"])
        
        timesteps = 0
        eval_rewards = []

        while timesteps < total_timesteps:

            # Collect rollout
            rollout = self.collect_rollout(steps_per_rollout)
            timesteps += steps_per_rollout
            
            # Update policy
            self.update(rollout)
            
            # Logging
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)
                log.log(f"Timesteps: {timesteps}/{total_timesteps} | "
                      f"Episodes: {len(self.episode_rewards)} | "
                      f"Recent Reward: {self.episode_rewards[-1]:.2f} | "
                      f"Mean Reward (last 100): {mean_reward:.2f}", ["ppo_debug", "console"])
            
            # Evaluation
            if timesteps % eval_freq < steps_per_rollout:
                eval_reward = self.evaluate(n_episodes=5)
                eval_rewards.append((timesteps, eval_reward))
                log.log(f"Evaluation Reward: {eval_reward:.2f}", ["ppo_debug, console"])
        
        # Save model
        if save_path:
            self.save(save_path)
            log.log(f"Model saved to {save_path}", "ppo_debug")
        
        return eval_rewards
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> float:
        """Evaluate the agent."""
        eval_env = gym.make(self.env_name, render_mode="human" if render else None)
        total_reward = 0
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                if render:
                    eval_env.render()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _, _ = self.network.get_action(state_tensor, deterministic=True)
                action_np = action.cpu().numpy().squeeze()
                
                if not self.continuous:
                    action_np = int(action_np)
                
                state, reward, terminated, truncated, _ = eval_env.step(action_np)
                done = terminated or truncated
                ep_reward += reward
            
            total_reward += ep_reward
        
        eval_env.close()
        return total_reward / n_episodes
    
    def save(self, path: str):
        """Save model and training info."""
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", path))
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'env_name': self.env_name,
            'continuous': self.continuous,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, save_path)
    
    def load(self, path: str):
        """Load model."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", path))
        checkpoint = torch.load(load_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
    
    def plot_training(self, path: Optional[str] = None, display = False):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Episode rewards
        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            window = min(100, len(self.episode_rewards))
            if len(self.episode_rewards) >= window:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(self.episode_rewards)), moving_avg, label=f'Moving Avg ({window})')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Losses
        if len(self.losses) > 0:
            axes[1].plot(self.losses, alpha=0.3)
            window = min(100, len(self.losses))
            if len(self.losses) >= window:
                moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(self.losses)), moving_avg)
            axes[1].set_xlabel('Update Step')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if path:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", path))
            plt.savefig(save_path)
        if display:
            plt.show()

class SACTrainer:
    """SAC trainer for continuous control Gymnasium environments."""
    def __init__(
        self,
        env_name: str,
        hidden_sizes: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: str = "auto",
        reward_mode: str = "sac"
    ):
        # Create environment
        self.env = gym.make(env_name)
        self.env.unwrapped.reward_mode = reward_mode
        self.env_name = env_name
        
        # Determine observation and action space dimensions
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError("Only Box observation spaces supported")
        
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("SAC only supports continuous action spaces")
        
        self.act_dim = self.env.action_space.shape[0]
        self.action_scale = torch.FloatTensor((self.env.action_space.high - self.env.action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((self.env.action_space.high + self.env.action_space.low) / 2.0)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)
        
        # Initialize networks
        self.actor = mc.Actor(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic1 = mc.Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic2 = mc.Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic1_target = mc.Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic2_target = mc.Critic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_entropy = auto_entropy
        if auto_entropy:
            # >< trying to get more exploration
            self.target_entropy = -0.1 * self.act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = mc.ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []

        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "sac_debug.txt"))
        log.add_output("sac_debug", log_path, timestamps=True)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        
        action = action.cpu().numpy().squeeze()
        log.log(f"prescaled action: {action}", "sac_debug")
        action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        log.log(f"postscaled action: {action}", "sac_debug")
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    
    def update(self):
        """Update networks using SAC."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        # Normalize actions back to [-1, 1]
        actions = (actions - self.action_bias) / self.action_scale
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            if isinstance(self.alpha, torch.Tensor):
                alpha_value = self.alpha.detach()
            else:
                alpha_value = self.alpha
            
            target_q = target_q - alpha_value * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        
        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        if isinstance(self.alpha, torch.Tensor):
            alpha_value = self.alpha.detach()
        else:
            alpha_value = self.alpha
        
        actor_loss = (alpha_value * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        
        # Update temperature (alpha)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            self.alpha_losses.append(alpha_loss.item())
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # For debugging purposes
        log.log(f"Mean Q1: {current_q1.mean():.2f}, Q2: {current_q2.mean():.2f}", "sac_debug")
        log.log(f"Actor loss: {actor_loss.item():.3f}, Critic loss: {critic1_loss.item():.3f}", "sac_debug")
        if self.auto_entropy:
            log.log(f"Alpha: {self.alpha.item():.4f}", "sac_debug")
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, update_freq: int = 1, gradient_steps: int = 1, save_path: Optional[str] = None):
        """Train the agent."""
        log.log(f"Training on {self.env_name}", "console")
        log.log(f"Device: {self.device}", "console")
        log.log(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}", "console")
        log.log(f"Auto entropy tuning: {self.auto_entropy}", "console")
        log.log("-" * 50, "console")
        
        state, _ = self.env.reset()
        ep_reward = 0
        ep_length = 0
        eval_rewards = []
        
        for timestep in range(1, total_timesteps + 1):
            if timestep % 500 == 0:
                log.log(f"Timestep {timestep}", "console")

            # Select action
            if timestep < self.warmup_steps:
                action = self.env.unwrapped.scripted_policy()
            else:
                action = self.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            ep_reward += reward
            ep_length += 1
            
            # Update networks
            if timestep >= self.warmup_steps and timestep % update_freq == 0:
                for _ in range(gradient_steps):
                    self.update()
                    log.log(f"Action mean: {action.mean():.3f}, std: {action.std():.3f}", "sac_debug")
            
            if done:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                state, _ = self.env.reset()
                ep_reward = 0
                ep_length = 0
                
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    mean_reward = np.mean(recent_rewards)
                    log.log(f"Timesteps: {timestep}/{total_timesteps} | "
                          f"Episodes: {len(self.episode_rewards)} | "
                          f"Mean Reward (last 100): {mean_reward:.2f}", "console")
            else:
                state = next_state
            
            # Evaluation
            if timestep % eval_freq == 0 and timestep > self.warmup_steps:
                eval_reward = self.evaluate(n_episodes=5)
                eval_rewards.append((timestep, eval_reward))
                log.log(f"Evaluation Reward: {eval_reward:.2f}", "console")

        
        # Save model
        if save_path:
            self.save(save_path)
            log.log(f"Model saved to {save_path}", "console")
        
        return eval_rewards
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> float:
        """Evaluate the agent."""
        eval_env = gym.make(self.env_name, render_mode="human" if render else None)
        total_reward = 0
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                if render:
                    eval_env.render()
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
            
            total_reward += ep_reward
        
        eval_env.close()
        return total_reward / n_episodes
    
    def save(self, filename: str):
        """Save model and training info to the models folder."""
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'env_name': self.env_name,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, save_path)
    
    def load(self, filename: str):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
    
    def plot_training(self, path: Optional[str] = None, display = False):
        """Plot training progress."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Episode rewards
        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            window = min(100, len(self.episode_rewards))
            if len(self.episode_rewards) >= window:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(self.episode_rewards)), moving_avg, label=f'Moving Avg ({window})')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Actor losses
        if len(self.actor_losses) > 0:
            axes[1].plot(self.actor_losses, alpha=0.3)
            window = min(100, len(self.actor_losses))
            if len(self.actor_losses) >= window:
                moving_avg = np.convolve(self.actor_losses, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(self.actor_losses)), moving_avg)
            axes[1].set_xlabel('Update Step')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Actor Loss')
            axes[1].grid(True, alpha=0.3)
        
        # Critic losses
        if len(self.critic_losses) > 0:
            axes[2].plot(self.critic_losses, alpha=0.3)
            window = min(100, len(self.critic_losses))
            if len(self.critic_losses) >= window:
                moving_avg = np.convolve(self.critic_losses, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window-1, len(self.critic_losses)), moving_avg)
            axes[2].set_xlabel('Update Step')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Critic Loss')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if path:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", path))
            plt.savefig(save_path)
        if display:
            plt.show()

# Was gonna do DQN cause I have a predilection for that one, but it can't do continuous action spaces. This is the most similar one.
# This is very unstable though so we'll see if it actually works much at all.
class DDPGTrainer:
    """DDPG trainer for continuous control Gymnasium environments."""
    def __init__(
        self,
        env_name: str,
        hidden_sizes: List[int] = [256, 256],
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: str = "auto",
        reward_mode: str = "sac"
    ):
        # Create environment
        self.env = gym.make(env_name)
        self.env.unwrapped.reward_mode = reward_mode
        self.env_name = env_name
        
        # Determine observation and action space dimensions
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.obs_dim = self.env.observation_space.shape[0]
        else:
            raise ValueError("Only Box observation spaces supported")
        
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise ValueError("DDPG only supports continuous action spaces")
        
        self.act_dim = self.env.action_space.shape[0]
        self.action_high = torch.FloatTensor(self.env.action_space.high)
        self.action_low = torch.FloatTensor(self.env.action_space.low)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.action_high = self.action_high.to(self.device)
        self.action_low = self.action_low.to(self.device)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)
        
        # Initialize networks
        self.actor = mc.DDPGActor(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.actor_target = mc.DDPGActor(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic = mc.DDPGCritic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic_target = mc.DDPGCritic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = mc.ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ddpg_debug.txt"))
        log.add_output("ddpg_debug", log_path, timestamps=True)
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action from policy with optional exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        action = action.cpu().numpy().squeeze()
        
        # Add Gaussian noise for exploration
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
        
        # Scale and clip to action space
        action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        action = np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        log.log(f"Selected action: {action}", "ddpg_debug")
        return action
    
    def update(self):
        """Update networks using DDPG."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        # Normalize actions back to [-1, 1] range
        actions = (actions - self.action_bias) / self.action_scale
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        
        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        
        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Logging
        log.log(f"Mean Q: {current_q.mean():.2f}", "ddpg_debug")
        log.log(f"Actor loss: {actor_loss.item():.3f}, Critic loss: {critic_loss.item():.3f}", "ddpg_debug")
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, update_freq: int = 1, 
              gradient_steps: int = 1, save_path: Optional[str] = None):
        """Train the agent."""
        log.log(f"Training on {self.env_name}", "console")
        log.log(f"Device: {self.device}", "console")
        log.log(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}", "console")
        log.log("-" * 50, "console")
        
        state, _ = self.env.reset()
        ep_reward = 0
        ep_length = 0
        eval_rewards = []
        
        for timestep in range(1, total_timesteps + 1):
            if timestep % 500 == 0:
                log.log(f"Timestep {timestep}", "console")
            
            # Select action
            if timestep < self.warmup_steps:
                # Random actions during warmup
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            ep_reward += reward
            ep_length += 1
            
            # Update networks
            if timestep >= self.warmup_steps and timestep % update_freq == 0:
                for _ in range(gradient_steps):
                    self.update()
            
            if done:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                state, _ = self.env.reset()
                ep_reward = 0
                ep_length = 0
                
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    mean_reward = np.mean(recent_rewards)
                    log.log(f"Timesteps: {timestep}/{total_timesteps} | "
                          f"Episodes: {len(self.episode_rewards)} | "
                          f"Mean Reward (last 100): {mean_reward:.2f}", "console")
            else:
                state = next_state
            
            # Evaluation
            if timestep % eval_freq == 0 and timestep > self.warmup_steps:
                eval_reward = self.evaluate(n_episodes=5)
                eval_rewards.append((timestep, eval_reward))
                log.log(f"Evaluation Reward: {eval_reward:.2f}", "console")
        
        # Save model
        if save_path:
            self.save(save_path)
            log.log(f"Model saved to {save_path}", "console")
        
        return eval_rewards
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> float:
        """Evaluate the agent."""
        eval_env = gym.make(self.env_name, render_mode="human" if render else None)
        total_reward = 0
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                if render:
                    eval_env.render()
                action = self.select_action(state, add_noise=False)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
            
            total_reward += ep_reward
        
        eval_env.close()
        return total_reward / n_episodes
    
    def save(self, filename: str):
        """Save model and training info to the models folder."""
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'env_name': self.env_name,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, save_path)
    
    def load(self, filename: str):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
    
    def plot_training(self, path: Optional[str] = None, display: bool = False):
        """Plot training progress."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Episode rewards
        if len(self.episode_rewards) > 0:
            axes[0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            window = min(100, len(self.episode_rewards))
            if len(self.episode_rewards) >= window:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(self.episode_rewards)), moving_avg, label=f'Moving Avg ({window})')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Actor losses
        if len(self.actor_losses) > 0:
            axes[1].plot(self.actor_losses, alpha=0.3)
            window = min(100, len(self.actor_losses))
            if len(self.actor_losses) >= window:
                moving_avg = np.convolve(self.actor_losses, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(self.actor_losses)), moving_avg)
            axes[1].set_xlabel('Update Step')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Actor Loss')
            axes[1].grid(True, alpha=0.3)
        
        # Critic losses
        if len(self.critic_losses) > 0:
            axes[2].plot(self.critic_losses, alpha=0.3)
            window = min(100, len(self.critic_losses))
            if len(self.critic_losses) >= window:
                moving_avg = np.convolve(self.critic_losses, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window-1, len(self.critic_losses)), moving_avg)
            axes[2].set_xlabel('Update Step')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Critic Loss')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if path:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", path))
            plt.savefig(save_path)
        if display:
            plt.show()

racing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "racing_game"))
target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "target_game"))
sys.path.append(racing_dir)
from racing_env import RacingEnv
sys.path.append(target_dir)
from target_env import TargetEnv

gym.register("RacingEnv", RacingEnv)
gym.register("TargetEnv", TargetEnv)

if __name__ == "__main__":

    trainer = PPOTrainer(
        env_name="RacingEnv",
        hidden_sizes=[128, 128],
        lr=3e-4,
        batch_size=64
    )

    trainer2 = SACTrainer(
        env_name="RacingEnv",
        hidden_sizes=[256, 256],
        lr=3e-4,
        batch_size=256,
        warmup_steps=5000
    )

    trainer3 = DDPGTrainer(
        env_name="RacingEnv",
        hidden_sizes=[256, 256],
        actor_lr=1e-4,
        critic_lr=1e-3,
        warmup_steps=5000
    )

    trainer4 = PPOTrainer(
        env_name="TargetEnv",
        hidden_sizes=[128, 128],
        lr=3e-4,
        batch_size=64,
        reward_mode="ppo"
    )

    trainer5 = SACTrainer(
        env_name="TargetEnv",
        hidden_sizes=[256, 256],
        lr=3e-4,
        batch_size=256,
        warmup_steps=5000,
        reward_mode="sac"
    )

    trainer6 = DDPGTrainer(
        env_name="TargetEnv",
        hidden_sizes=[256, 256],
        actor_lr=1e-4,
        critic_lr=1e-3,
        warmup_steps=5000,
        reward_mode="sac" #haven't implemented the DDPG reward stuff yet
    )

    # REMINDER TO ENSURE STEPS PER ROLLOUT IS GREATER THAN MAX EPISODE (ideally a multiple)
    # Racing Game - 2500
    # Dungeon Game - garbage and scrapping it screw getting that to work
    # Target Game - 2000
    # I picked these basically at random tbh
    #trainer.train(total_timesteps=1000000, steps_per_rollout=2500)
    #trainer2.train(total_timesteps=50000, eval_freq=1000, update_freq=50, gradient_steps=50)

    # 
    trainer4.train(total_timesteps=1000000, steps_per_rollout=2000)
    trainer5.train(total_timesteps=50000, eval_freq=1000, update_freq=50, gradient_steps=50)

    # Save them all
    #trainer.save("race_ppo_cont1.pt")
    #trainer2.save("race_sac_cont1.pt")
    trainer4.save("target_ppo_seed10.pt")
    trainer5.save("target_sac_seed10.pt")

    # Abusing the fact that matlab holds plots so that I can save it to look at in morning
    #trainer.plot_training(path="race_ppo_cont1.png", display=True)
    #log.log("\nEvaluating trained agent...", "console")
    #eval_reward = trainer.evaluate(n_episodes=1, render=True)
    #log.log(f"Average evaluation reward: {eval_reward:.2f}", "console")

    #trainer2.plot_training(path="race_sac_cont1.png", display=True)
    #log.log("\nEvaluating trained agent...", "console")
    #eval_reward = trainer2.evaluate(n_episodes=1, render=True)
    #log.log(f"Average evaluation reward: {eval_reward:.2f}", "console")

    trainer4.plot_training(path="target_ppo_seed10.png", display=True)
    log.log("\nEvaluating trained agent...", "console")
    eval_reward = trainer4.evaluate(n_episodes=1, render=True)
    log.log(f"Average evaluation reward: {eval_reward:.2f}", "console")

    trainer5.plot_training(path="target_sac_seed10.png", display=True)
    log.log("\nEvaluating trained agent...", "console")
    eval_reward = trainer5.evaluate(n_episodes=1, render=True)
    log.log(f"Average evaluation reward: {eval_reward:.2f}", "console")

log.close()