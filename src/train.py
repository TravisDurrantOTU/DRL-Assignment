import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import json
import os


class ActorCriticNetwork(nn.Module):
    """Neural network for both policy (actor) and value function (critic)."""
    
    def __init__(self, obs_dim: int, act_dim: int, continuous: bool = False, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        self.continuous = continuous
        
        # Shared feature extraction layers
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        if continuous:
            self.actor_mean = nn.Linear(prev_size, act_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        else:
            self.actor = nn.Linear(prev_size, act_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple:
        features = self.shared(x)
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd.clamp(-20, 2))
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            if self.continuous:
                mean, std, value = self.forward(x)
                if deterministic:
                    return mean, None, value
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                return action, log_prob, value
            else:
                logits, value = self.forward(x)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                return action, log_prob, value


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
        device: str = "auto"
    ):
        # Create environment
        self.env = gym.make(env_name)
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
        self.network = ActorCriticNetwork(
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
        print(f"Training on {self.env_name}")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}")
        print(f"Continuous: {self.continuous}")
        print("-" * 50)
        
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
                print(f"Timesteps: {timesteps}/{total_timesteps} | "
                      f"Episodes: {len(self.episode_rewards)} | "
                      f"Mean Reward (last 100): {mean_reward:.2f}")
            
            # Evaluation
            if timesteps % eval_freq < steps_per_rollout:
                eval_reward = self.evaluate(n_episodes=5)
                eval_rewards.append((timesteps, eval_reward))
                print(f"Evaluation Reward: {eval_reward:.2f}")
        
        # Save model
        if save_path:
            self.save(save_path)
            print(f"Model saved to {save_path}")
        
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
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'env_name': self.env_name,
            'continuous': self.continuous,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
    
    def plot_training(self, save_path: Optional[str] = None):
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
        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # Example usage for different environments
    
    # Discrete action space example (CartPole)
    print("Training on CartPole-v1")
    trainer = PPOTrainer(
        env_name="CartPole-v1",
        hidden_sizes=[128, 128],
        lr=3e-4,
        batch_size=64,
    )
    trainer.train(total_timesteps=100000, steps_per_rollout=2048)
    trainer.plot_training(save_path="cartpole_training.png")
    trainer.save("cartpole_model.pt")
    
    # Evaluate
    print("\nEvaluating trained agent...")
    eval_reward = trainer.evaluate(n_episodes=10)
    print(f"Average evaluation reward: {eval_reward:.2f}")
    
    # Continuous action space example (uncomment to try)
    # print("\n" + "="*50)
    # print("Training on Pendulum-v1")
    # trainer_continuous = PPOTrainer(
    #     env_name="Pendulum-v1",
    #     hidden_sizes=[256, 256],
    #     lr=3e-4,
    # )
    # trainer_continuous.train(total_timesteps=200000, steps_per_rollout=2048)
    # trainer_continuous.plot_training(save_path="pendulum_training.png")