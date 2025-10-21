import torch
import torch.nn as nn
from typing import List, Tuple
from torch.distributions import Categorical, Normal
from collections import deque
import random
import numpy as np

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

class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Gaussian policy network."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_size, act_dim)
        self.log_std_head = nn.Linear(prev_size, act_dim)
        
    
    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Reparameterization trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability with correction for tanh squashing
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean

class Critic(nn.Module):
    """Q-function network."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim + act_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)