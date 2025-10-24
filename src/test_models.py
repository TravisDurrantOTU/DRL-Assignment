import torch
import gymnasium as gym
import models_core as mc
import numpy as np
import sys
import os
from abc import abstractmethod, ABC

# TODO: FINISH THIS SHIT
# TODO: Fuck this really should be an inheritance situation goddammit
# TODO: Do these actually need to be different? I could just change the dict keys for how it's saved maybe
# TODO: FIgure this bullshit out

class ModelTester(ABC):
    def __init__(self, filename : str, environment : gym.Env):
        """Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """
        if filename is None or environment is None:
            raise ValueError("Filename and environment must be provided.")

        self.filename = filename
        self.env = environment
        self.hidden_sizes = None

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, n_episodes : int = 10, visual : bool = False):
        """Function for testing a loaded model.
        
        Args:
            - n_episodes : The number of episodes to train on.
            - visual : Whether or not to render the environment. Not reccommended for n_episodes > 1.
            
        Returns:
            tuple : episode rewards, episode_info
                - episode_rewards : List of the rewards per episode.
                - episode_info : Disgusting list of info dicts.

        """        
        obs, info = self.env.reset()

        # Data analysis about to be MISERABLE
        episode_rewards = []
        episode_infos = [[] for _ in range(n_episodes)]
        terminal_infos = []

        for i in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if visual:
                    self.env.render()

                action = self._get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

                episode_infos[i].append(info)

            print(f"Episode {i+1} finished: total reward = {total_reward:.2f}")
            episode_rewards.append(total_reward)
            terminal_infos.append(info)


        # Unfortunately the info dict is going to depend on the env so can't be standardized
        return episode_rewards, episode_infos, terminal_infos


    @abstractmethod
    def _load(self, filename : str):
        pass

    @abstractmethod
    def _get_action(self, state):
        pass
        
class DDPGTester(ModelTester):
    def __init__(self, filename: str, environment: gym.Env):
        """Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """
        super().__init__(filename, environment)

        # DDPG specific init
        self.actor = None
        self.hidden_sizes = [256, 256]
        
        # Action space bounds for scaling
        self.action_high = torch.FloatTensor(self.env.action_space.high).to(self.device)
        self.action_low = torch.FloatTensor(self.env.action_space.low).to(self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self._load(filename)

    def _load(self, filename):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        self.checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.actor = mc.DDPGActor(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
        self.actor.eval()
        
    def _get_action(self, state):
        """Get deterministic action from the actor (no noise during testing)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        action = action.squeeze(0).cpu().numpy()
        
        # Scale action to environment bounds
        action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        action = np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        return action

    def test(self, n_episodes: int = 10, visual: bool = False):
        return super().test(n_episodes=n_episodes, visual=visual)


class SACTester(ModelTester):
    def __init__(self, filename : str, environment: gym.Env):
        """Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """
        super().__init__(filename, environment)

        # TODO: SAC specific init
        self.actor = None
        self.hidden_sizes = [256, 256]

        self._load(filename)

    def _load(self, filename):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        self.checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.actor = mc.Actor(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
        self.actor.eval()
        
    def _get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(state_tensor)
        action = action.squeeze(0).detach().cpu().numpy()
        return action

    def test(self, n_episodes : int = 10, visual : bool = False):
        return super().test(n_episodes=n_episodes, visual=visual)

class PPOTester(ModelTester):
    def __init__(self, filename : str, environment: gym.Env):
        """Class for testing a trained PPO-based model.
        
        Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """        
        super().__init__(filename, environment)

        # TODO: PPO specific init
        self.network = None
        self.hidden_sizes = [128, 128]

        self._load(filename)

    def _load(self, filename: str):
        """Load PPO model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.network = mc.ActorCriticNetwork(self.obs_dim, self.act_dim, True, self.hidden_sizes).to(self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])

    def _get_action(self, state):
        """Function to get action from PPO-trained network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.network.get_action(state_tensor)
        action = action.squeeze(0).detach().cpu().numpy()
        return action

    def test(self, n_episodes : int = 10, visual : bool = False):
        return super().test(n_episodes=n_episodes, visual=visual)
        
racing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "racing_game"))
target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "target_game"))
sys.path.append(racing_dir)
from racing_env import RacingEnv
sys.path.append(target_dir)
from target_env import TargetEnv

if __name__ == "__main__":
    renv = RacingEnv(render_mode = 'human')
    s = SACTester('race_sac_base.pt', renv)
    p = PPOTester('race_ppo_base.pt', renv)
    #print(s.test(n_episodes=1, visual=True))
    #print(p.test(n_episodes=1, visual=True))

    tenv = TargetEnv(render_mode = 'human')
    s = SACTester('target_sac_seed10.pt', tenv)
    p = PPOTester('target_ppo_seed10.pt', tenv)
    s.test(n_episodes=1, visual=True)
    p.test(n_episodes=1, visual=True)