import torch
import gymnasium as gym
import models_core as mc
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

        if self.network is None:
            raise ValueError("Model must be intiialized to test it.")
        
        obs, info = self.env.reset()

        # Data analysis about to be MISERABLE
        episode_rewards = []
        episode_infos = [[]*n_episodes]

        for i in range(n_episodes):            
            action = self.env._get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if visual:
                self.env.render()

            # TODO: Append the data to something
            episode_rewards.append(reward)
            episode_infos[i].append(info) # this is gonne be fucking digusting but whatever


            if terminated or truncated:
                print(f"Episode finished: Reward={info['episode_reward']:.2f}, Steps={info['current_step']}")
                obs, info = self.env.reset()

        # Unfortunately the info dict is going to depend on the env so can't be standardized
        return episode_rewards, episode_infos


    @abstractmethod
    def _load(self, filename : str):
        pass

    @abstractmethod
    def _get_action(self, state):
        pass
        
    @abstractmethod
    def load(self, filename: str):
        pass

class SACTester(ModelTester):
    def __init__(self, filename : str, environment: gym.Env):
        """Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """
        super.__init__(self, filename, environment)

        # TODO: SAC specific init
        self.actor = None

        self._load(filename)

    def _load(self, filename):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        self.checkpoint = torch.load(load_path, map_location=self.device)
        self.hidden_sizes = self.checkpoint['hidden_sizes']

        self.actor = mc.Actor(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        #self.critic1 = mc.Critic(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        #self.critic2 = mc.Critic(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        #self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        #self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        #self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        #self.episode_rewards = checkpoint['episode_rewards']
        
    def _get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(state_tensor)
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
        super.__init__(self, filename, environment)

        # TODO: PPO specific init
        self.network = None

        self._load(filename)

    def _load(self, filename: str):
        """Load PPO model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        checkpoint = torch.load(load_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.episode_rewards = checkpoint['episode_rewards']

    def _get_action(self, state):
        """Function to get action from PPO-trained network"""
        action, _, _ = self.network.get_action(state)
        
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
    print("write actual code in here")