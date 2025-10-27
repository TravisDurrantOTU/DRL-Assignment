import torch
import gymnasium as gym
import models_core as mc
import numpy as np
import sys
import os
from abc import abstractmethod, ABC
import multilogger as ml
import time
import json

log = ml.MultiLogger()
log.add_output("console", sys.stdout, timestamps=True)
log.log("MultiLogger Initialized")

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
        
        # Setup testing log
        self.testlogname = f"testing-{self.__class__.__name__}-{time.time()}"
        self.testjsonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", f"testing-{self.__class__.__name__}-{time.time()}.json"))
        testlogpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", f"testing-{self.__class__.__name__}-{time.time()}.txt"))
        log.add_output(self.testlogname, testlogpath, timestamps=False)

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

        # Log test start
        log.log("=" * 70, "console")
        log.log(f"Testing Model: {self.filename}", "console")
        env_name = getattr(getattr(self.env, "spec", None), "id", "Unknown")
        log.log(f"Environment: {env_name}", "console")
        log.log(f"Number of Episodes: {n_episodes}", "console")
        log.log(f"Device: {self.device}", "console")
        log.log("=" * 70, "console")
        
        obs, info = self.env.reset()

        # Data analysis about to be MISERABLE
        episode_rewards = []
        episode_lengths = []
        episode_infos = [[] for _ in range(n_episodes)]
        terminal_infos = []

        for i in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                if visual:
                    self.env.render()

                action = self._get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                episode_infos[i].append(info)

            # Log episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            terminal_infos.append(info)
            
            episode_log = (
                f"Episode {i+1}/{n_episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Length: {steps} | "
            )
            
            # log the info dict
            for key in info.keys():
                episode_log += f"{key}: {info[key]}\n"
            log.log(episode_log.rstrip(" | "), [self.testlogname, "console"])

        # Log summary statistics
        log.log("-" * 70, self.testlogname)
        log.log("TESTING SUMMARY", [self.testlogname, "console"])
        log.log("-" * 70, self.testlogname)
        
        summary_stats = (
            f"Total Episodes: {n_episodes} | "
            f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f} | "
            f"Min Reward: {np.min(episode_rewards):.2f} | "
            f"Max Reward: {np.max(episode_rewards):.2f} | "
            f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
        )
        
        log.log(summary_stats, [self.testlogname, "console"])
        
        # Log success rate if available
        if terminal_infos and 'success' in terminal_infos[0]:
            success_count = sum(1 for info in terminal_infos if info.get('success', False))
            success_rate = (success_count / n_episodes) * 100
            log.log(f"Success Rate: {success_rate:.1f}% ({success_count}/{n_episodes})", [self.testlogname, "console"])
        
        # Log collision rate if available
        if terminal_infos and 'collision' in terminal_infos[0]:
            collision_count = sum(1 for info in terminal_infos if info.get('collision', False))
            collision_rate = (collision_count / n_episodes) * 100
            log.log(f"Collision Rate: {collision_rate:.1f}% ({collision_count}/{n_episodes})", [self.testlogname, "console"])
        
        log.log("=" * 70, self.testlogname)

        with open(self.testjsonpath, 'w') as f:
            json.dump(episode_infos, f, indent=2)

        # Unfortunately the info dict is going to depend on the env so can't be standardized
        return episode_rewards, episode_infos, terminal_infos


    @abstractmethod
    def _load(self, filename : str):
        pass

    @abstractmethod
    def _get_action(self, state):
        pass

class A2CTester(ModelTester):
    def __init__(self, filename: str, environment: gym.Env):
        """Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """
        super().__init__(filename, environment)

        # A2C specific init
        self.network = None
        self.hidden_sizes = [256, 256]
        
        # Determine if continuous or discrete action space
        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)
        if not self.continuous:
            self.act_dim = self.env.action_space.n

        log.log(f"A2C Tester initialized - Continuous: {self.continuous}, Hidden sizes: {self.hidden_sizes}", self.testlogname)
        self._load(filename)

    def _load(self, filename):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        log.log(f"Loading model from: {load_path}", self.testlogname)
        self.checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.network = mc.ActorCriticNetwork(
            self.obs_dim, self.act_dim, self.continuous, self.hidden_sizes
        ).to(self.device)
        self.network.load_state_dict(self.checkpoint['network_state_dict'])
        self.network.eval()
        log.log("Model loaded successfully", [self.testlogname, "console"])
        
    def _get_action(self, state):
        """Get action from the actor (deterministic for testing)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(state_tensor, deterministic=True)
        
        action = action.squeeze(0).cpu().numpy()
        
        if not self.continuous:
            action = int(action)
        
        return action

    def test(self, n_episodes: int = 10, visual: bool = False):
        return super().test(n_episodes=n_episodes, visual=visual)

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

        log.log(f"DDPG Tester initialized - Hidden sizes: {self.hidden_sizes}", self.testlogname)
        log.log(f"Action bounds - Low: {self.action_low.cpu().numpy()}, High: {self.action_high.cpu().numpy()}", self.testlogname)
        self._load(filename)

    def _load(self, filename):
        """Load model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        log.log(f"Loading model from: {load_path}", self.testlogname)
        self.checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.actor = mc.DDPGActor(self.obs_dim, self.act_dim, self.hidden_sizes).to(self.device)
        self.actor.load_state_dict(self.checkpoint['actor_state_dict'])
        self.actor.eval()
        log.log("Model loaded successfully", [self.testlogname, "console"])
        
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

class PPOTester(ModelTester):
    def __init__(self, filename : str, environment: gym.Env):
        """Class for testing a trained PPO-based model.
        
        Args:
            filename (string): The name of the saved model file, saved at ./../models/filename
            environment (gym.env): object to test the environment on, preferably unwrapped.
        """        
        super().__init__(filename, environment)

        # PPO specific init
        self.network = None
        self.hidden_sizes = [128, 128]
        
        # Determine if continuous or discrete action space
        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)
        if not self.continuous:
            self.act_dim = self.env.action_space.n

        log.log(f"PPO Tester initialized - Continuous: {self.continuous}, Hidden sizes: {self.hidden_sizes}", self.testlogname)
        self._load(filename)

    def _load(self, filename: str):
        """Load PPO model from the models folder."""
        load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", filename))
        log.log(f"Loading model from: {load_path}", self.testlogname)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.network = mc.ActorCriticNetwork(self.obs_dim, self.act_dim, self.continuous, self.hidden_sizes).to(self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        log.log("Model loaded successfully", [self.testlogname, "console"])

    def _get_action(self, state):
        """Function to get action from PPO-trained network (deterministic for testing)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(state_tensor, deterministic=True)
        
        action = action.squeeze(0).cpu().numpy()
        
        if not self.continuous:
            action = int(action)
        
        return action

    def test(self, n_episodes : int = 10, visual : bool = False):
        return super().test(n_episodes=n_episodes, visual=visual)
    