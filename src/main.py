import argparse
import os
import sys
import subprocess
from pathlib import Path
import torch
import gymnasium as gym

# Local imports
from train import PPOTrainer, SACTrainer, DDPGTrainer, A2CTrainer
from test_models import PPOTester, SACTester, DDPGTester, A2CTester
import multilogger as ml

# Initialize logger
log = ml.MultiLogger()
log.add_output("console", sys.stdout, timestamps=True)
log.log("DRL Control Interface Initialized")

# Cheesed out way to make the training log print to same console
import train
train.log = log

# Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ENVS_DIR = BASE_DIR / ".." / "envs"

def select_algorithm(name: str, env: str):
    name = name.lower()
    if name == "ppo":
        return PPOTrainer(env, reward_mode="ppo")
    elif name == "sac":
        return SACTrainer(env, reward_mode="sac")
    elif name == "ddpg":
        return DDPGTrainer(env, reward_mode="ddpg")
    elif name == "a2c":
        return A2CTrainer(env, reward_mode="a2c")
    else:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from: PPO, SAC, DDPG, A2C")

def select_tester(name: str, model_path: str, env: str):
    name = name.lower()
    if name == "ppo":
        return PPOTester(model_path, env)
    elif name == "sac":
        return SACTester(model_path, env)
    elif name == "ddpg":
        return DDPGTester(model_path, env)
    elif name == "a2c":
        return A2CTester(model_path, env)
    else:
        raise ValueError(f"No tester implemented for algorithm '{name}'")

def train_agent(trainer, total_steps: int, save_name: str):
    log.log(f"Starting training for {trainer.__class__.__name__} on {trainer.env_name}")
    results = trainer.train(total_timesteps=total_steps, save_path=save_name)
    trainer.plot_training(path=f"{trainer.env_name}_{trainer.__class__.__name__}_training.png")
    log.log("Training complete.")
    return results

def load_agent(algorithm: str, env: str, filename: str):
    trainer = select_algorithm(algorithm, env)
    trainer.load(filename)
    log.log(f"Loaded {algorithm.upper()} model from {filename}")
    return trainer

def discover_env_scripts(envs_dir: Path):
    """Scan envs folder for *_game folders and return play/scripted modes."""
    modes = {}
    for folder in envs_dir.iterdir():
        if folder.is_dir() and folder.name.endswith("_game"):
            env_name = folder.name.replace("_game", "")
            core_file = folder / f"{env_name}_core.py"
            env_file = folder / f"{env_name}_env.py"
            if core_file.exists():
                modes[f"play_{env_name}"] = core_file
            if env_file.exists():
                modes[f"scripted_{env_name}"] = env_file
    return modes

def run_script(script_path: Path):
    """Launch an environment script."""
    if not script_path.exists():
        log.log(f"[ERROR] Could not find script: {script_path}")
        return
    log.log(f"Running script: {script_path.relative_to(BASE_DIR)}")
    subprocess.run([sys.executable, str(script_path)], check=False)

def main():
    env_modes = discover_env_scripts(ENVS_DIR)
    valid_modes = ["train", "resume", "test"] + list(env_modes.keys())

    parser = argparse.ArgumentParser(
        description="Unified control interface for DRL training, testing, and demos"
    )
    parser.add_argument("--algo", type=str, help="Algorithm to use: PPO, SAC, or DDPG")
    parser.add_argument("--env", type=str, help="Gymnasium environment name, e.g., 'RacingEnv-v0'")
    parser.add_argument("--mode", type=str, required=True, choices=valid_modes,
                        help=f"Mode: {', '.join(valid_modes)}")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for testing")
    parser.add_argument("--render", action="store_true", help="Render during test")
    parser.add_argument("--model", type=str, default=None, help="Path to model file for saving/loading")
    args = parser.parse_args()

    # Default model filename
    model_file = args.model or (f"{args.algo.lower()}_{args.env}_model.pt" if args.algo and args.env else None)
    model_path = str(MODELS_DIR / model_file) if model_file else None

    # Handle demo/play modes dynamically
    if args.mode in env_modes:
        run_script(env_modes[args.mode])
        return

    # Algorithmic modes
    if not args.algo or not args.env:
        log.log("[ERROR] --algo and --env are required for this mode.")
        sys.exit(1)

    env = gym.make(args.env, render_mode="human" if args.render else None)

    if args.mode == "train":
        trainer = select_algorithm(args.algo, args.env)
        train_agent(trainer, total_steps=args.steps, save_name=model_file)

    elif args.mode == "resume":
        trainer = load_agent(args.algo, args.env, model_file)
        log.log("Resuming training from saved model...")
        train_agent(trainer, total_steps=args.steps, save_name=model_file)

    elif args.mode == "test":
        tester = select_tester(args.algo, model_file, env)
        rewards, infos, terminals = tester.test(n_episodes=args.episodes, visual=args.render)
        avg_reward = sum(rewards) / len(rewards)
        log.log(f"Testing complete. Average reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()