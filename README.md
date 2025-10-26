# DRL-Assignment
assignment for topics in cs i

# Commands for the presentation:
- python main.py --mode demo_racing
- python main.py --mode test --episodes 1 --algo PPO --env RacingEnv --model pretrained_PPO_racing.pt --render
- python main.py --mode test --episodes 1 --algo SAC --env RacingEnv --model pretrained_SAC_racing.pt --render
- python main.py --mode test --episodes 1 --algo SAC --env TargetEnv --model pretrained_SAC_target.pt --render
- python main.py --mode test --episodes 1 --algo DDPG --env TargetEnv --model pretrained_DDPG_target.pt --render
- python main.py --mode test --episodes 1 --algo PPO --env TargetEnv --model pretrained_PPO_racing.pt --render (if I want to use consistent algorithms for pres)


## TODOLIST
# Before Submission
- Do data analysis in Jupyter notebooks
- Make it actually save csv's (can just do with MultiLogger?)
- Ensure README is an actual project README (probably a quickstart guide) not my tasklist
- Factcheck the docstrings to be correct (particularly in the used env.s)

# Dependencies
Gymnasium
pytorch
numpy
matplotlib

## Commands
A note on notation: () is being used to enclose variable values.

CWD is expected to be (your abs path)/DRL-Assignment/src

# Human Play - Special Mode
python main.py --mode play_(game)
Valid modes for this are racing and target (the environments agents play) and dungeon (because I wrote all that code and gosh darnit I will be including it somehow).
Running this command will allow a human to play the relevant game.

# Scripted Policies - Special Mode
python main.py --mode demo_(game)
Valid modes for this are racing and target.
Running this command will run the environment with the scripted policy used in SAC and DDPG warmup. Yes it is required. No the scripted policy is not good enough that you should count it as a hand-scripted bot I promise.

## Actual DRL Relevant Flags
# --mode
Valid args:
    train
    test
    resume
Admittedly resume is probably useless as I never bothered saving intermediate steps. But it's there.
# --env
Valid args:
    TargetEnv
    RacingEnv
    Any environment producable with Gymnasium.make
# --algo
Valid args:
    PPO
    SAC
    DDPG
    A2C
Required for testing AND training
# --steps
The amount of timesteps to use for training. Note that SAC and DDPG require many less then PPO.
# --episodes
The number of episodes to use for testing.
# --model
The file name to save to or load from, dependent on mode. Note that training and testing code already looks for these in the models folder, so don't try to use an abspath. Just use a filename.
# --render
Flag that when provided causes the program to render the environment. Do not use with episodes > 1 unless you want your computer to die.

Quickstart Guide:
1. Either review the code or make sure you trust me. The way I load the models can allow for ARBITRARY CODE EXECUTION. I have not added ACE or RCE exploits to my code, but you should probably make sure of that. Don't trust people on the internet. Also navigate to the previously specified working directory (your abspath)/DRL-Assignment/src
2. If you want to train new models use the following commands:
python main.py --mode train --steps 500000 --algo PPO --env RacingEnv --model PPO_racing.pt
python main.py --mode train --steps 500000 --algo PPO --env TargetEnv --model PPO_target.pt
python main.py --mode train --steps 150000 --algo SAC --env RacingEnv --model SAC_racing.pt
python main.py --mode train --steps 150000 --algo SAC --env TargetEnv --model SAC_target.pt
python main.py --mode train --steps 100000 --algo DDPG --env RacingEnv --model DDPG_racing.pt
python main.py --mode train --steps 100000 --algo DDPG --env TargetEnv --model DDPG_target.pt
python main.py --mode train --steps 300000 --algo A2C --env RacingEnv --model A2C_racing.pt
python main.py --mode train --steps 300000 --algo A2C --env TargetEnv --model A2C_target.pt
Alternatively pretrained ones are available at the following filenames:
pretrained_(PPO, SAC, DDPG, A2C)_(racing, target).pt
3. Take a look at the scripted policies used in the warmups so you can decide they aren't good enough to count as handscripted bot policies and that I don't deserve to be docked marks for including them.
python main.py --mode demo_racing
python main.py --mode demo_target
4. You can watch the models you just trained with the following commands:
python main.py --mode test --episodes 1 --algo PPO --env RacingEnv --model PPO_racing.pt --render
python main.py --mode test --episodes 1 --algo PPO --env TargetEnv --model PPO_target.pt --render
python main.py --mode test --episodes 1 --algo SAC --env RacingEnv --model SAC_racing.pt --render
python main.py --mode test --episodes 1 --algo SAC --env TargetEnv --model SAC_target.pt --render
python main.py --mode test --episodes 1 --algo DDPG --env RacingEnv --model DDPG_racing.pt --render
python main.py --mode test --episodes 1 --algo DDPG --env TargetEnv --model DDPG_target.pt --render
python main.py --mode test --episodes 1 --algo A2C --env RacingEnv --model A2C_racing.pt --render
python main.py --mode test --episodes 1 --algo A2C --env TargetEnv --model A2C_target.pt --render
5. Go inspect the jupyter notebooks that include some pretty snazzy data analysis. You should probably run the things too, I guess. They should come prefilled in with the results I did the analysis on though.

