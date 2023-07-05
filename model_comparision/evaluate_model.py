import os
import sys
from datetime import datetime

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3 import A2C
from sumo_rl import SumoEnvironment





if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")




now = datetime.now()
local_time = now.strftime("%d-%m_%H-%M")


env = SumoEnvironment(
    net_file="nets/2way-single-intersection/single-intersection.net.xml",
    route_file='training/routes/2x2_tr_ep2.rou.xml',
    single_agent=True,
    use_gui=True,
    num_seconds=5400,
)

models = [A2C.load(f"training/models/A2C_30-06_22-40_Episode_20",env = env, print_system_info=True),DQN.load("training/models/DQN_30-06_20-58_Episode_20",env = env, print_system_info=True)]



for model in models:
    obs = env.reset()
    obs = obs[0]
    done2 = False

    while not done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards,done,done2,  info = env.step(action)
        env.render()

    env.close()