import os
import sys
from datetime import datetime

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3 import A2C
from sumo_rl import SumoEnvironment
from generator import generator



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


no_of_episodes = 10


if __name__ == "__main__":

    now = datetime.now()
    local_time = now.strftime("%d-%m_%H-%M")


    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        route_file='training/routes/2x2_tr_ep2.rou.xml',
        single_agent=True,
        use_gui=True,
        num_seconds=5400,
    )

    model = A2C.load(f"training/models/A2C_30-06_22-40_Episode_20",env = env, print_system_info=True)

    env.reset()
    
    obs, rewards, done,done2, info = env.step(0)
    done = False
    while not done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards,done,done2,  info = env.step(action)
        env.render()

    env.close()