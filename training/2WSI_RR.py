import os
import sys
from datetime import datetime
from time import sleep

from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


no_of_episodes = 5


env = SumoEnvironment(
            net_file="nets/2way-single-intersection/single-intersection.net.xml",
            route_file=f'training/routes/2x2_tr_ep2.rou.xml',
            out_csv_name="./RR_2WSI_1",
            use_gui=True,
            num_seconds=5400,
            delta_time = 23,
            yellow_time= 3,
            min_green= 20
        )


env.reset()

i = 0

while True :
    i +=1
    action = i % 4
    new_obs, rewards, dones, infos = env.step({'t' : action})
    if True in dones.values() :
        break
    print(f"Step {i} Action {action}")
