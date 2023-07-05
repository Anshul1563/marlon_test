import os
import sys
from datetime import datetime

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment
from generator import generator



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


no_of_episodes = 20


if __name__ == "__main__":

    now = datetime.now()
    local_time = now.strftime("%d-%m_%H-%M")

    for episode in range(no_of_episodes) :

        generator('nets/2way-single-intersection/single-intersection.net.xml', f'training/routes/2x2_tr_ep{episode}.rou.xml', 5400, 500, episode)   

        env = SumoEnvironment(
            net_file="nets/2way-single-intersection/single-intersection.net.xml",
            route_file=f'training/routes/2x2_tr_ep{episode}.rou.xml',
            single_agent=True,
            use_gui=True,
            num_seconds=5400,
        )

        env.reset()

        if episode > 0 :
            model = DQN.load(f"training/models/DQN_{local_time}_Episode_{episode}",env = env, print_system_info=True)
        else :
            model = DQN(
                env=env,
                policy="MlpPolicy",
                learning_rate=0.001,
                learning_starts=0,
                train_freq=1,
                target_update_interval=200,
                exploration_initial_eps=0.05,
                exploration_final_eps=0.01,
                verbose=1,
            )
        
        model.learn(total_timesteps=(5400)/5)
        model.save(f"training/models/DQN_{local_time}_Episode_{episode + 1}")
        env.close()