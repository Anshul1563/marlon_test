import argparse
import os
import sys
import pandas as pd
from generator import generator


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from stable_baselines3.dqn.dqn import DQN



alpha = 0.1
gamma = 0.99
decay = 1
runs = 1
episodes = 10

df = pd.DataFrame(columns=['episode', 'reward'])


for episode in range(1, episodes + 1):
	generator('nets/2x2grid/2x2.net.xml', f'training/routes/2x2_tr_ep{episode}.rou.xml', 5400, 500, episode)
	
	env = SumoEnvironment(
	net_file="nets/2x2grid/2x2.net.xml",
	route_file=f'training/routes/2x2_tr_ep{episode}.rou.xml',
	use_gui=True,
	out_csv_name=f"outputs/4x4/QL_4x4",
	num_seconds=5400,
	add_system_info = True,
	min_green=5,
	delta_time=5,
	)

	initial_states = env.reset()

	ql_agents = {
	ts: QLAgent(
		starting_state=env.encode(initial_states[ts], ts),
		state_space=env.observation_space,
		action_space=env.action_space,
		alpha=alpha,
		gamma=gamma,
		exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
	)
	for ts in env.ts_ids
	}

	if episode != 1:
		initial_states = env.reset()
		for ts in initial_states.keys():
			ql_agents[ts].state = env.encode(initial_states[ts], ts)

	infos = []
	done = {"__all__": False}
	cumalative_reward = 0


	while not done["__all__"]:
		actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

		s, r, done, info = env.step(action=actions)
		cumalative_reward += sum(r.values())
		for agent_id in s.keys():
			ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])    

	df2 = {'episode': episode, 'reward': cumalative_reward}
	df = df._append(df2, ignore_index = True)
	env.close()

print(df)
df.to_csv('training/outputs/QL_2x2_Rewards.csv', index=False)
