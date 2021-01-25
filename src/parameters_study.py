from greedy_agent import GreedyAgent
from game_runner import GameRunner

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


config = {'players': 2, 'verbose': False}
EPISODES = 100

DISCARDS = [.1 * i for i in range(11)]
PLAYS = [.5 + .05 * i for i in range(11)]

max_rewards = []
mean_rewards = []
median_rewards = []
for play_threshold in PLAYS:
    max_reward = []
    mean_reward = []
    median_reward = []
    for base_discard in DISCARDS:
        agents = [GreedyAgent(config) for i in range(2)]
        for agent in agents:
            agent.play_threshold = play_threshold
            agent.discard_thresholds = [base_discard + i * (play_threshold - base_discard) / 9 for i in range(9)]
        runner = GameRunner(agents, EPISODES, config)
        runner.run()
        rewards = runner.rewards
        max_reward.append(np.max(rewards))
        mean_reward.append(np.mean(rewards))
        median_reward.append(np.median(rewards))
    max_rewards.append(max_reward)
    mean_rewards.append(mean_reward)
    median_rewards.append(median_reward)

# Plot 3D data
fig = plt.figure()

X = np.array([[PLAYS[i] for j in range(len(DISCARDS))] for i in range(len(PLAYS))])
Y = np.array([DISCARDS[:] for i in range(len(PLAYS))])

ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(X, Y, np.array(max_rewards), label="max")
plt.xlabel("Play threshold")
plt.ylabel("Discard threshold")
plt.title("Max reward")
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(X, Y, np.array(mean_rewards), label="mean")
plt.xlabel("Play threshold")
plt.ylabel("Discard threshold")
plt.title("Mean reward")
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(X, Y, np.array(median_rewards), label="median")
plt.xlabel("Play threshold")
plt.ylabel("Discard threshold")
plt.title("Median reward")
plt.show()
