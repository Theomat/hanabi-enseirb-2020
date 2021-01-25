from hanabi_learning_environment import rl_env

from typing import Dict

import numpy as np


class GameRunner():
    """
    Run games passing the right events for the agent to update its partial belief.
    """

    def __init__(self, agents, episodes: int, config: Dict = None):
        self.agents = agents
        self.episodes = episodes
        self.rewards = np.zeros(self.episodes)
        self.config = config or {"players": len(agents)}

    def run(self):
        for episode in range(self.episodes):
            environment = rl_env.make('Hanabi-Full', num_players=len(self.agents))
            observations = environment.reset()
            agents = self.agents
            for agent in agents:
                agent.reset(self.config)
            done = False
            episode_reward = 0
            turns = 0
            player_id = -1
            current_player_action = None
            while not done:
                for agent_id, agent in enumerate(agents):
                    agent.id = agent_id
                    observation = observations['player_observations'][agent_id]
                    if observations['current_player'] == agent_id:
                        action = agent.act(observation)
                        assert action is not None
                        current_player_action = action
                        player_id = agent_id
                observations, reward, done, unused_info = environment.step(current_player_action)
                if current_player_action is not None:
                    hand = None
                    # If an action is a reveal we need to pass the target hand
                    if current_player_action["action_type"].startswith("REVEAL"):
                        observed_hands = observations['player_observations'][player_id]["observed_hands"]
                        hand = observed_hands[current_player_action["target_offset"]]
                    # Pass action to agents
                    for agent_id, agent in enumerate(agents):
                        agent.on_action(player_id, current_player_action, hand)

                    # Make an environment step.
                episode_reward += reward
                turns += 1
            self.rewards[episode] = episode_reward
