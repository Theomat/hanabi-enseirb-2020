"""Greedy Agent."""

from hanabi_learning_environment.rl_env import Agent
from partial_belief import PartialBelief

import numpy as np


class GreedyAgent(Agent):

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.reset(config)
        self.play_threshold = .9
        self.discard_thresholds = [.4 + .5 * i / self.max_information_tokens for i in range(self.max_information_tokens + 1)]

    def act(self, observation):
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            return None

        # Update beliefs
        for belief in self.player_beliefs:
            belief.update(observation)

        # Check if can play
        fireworks = observation["fireworks"]
        for card_index, hint in enumerate(observation['card_knowledge'][0]):
            if self.player_beliefs[0].probability_playable(card_index, fireworks) >= self.play_threshold:
                return {'action_type': 'PLAY', 'card_index': card_index}

        # Check if can discard
        if observation["information_tokens"] < self.max_information_tokens:
            best_discard = None
            best_prob = -1
            # We need to remember the best discard choice in case we are forced to discard
            discard_threshold = self.discard_thresholds[observation['information_tokens']]
            for card_index, hint in enumerate(observation['card_knowledge'][0]):
                p = self.player_beliefs[0].probability_useless(card_index, fireworks)
                if p > discard_threshold:
                    return {'action_type': 'DISCARD', 'card_index': card_index}
                elif p > best_prob:
                    best_prob = p
                    best_discard = {'action_type': 'DISCARD', 'card_index': card_index}

            if observation['information_tokens'] == 0:
                # print("[INFO] DISCARD:", best_discard["card_index"])
                if best_prob == -1:
                    best_discard = {'action_type': 'DISCARD',
                                    'card_index': np.argmax([self.player_beliefs[0].probability_useless(offset, fireworks)
                                                             for offset in range(len(observation['card_knowledge'][0]))])}
                return best_discard

        # Check if can hint
        best_info = -9999
        best_action = None
        for player in range(1, len(self.player_beliefs)):
            belief = self.player_beliefs[player]
            action, info = belief.most_informative_hint(observation["observed_hands"][player])
            if info > best_info:
                best_info = info
                action["target_offset"] = player
                best_action = action
        if self.verbose:
            print(f"Chose hint that gave {best_info:.2f} bits of information.")
        return best_action

    def on_action(self, agent_id: int, action, hand):
        self.player_beliefs[(agent_id - self.id) % len(self.player_beliefs)].update_with_action(action)
        if action["action_type"].startswith("REVEAL"):
            target = action["target_offset"]
            self.player_beliefs[(agent_id - self.id + target) % len(self.player_beliefs)].update_with_hint(action, hand)

    def reset(self, config):
        self.config = config
        self.verbose = config.get('verbose', False)
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)
        self.player_beliefs = [PartialBelief(config.get('players', 2), player, 0)
                               for player in range(config.get('players', 2))]
