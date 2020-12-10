# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Agent."""

from hanabi_learning_environment.rl_env import Agent

# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

# from PIL import Image

import torch

import torch.optim as optim

import torchvision.transforms as T

from replay_memory import ReplayMemory
from dqn import DQN

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(
        self, config, encoded_observation_size, *args, **kwargs
    ):  # !! There must be a way to avoid "encoded_observation_size" being a parameter
        """Initialize the agent."""
        self.config = config
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get("information_tokens", 8)
        # DQN Params
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        # initialise DQN
        self.n_actions = (
            2 * config["hand_size"] + 2 * config["players"] * config["hand_size"]
        )  #!! handcoded... should depend on config or smth

        self.policy_net = DQN(
            input_size=encoded_observation_size, output_size=self.n_actions
        ).to(device)
        self.target_net = DQN(
            input_size=encoded_observation_size, output_size=self.n_actions
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    @staticmethod
    def playable_card(card, fireworks):
        """A card is playable if it can be placed on the fireworks pile."""
        return card["rank"] == fireworks[card["color"]]

    def build_action_space(self, observation):
        """
    returns all possible actions in an ordered list
    """
        action_space = []
        for i in range(len(observation["observed_hands"][0])):
            action_space.append({"action_type": "PLAY", "card_index": i})
            action_space.append({"action_type": "DISCARD", "card_index": i})

        for player_offset in range(1, observation["num_players"]):
            player_hand = observation["observed_hands"][player_offset]
            for card in player_hand:
                action_space.append(
                    {
                        "action_type": "REVEAL_COLOR",
                        "color": card["color"],
                        "target_offset": player_offset,
                    }
                )
                action_space.append(
                    {
                        "action_type": "RevealRank",
                        "rank": card["rank"],
                        "target_offset": player_offset,
                    }
                )

        return action_space

    def select_action(self, observation):
        action_space = self.build_action_space(observation)
        if observation["current_player_offset"] != 0:
            return None
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                ordered_moves = self.policy_net(observation["vectorized"]).argsort(
                    descending=True
                )
                i = 0
                action_index = ordered_moves[i].view(1, 1)
                while action_space[action_index] not in observation["legal_moves"]:
                    i += 1
                    action_index = ordered_moves[i].view(1, 1)
                return action_space[action_index]
        else:
            action = action_space[random.randrange(self.n_actions)]
            while action not in observation["legal_moves"]:
                action = action_space[random.randrange(self.n_actions)]
            return action

