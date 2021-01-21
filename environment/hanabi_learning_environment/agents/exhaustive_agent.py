"""Extensive search of all possible moves, """
# On peut, outre le score de partie basique, donner plus de poids à d'autres choses (donc créer une heuristique) -> indices valent des points par exemple,
#ou score non-linéaire (poser le 5 vaut plus que poser le 2)
""" memo:                       {'current_player': 0,
                                  'current_player_offset': 0,
                                  'deck_size': 40,
                                  'discard_pile': [],
                                  'fireworks': {'B': 0,
                                                'G': 0,
                                                'R': 0,
                                                'W': 0,
                                                'Y': 0},
                                  'information_tokens': 8,
                                  'legal_moves': [{'action_type': 'PLAY',
                                                   'card_index': 0},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 1},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 2},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 3},
                                                  {'action_type': 'PLAY',
                                                   'card_index': 4},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'R',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'G',
                                                   'target_offset': 1},
                                                  {'action_type':
                                                  'REVEAL_COLOR',
                                                   'color': 'B',
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 0,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 1,
                                                   'target_offset': 1},
                                                  {'action_type': 'REVEAL_RANK',
                                                   'rank': 2,
                                                   'target_offset': 1}],
                                  'life_tokens': 3,
                                  'observed_hands': [[{'color': None, 'rank':
                                  -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1},
                                                      {'color': None, 'rank':
                                                      -1}],
                                                     [{'color': 'G', 'rank': 2},
                                                      {'color': 'R', 'rank': 0},
                                                      {'color': 'R', 'rank': 1},
                                                      {'color': 'B', 'rank': 0},
                                                      {'color': 'R', 'rank':
                                                      1}]],
                                  'num_players': 2,
                                  'vectorized': [ 0, 0, 1, ... ]}
"""
#bool GetDealSpecificMove(int card_index, int player, int color, int rank, pyhanabi_move_t* move)

# TODO: use " HanabiState : get_deal_specific_move(card_index, player, color, rank) " in conjunction with " HanabiState : apply_move(self, move)" from " pyhanabi.py "
# and maybe " HanabiState : copy(self) " (to save before a move, but it's pretty heavy to do it before every move)

import sys
import os
import numpy as np
import math  
import copy
import random 
import time
#DOSSIER_COURANT = os.path.dirname(os.path.abspath(__file__))
#DOSSIER_PARENT = os.path.dirname(DOSSIER_COURANT)
#print(DOSSIER_PARENT)
#sys.path.append(DOSSIER_PARENT) sleep print


#from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.rl_env import Agent
from hanabi_learning_environment.pyhanabi import color_char_to_idx, color_idx_to_char
from hanabi_learning_environment.pyhanabi import HanabiGame, HanabiState, HanabiMoveType, HanabiMove
from hanabi_learning_environment.pyhanabi import CHANCE_PLAYER_ID
import hanabi_learning_environment.partial_belief as pb # Remember to use update function
#from hanabi_learning_environment
from hanabi_learning_environment.pyhanabi import ObservationEncoder, ObservationEncoderType


def recDichoSearchCard(card, cards, i, j):
    """ The cards must be in order, first colors, then ranks (like : (0 (c),0 (r)) (0,1) (1,1)  (1,2) ...) """
    if (j - i) > 0:
        center = (i + j) // 2
        if (
            color_char_to_idx(cards[center]["color"]) == card["color"]
            and cards[center]["rank"] == card["rank"]
        ):
            return center
        elif color_char_to_idx(cards[center]["color"]) == card["color"]:
            if cards[center]["rank"] > card["rank"]:
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
        else:
            if color_char_to_idx(cards[center]["color"]) > card["color"]:
                return recDichoSearchCard(card, cards, i, center - 1)
            else:
                return recDichoSearchCard(card, cards, center + 1, j)
    elif (
        j - i
    ) == 0:  # Useless, when (j - i) == -1, it will return -1 (all elements already checked)
        if color_char_to_idx(cards[i]["color"]) == card["color"] and cards[i]["rank"] == card["rank"]:
            return i
    return -1


def dichoSearchCard(card, cards):
    if card["color"] != None and card["rank"] != -1 and cards != []:
        card["color"] = color_char_to_idx(str(card["color"]))
        i = 0
        j = len(cards) - 1
        return recDichoSearchCard(card, cards, i, j)
    return -1


class ExtensiveAgent(Agent):
    """Agent that applies an exhaustive search to find the best possible move."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        """ Args:
      config: dict, With parameters for the game. Config takes the following
        keys and values.
          - colors: int, Number of colors \in [2,5].
          - ranks: int, Number of ranks \in [2,5].
          - players: int, Number of players \in [2,5].
          - hand_size: int, Hand size \in [4,5].
          - max_information_tokens: int, Number of information tokens (>=0)
          - max_life_tokens: int, Number of life tokens (>=0)
          - seed: int, Random seed.
          - random_start_player: bool, Random start player.
          - max_iteration: int, Maximum Depth of the search. """

        self.config = config
        print("La configuration de la game:", config)
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get("information_tokens", 8)
        self.max_iteration = config.get("max_iteration", 2)
        self.config["random_start_player"] = False # To start at 0
        self.global_game = HanabiGame(self.config)
        self.global_game_state = self.global_game.new_initial_state()
        self.observation_encoder = ObservationEncoder(
            self.global_game, ObservationEncoderType.CANONICAL)

        # Both are used in the expected values computation, to limit the time needed
        self.threshold_random = 1000
        self.limit_tests = 0

        self.hands_initialized = False
        self.saved_observation = None
        self.previous_observation = None
        self.offset_real_local = None # The offset to go from the real player id to the local player id (because the starting player can't be chosen precisely, only 0 or random)
                                                                                                 #must use its local id (for coherence)
    

    def produce_current_state_observation(self, local_player_id = None, state = None):
        if local_player_id is None:
            local_player_id = self.local_id(0)
        if state is None:
            return self._extract_dict_from_backend(local_player_id, self.global_game_state.observation(local_player_id))
        else:
            return self._extract_dict_from_backend(local_player_id, state.observation(local_player_id), state)
    
    def assure_agent_hand_compatible(self, card_to_verify, available = None, player_offset = 0):
        #print("assure_agent_hand_compatible debut:", available, self.global_game_state.player_hands, card_to_verify ,flush = True)
        verif_color = color_char_to_idx(card_to_verify["color"])
        verif_rank = card_to_verify["rank"]
        player_local_id = (player_offset + self.local_id(0)) % self.config["players"] 
        if available is None:
            available_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation())
        else:
            available_cards = available
        # We only need to free one instance of card_to_verify at each call, so we can stop after that (if it's needed)
        for n, card in enumerate(self.global_game_state.player_hands()[player_local_id]):
            if card.color() == verif_color and card.rank() == verif_rank:
                for i in range(len(available_cards)):
                    if i != verif_color:
                        for j in range(len(available_cards[i])):
                            if j != verif_rank and available_cards[i][j] > 0:
                                available_cards[i][j] -= 1 # We just decided to take it
                                available_cards[verif_color][verif_rank] += 1 # We just freed it
                                for k in range(self.config["players"]):
                                    #print(self.global_game_state.player_hands(), k, k, k, k)
                                    self.global_game_state.set_individual_card(player_local_id, n, 
                                        {"color": color_idx_to_char(i), "rank": j})
                                #print("assure_agent_hand_compatible fin:", available, self.global_game_state.player_hands, flush = True)
                                return available_cards
            
        return available_cards

    @staticmethod
    def count_real_moves(l):

        count = 0
        
        for current_history_item in l:
            current_move = current_history_item.move()
            current_move_type = current_move.type()
            
            if (current_move_type == HanabiMoveType.PLAY
                or current_move_type == HanabiMoveType.DISCARD
                or current_move_type == HanabiMoveType.REVEAL_RANK
                or current_move_type == HanabiMoveType.REVEAL_COLOR
                or current_move_type == HanabiMoveType.INVALID):
                count += 1
                
        return count
    
    def select_card(self, condition, respect_condition, type_of_condition, l): # TODO: Check the current hand 
        """ If respect_condition == True, give a good card, if not, give a bad card (relative to the condition). """

        for i in range(len(l)):
            # The first-level condition is used to filter the color-typed condition
            if ((type_of_condition == "color" and respect_condition == (condition == i))
                or (type_of_condition == "rank")):
                for j in range(len(l[i])):
                    if ((type_of_condition == "rank" and respect_condition == (condition == j))
                        or (type_of_condition == "color")):
                        if l[i][j] > 0:
                            l[i][j] -= 1
                            return l, {"color": color_idx_to_char(i), "rank": j}
        return None
                

    
    def draw_good_card(self, local_id, observation):
        #print("Starting draw_good_card:", local_id)

        # If there's no more card, we can't give a card
        if self.global_game_state.deck_size() == 0:
            return 


        self.global_game_state.deal_random_card()

        relative = self.relative_id(local_id)
        available_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation())
                    
        # Preparing the available cards by reserving the useful ones
        
        # If not, it is the agent, and we don't know which card to give 
        if relative != 0:
            for card in self.global_game_state.player_hands()[self.local_id(0)]: #TODO: hand = agent hand
                available_cards[card.color()][card.rank()] -= 1
            #print("Qu'est ce que c'est que cette erreur sur la couleur ?:",observation["observed_hands"][relative][-1], flush = True)
            self.assure_agent_hand_compatible(observation["observed_hands"][relative][-1], available_cards)
            # The same bug explanation for this loop
            card_to_set = observation["observed_hands"][relative][-1]
            for i in range(self.config["players"]):
                self.global_game_state.set_individual_card(local_id, self.config["hand_size"] - 1, card_to_set)

        return 0

    def prepare_hand(self, local_id, observation, type_of_reveal, value):
        #print("Starting prepare_hand:", local_id, flush = True)
        relative = self.relative_id(local_id)

        # If not, it is theorically useless to change the hand (unless somme mistakes have been made previously, creating a wrong state)
        if relative == 0:
            hand_to_be_set = []
            all_cards_in_hand = self.global_game_state.player_hands()[self.local_id(0)]
            possible_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation()) 
            #print("Possible cards:", possible_cards)

            #TODO: CHECK IF NECESSARY, BECAUSE CAN CRASH IN END-GAME !!! We reserve the cards currently in hand (maybe not necessary)
            #for card in all_cards_in_hand:
            #    possible_cards[card.color()][card.rank()] -= 1
            player_knowledge = observation["pyhanabi"].card_knowledge()[relative]
            for n, card_knowledge in enumerate(player_knowledge):
                card = None
                if type_of_reveal == "color":
                    if card_knowledge.color() == value:
                        possible_cards, card = self.select_card(value, True, type_of_reveal, possible_cards)
                    else:
                        possible_cards, card = self.select_card(value, False, type_of_reveal, possible_cards)
                elif type_of_reveal == "rank":
                    if card_knowledge.rank() == value:
                        possible_cards, card = self.select_card(value, True, type_of_reveal, possible_cards)
                    else:
                        possible_cards, card = self.select_card(value, False, type_of_reveal, possible_cards)
                else:
                    #print("Bad reveal_type in prepare_hand:", local_id, observation, type_of_reveal, value, flush=True)
                    return -1
                #possible_cards[all_cards_in_hand[n].color()][all_cards_in_hand[n].rank()] += 1
                if card is not None:
                    hand_to_be_set.append(card)
                else:
                    #TODO: This case must be treated, by replacing the already present cards in hand (assure_agent_hand_compatible function ?)
                    print("Pas assez de carte pour compléter la main correctement:", hand_to_be_set, possible_cards, flush=True)
                    return -1
            #print("Hand to be set:", hand_to_be_set, flush = True)
            #time.sleep(1)
            for i in range(self.config["players"]):
                self.global_game_state.set_hand(local_id, hand_to_be_set)

        
        return 0
                

    def print_state_info(self):
        print("Hands of all the players:", self.global_game_state.player_hands(), flush = True)
        if self.offset_real_local is not None:
            print("Hanabi Observation:", self.global_game_state.observation(self.offset_real_local), flush = True)
        pass

    def local_id(self, i):
        """ The local id of the player in global_game_state. """
        return (i + self.offset_real_local) % self.config["players"]

    def relative_id(self, i):
        """ The relative place of the player (his offset compared to the agent) """
        return (i - self.offset_real_local) % self.config["players"]
                
    def prepare_global_game_state(self, observation):
        """ This function build the state thanks to the informations given in the observation. """
        obs_pyhanabi = observation["pyhanabi"]
        
        ################## FIRST TURN PART ###################
        if self.offset_real_local is None:
            while self.global_game_state.cur_player() == CHANCE_PLAYER_ID:
                self.global_game_state.deal_random_card()

            # The number of players who have played before the first call
            self.offset_real_local = ExtensiveAgent.count_real_moves(obs_pyhanabi.last_moves()) # Mayb another way ?

            current_player = self.global_game_state.cur_player()

            print("Avant l'initialisation", flush = True)
            print(self.global_game_state.player_hands(), flush = True)

            #Possible probleme si on a 9 joueurs (donc pas de deck, mais on s'en fiche un peu)
            available_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation())


            # First, we gather the available cards
            for i, hand in enumerate(self.saved_observation["observed_hands"]):

                # Our hand is invalid, so we start from the 2nd player 
                if i != 0:
                    # Preparing the available cards by reserving the useful ones
                    for card in hand:
                        available_cards[color_char_to_idx(card["color"])][card["rank"]] -= 1
            
            hands_to_be_removed = self.global_game_state.player_hands()
            for i in range(self.config["players"]):
                for j in range(self.config["hand_size"]):
                    current_card = hands_to_be_removed[i][j]
                    available_cards[current_card.color()][current_card.rank()] -= 1


            # Then, we prepare each hand to be compatible with the hands to be set
            for i, hand in enumerate(self.saved_observation["observed_hands"]):
                    
                if i != 0:
                    for card in hand:
                        print("On vire la carte:", card, flush = True)
                        print("Dans les mains", self.global_game_state.player_hands() , flush = True)
                        print("Carte disponibles:", available_cards)
                        for p in range(self.config["players"]):
                            available_cards = self.assure_agent_hand_compatible(card, available_cards, player_offset = p)

            for i, hand in enumerate(self.saved_observation["observed_hands"]):

                # Our hand is invalid, so we start from the 2nd player 
                if i != 0:
                    #print("The current hands on the board, and the hand to be set:", self.global_game_state.player_hands(), hand)
                    self.global_game_state.set_hand(self.local_id(i), hand)

            #There's a bug in set_hand, so we have to correct the current_player by setting another time the last hand
            if self.global_game_state.cur_player() != current_player:
                self.global_game_state.set_hand(self.local_id(self.config["players"] - 1),self.saved_observation["observed_hands"][self.config["players"] - 1])

            print("Après l'initialisation", flush = True)
            print(self.global_game_state.player_hands(), flush = True)
            self.initialize_hands = True

            

        last_moves_ordered = obs_pyhanabi.last_moves()
        last_moves_ordered.reverse()
        print("Starting the generalpart of building", flush = True)
        ################## GENERAL PART ###################
        for current_history_item in last_moves_ordered:
            #print("Current history item:", current_history_item, flush = True)
            #print("Current Local player (to play):", self.global_game_state.cur_player())
            #print("Offset real / local (so the id of the agent):", self.offset_real_local, self.local_id(0))

            current_move = current_history_item.move()
            current_move_type = current_move.type()
            
            # The ExtensiveAgent is always player 0 in moves, so we have to correct this by adding the local index of the agent.
            #
            # EXAMPLE: the player given is 1, and we play at the second place in a game with 2 players.
            # The current_local_player is (1 + 1) % nb_player = 0, so it's the first player (which is true).
            relative_player = current_history_item.player()
            current_local_player = self.local_id(relative_player) 
            
            

            if current_move_type == HanabiMoveType.PLAY: # TODO: put the good cards in hand
                #print("Entering the PLAY condition", flush = True)
                if self.global_game_state.move_is_legal(current_move):
                    #print("PLAY condition verified", flush = True)

                    if relative_player == 0:
                        for i in range(self.config["players"]):
                            card = {"color": color_idx_to_char(current_history_item.color()), "rank": current_history_item.rank()}
                            available_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation())
                            hand = self.global_game_state.player_hands()[self.local_id(0)] # The agent's hand
                            
                            # Preparing the available cards by reserving the useful ones
                            for c in hand:
                                available_cards[c.color()][c.rank()] -= 1
                            available_cards = self.assure_agent_hand_compatible(card, available_cards)
                            self.global_game_state.set_individual_card(current_local_player, current_move.card_index(), 
                                card)
                            #print("PLAY move hand changed successfully", flush = True)
                    #time.sleep(1)

                    #print("LOCAL PLAYER ID BEFORE MOVE:", self.global_game_state.cur_player(), flush = True)
                    #print("LEGAL MOVES:", self. global_game_state.legal_moves())
                    self.global_game_state.apply_move(current_move)
                    #print("PLAY move applied successfully", flush = True)
                    self.draw_good_card(current_local_player, observation)
                    
                else:
                    print("The PLAY move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.DISCARD: # TODO: put the good cards in hand
                #print("Entering the DISCARD condition", flush = True)
                if self.global_game_state.move_is_legal(current_move):
                    #print("DISCARD condition verified", flush = True)
                    if relative_player == 0:
                        for i in range(self.config["players"]):
                            card = {"color": color_idx_to_char(current_history_item.color()), "rank": current_history_item.rank()}
                            #print("GENERAL, card to bet set:", card)
                            available_cards = ExtensiveAgent.unseen_cards(self.produce_current_state_observation())
                            hand = self.global_game_state.player_hands()[self.local_id(0)] # The agent's hand
                            # Preparing the available cards by reserving the useful ones
                            for c in hand:
                                available_cards[c.color()][c.rank()] -= 1
                            available_cards = self.assure_agent_hand_compatible(card, available_cards)
                            self.global_game_state.set_individual_card(current_local_player, current_move.card_index(), 
                                card)
                            #print("DISCARD move hand changed successfully", flush = True)
                    #time.sleep(1)
                    
                    #print("LOCAL PLAYER ID BEFORE MOVE:", self.global_game_state.cur_player(), flush = True)
                    #print("LEGAL MOVES:", self. global_game_state.legal_moves())
                    self.global_game_state.apply_move(current_move)
                    #print("DISCARD move applied successfully", flush = True)
                    self.draw_good_card(current_local_player, observation)
                    
                else:
                    print("The DISCARD move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.REVEAL_RANK:
                #print("Entering the REVEAL_RANK condition", flush = True)

                self.prepare_hand((current_local_player + current_move.target_offset()) % self.config["players"] , observation, "rank", current_move.rank())
                #time.sleep(1)
                #print("REVEAL_RANK hand set successfully", flush = True)
                if self.global_game_state.move_is_legal(current_move):
                    #time.sleep(1)
                    #print("REVEAL_RANK condition verified", flush = True)
                    #print("LOCAL PLAYER ID BEFORE MOVE:", self.global_game_state.cur_player(), flush = True)
                    self.global_game_state.apply_move(current_move)
                    #print("REVEAL_RANK move applied successfully", flush = True)
                    
                else:
                    print("The REVEAL_RANK move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.REVEAL_COLOR:
                #print("Entering the REVEAL_COLOR condition", flush = True)

                
                self.prepare_hand((current_local_player + current_move.target_offset()) % self.config["players"] , observation, "color", current_move.color())
                #time.sleep(1)
                #print("REVEAL_COLOR hand set successfully", flush = True)
                if self.global_game_state.move_is_legal(current_move):
                    #time.sleep(1)
                    #print("REVEAL_COLOR condition verified", flush = True)
                    #print("LOCAL PLAYER ID BEFORE MOVE:", self.global_game_state.cur_player(), flush = True)
                    self.global_game_state.apply_move(current_move)
                    #print("REVEAL_COLOR move applied successfully", flush = True)
                    
                else:
                    print("The REVEAL_COLOR move given isn't legal:", current_move, flush = True)
                    return -1

            elif current_move_type == HanabiMoveType.DEAL:
                pass

        self.print_state_info()
        return 0
        

    @staticmethod
    def transform_dict_to_move(dic):
        """ Transform a dict-shaped move into an HanabiMove """
        if dic["action_type"] == "PLAY":
            return HanabiMove.get_play_move(dic["card_index"])
        elif dic["action_type"] == "DISCARD":
            return HanabiMove.get_discard_move(dic["card_index"])
        elif dic["action_type"] == "REVEAL_COLOR":
            return HanabiMove.get_reveal_color_move(dic["target_offset"], dic["color"])
        elif dic["action_type"] == "REVEAL_RANK":
            return HanabiMove.get_reveal_color_move(dic["target_offset"], dic["rank"])
        else:
            print("Ce move n'existe pas, il n'est donc pas possible de le transformer en HanabiMove.")
            return None


    @staticmethod
    def unseen_cards(observation):
        currently_unseen_cards = np.array([
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1],
            [3, 2, 2, 2, 1]
        ], dtype=np.int) # Thx Theo for the numbers <3
        
        for card in observation["discard_pile"]:
            currently_unseen_cards[color_char_to_idx(card["color"])][card["rank"]] -= 1
        for i in range(len(observation["observed_hands"]) - 1):
            for card in observation["observed_hands"][i + 1]:
                currently_unseen_cards[color_char_to_idx(card["color"])][card["rank"]] -= 1
        for key in observation["fireworks"]:
            #print(observation["fireworks"], flush = True)
            for j in range(observation["fireworks"][key]):
                currently_unseen_cards[color_char_to_idx(key)][j] -= 1
        return currently_unseen_cards
    
    def _extract_dict_from_backend(self, player_id, observation, state = None): # Copied from rl_env !
        """Extract a dict of features from an observation from the backend.

    Args:
      player_id: Int, player from whose perspective we generate the observation.
      observation: A `pyhanabi.HanabiObservation` object.

    Returns:
      obs_dict: dict, mapping from HanabiObservation to a dict.
    """
        obs_dict = {}
        if state is None:
            obs_dict["current_player"] = self.global_game_state.cur_player()
        else:
            obs_dict["current_player"] = state.cur_player()
        obs_dict["current_player_offset"] = observation.cur_player_offset()
        obs_dict["life_tokens"] = observation.life_tokens()
        obs_dict["information_tokens"] = observation.information_tokens()
        obs_dict["num_players"] = observation.num_players()
        obs_dict["deck_size"] = observation.deck_size()

        obs_dict["fireworks"] = {}
        if state is None:
            fireworks = self.global_game_state.fireworks()
        else:
            fireworks = state.fireworks()
        for color, firework in zip(pyhanabi.COLOR_CHAR, fireworks):
          obs_dict["fireworks"][color] = firework

        obs_dict["legal_moves"] = []
        obs_dict["legal_moves_as_int"] = []
        for move in observation.legal_moves():
          obs_dict["legal_moves"].append(move.to_dict())
          #obs_dict["legal_moves_as_int"].append(self.global_game.get_move_uid(move))

        obs_dict["observed_hands"] = []
        for player_hand in observation.observed_hands():
          cards = [card.to_dict() for card in player_hand]
          obs_dict["observed_hands"].append(cards)

        obs_dict["discard_pile"] = [
            card.to_dict() for card in observation.discard_pile()
        ]

        # Return hints received.
        obs_dict["card_knowledge"] = []
        for player_hints in observation.card_knowledge():
          player_hints_as_dicts = []
          for hint in player_hints:
            hint_d = {}
            if hint.color() is not None:
              hint_d["color"] = color_idx_to_char(hint.color())
            else:
              hint_d["color"] = None
            hint_d["rank"] = hint.rank()
            player_hints_as_dicts.append(hint_d)
          obs_dict["card_knowledge"].append(player_hints_as_dicts)

        # ipdb.set_trace()
        obs_dict["vectorized"] = self.observation_encoder.encode(observation)
        obs_dict["pyhanabi"] = observation

        return obs_dict

    @staticmethod
    def score_game(state):
        """returns the game score displayed by fireworks played up to now in the game.
         for now no heuristic is used to determine which hand is the most promising for a given score"""
        score = 0
        fw = state.fireworks()
        for key in fw:
            score += fw[key]
        return score

                
    def calculate_all_possible_cards_and_prob(self, state, card_index, local_player_id, use_all_hands = False, available = None, already_tracked_hand = False): # We will do the probas with the cards we own ourselves, except for level 0,
                                                                        # because we've transformed the game into a completely known state
                                                                        # !!! BE CAREFUL, don't take into account the target card (it will be returned)
        #print("calculate_all_possible_cards_and_prob:",card_index,local_player_id, available)
        #TODO: assure that sum(probas) = 1 ! (normalize if needed)
        observation = self.produce_current_state_observation(local_player_id, state) #Check if the observation given is correct
        #print("Not Yet Implemented")
        possible_cards = []
        probabilities = []
        all_cards_count = 0 #TODO: transform in float before division(not now because operations are probably slower on floats)
        if available is None:
            available_cards = ExtensiveAgent.unseen_cards(observation) # TODO: return this, so the user can reuse it without recalculating this every call
        else:
            available_cards = available


        if use_all_hands and not(already_tracked_hand):
            available_cards = copy.deepcopy(available_cards) #We maybe don't want to modify the object given
            for current_card_index, current_card in enumerate(state.player_hands()[local_player_id]):
                # Use all the cards in the hand, except the card we want to change
                if card_index != current_card_index:
                    available_cards[current_card.color()][current_card.rank()] -= 1
            #print("Maintenant ca doit y être:", available_cards)
        

        ####################### The part where we use the hints #######################

        for c in range(self.config["colors"]):
            if observation["pyhanabi"].card_knowledge()[0][card_index].color_plausible(c): # 0 because it's from our point of view !
                for k in range(self.config["ranks"]):
                    if observation["pyhanabi"].card_knowledge()[0][card_index].rank_plausible(k): # same !
                        if available_cards[c][k] > 0:
                            # If this type of card is possible
                            all_cards_count += available_cards[c][k]
                            possible_cards.append({"color": color_idx_to_char(c), "rank": k})
                            probabilities.append(available_cards[c][k])

        all_cards_count = float(all_cards_count) # If == 0, the list comprehension will not explode, because there will be no elements to create anyway
        probabilities = [proba / all_cards_count for proba in probabilities]
        return available_cards, probabilities, possible_cards


    def enumerate_hands(self, possible_cards_in_each_position, current_indices_hand = None, nb_tests = 0): #, current_tested_hand
        """ Use the current tested hand indices, the observation and the possible cards in each hand position to return the next hand to be tested

        current_indices_hand is used to remember indices of the possible_cards_in_each_position 2D list, so we don't have to loose time searching
        for current tested cards at each call."""

        #If it's the first call, we give the first hand
        if current_indices_hand is None:
            current_indices_hand = [0] * self.config["hand_size"]
            return current_indices_hand, [possible_cards_in_each_position[i][current_indices_hand[i]] for i in range(self.config["hand_size"])]
        
        if nb_tests < self.threshold_random:
            current_position = 0
            while 1:
                current_indices_hand[current_position] += 1
                if len(possible_cards_in_each_position[current_position]) <= current_indices_hand[current_position]: # all the possible cards in this position has been tried
                    current_indices_hand[current_position] = 0
                    current_position += 1
                    if current_position >= self.config["hand_size"]:
                        return None # All the hands have been tested
                else:
                    return current_indices_hand, [possible_cards_in_each_position[i][current_indices_hand[i]] for i in range(self.config["hand_size"])]
        
        else:
            if self.limit_tests >= self.threshold_random:
                self.limit_tests = 0
                return None
            self.limit_tests += 1
            new_indices = [ random.randint(0, len(possible_cards_in_each_position[i]) - 1) for i in range(len(current_indices_hand))]
            return new_indices, [possible_cards_in_each_position[i][new_indices[i]] for i in range(self.config["hand_size"])]

    def hand_probability(self, current_indices_hand, list_proba_cards):
        proba = 1
        for card_index, indice in enumerate(current_indices_hand):
            proba *= list_proba_cards[card_index][indice]
        return proba

    def do_all_actions(self, all_actions, state, local_player_id, iteration_level, return_list = False, already_probas = True):
        if return_list:
            total = np.zeros(len(all_actions), dtype = float)
        else:
            total = 0
        
        available = None
        next_local_player_id = (local_player_id + 1) % self.config["players"]
        for action_index, action in enumerate(all_actions):
                tempo_state = state.copy() # Could be optimized a lot if a function "pop" was created (instead of cloning the entire state each time !)
                proba_current_move = 1 #Will be used in PLAY and DISCARD moves
                tempo_score = 0
                #print("game:",tempo_state._game)
                if tempo_state.move_is_legal(action):
                    if ((action.type() == HanabiMoveType.DISCARD 
                        or action.type() == HanabiMoveType.PLAY) and not(already_probas)): 
                        available = ExtensiveAgent.unseen_cards(self.produce_current_state_observation(local_player_id, tempo_state))
                        available, probabilities, possible_cards = self.calculate_all_possible_cards_and_prob(tempo_state, action.card_index(), local_player_id, use_all_hands = False, 
                            available = available, already_tracked_hand = False)
                        card_to_be_played = tempo_state.player_hands()[local_player_id][action.card_index()]
                        proba_indice = dichoSearchCard({"color": color_idx_to_char(card_to_be_played.color()), "rank": card_to_be_played.rank()}, possible_cards)
                        tempo_score = ExtensiveAgent.score_game(tempo_state)
                        if proba_indice == -1:
                            print("LA CARTE PRESENTE DANS LA MAIN N'EST PAS PRESENTE DANS LA LISTE DES POSSIBLES !!!", 
                                tempo_state.player_hands()[local_player_id][action.card_index()],
                                possible_cards)
                            return None
                        proba_current_move = probabilities[proba_indice]
                    tempo_state.apply_move(action)
                else:
                    print("THIS MOVE ISN'T LEGAL IN THE TEMPO_STATE:", action, tempo_state)
                    return None

                #If the game is finished after the move
                if tempo_state.is_terminal():
                    if return_list:
                        total[action_index] += ExtensiveAgent.score_game(tempo_state)
                    else:
                      total += ExtensiveAgent.score_game(tempo_state)
                    continue
                
                #An observation from the same player, but he hasn't the play, because he just played ! (TODO: if not used at all in REVEAL case, move it in the "if" condition)
                tempo_observation = self.produce_current_state_observation(local_player_id , tempo_state) 
                if (action.type() == HanabiMoveType.DISCARD 
                    or action.type() == HanabiMoveType.PLAY): 
                    
                    

                    #as usual, start with random, But it's theorically useless to assure that the card we want to set isn't already used in the hand 
                    #(because we've created a plausible game at iteration 0, so we can use all the hands in the state)  
                    tempo_state.deal_random_card()
                    #TODO: reuse the previous available, but remove the just-played card from it
                    available = ExtensiveAgent.unseen_cards(self.produce_current_state_observation(local_player_id, tempo_state)) 
                    


                    #print("available in all_moves:",tempo_state.fireworks(),available)
                    
                    # cards which can replace the card that has been played on a PLAY or DISCARD (last card, because just drawn)
                    _, proba_cards, cards = self.calculate_all_possible_cards_and_prob(tempo_state, self.config["hand_size"] - 1, local_player_id, True, available, False) 
                    for card_idx, card_possible in enumerate(cards):

                        # Always the same annoying bug ...
                        for p in range(self.config["players"]):
                            tempo_state.set_individual_card(local_player_id, self.config["hand_size"] - 1, card_possible) #Please be okay ;^;
                        if return_list:
                            total[action_index] += self.calculate_expected_value( iteration_level + 1, tempo_state, next_local_player_id) * proba_cards[card_idx]
                        else:
                            total += self.calculate_expected_value( iteration_level + 1, tempo_state, next_local_player_id) * proba_cards[card_idx]
                    total = total * proba_current_move 
                # So a REVEAL or something else (nothing to set, only informations were revealed (easier))
                else: 
                    if return_list:
                        total[action_index] += self.calculate_expected_value( iteration_level + 1, tempo_state, next_local_player_id)
                    else:
                        total += self.calculate_expected_value( iteration_level + 1, tempo_state, next_local_player_id) * proba_current_move + tempo_score * (1-proba_current_move)
                next_local_player_id = (local_player_id + 1) % self.config["players"]

        return total

    def is_possible_hand(self, current_hand, available_cards):
        #print("is_possible_hand debug:", current_hand, available_cards)
        test = np.zeros((self.config["colors"], self.config["hand_size"]), dtype = int)

        for card in current_hand:
            test[color_char_to_idx(card["color"])][card["rank"]] += 1

        for i in range(len(test)):
            for j in range(len(test[0])):
                if test[i][j] > available_cards[i][j]:
                    return False
        
        return True

    def calculate_expected_value(self, iteration_level, state, local_player_id):
        # THE STATE MUST NOT BE TOUCHED !

        if iteration_level >= self.max_iteration or state.is_terminal(): #TODO: In theory, not for iteration_level == 0, but maybe take into account this
            return ExtensiveAgent.score_game(state) # We will do the wheighted mean score of all children, so we have to return it

        observation =  self.produce_current_state_observation(local_player_id, state)

        if iteration_level != 0:
            total = 0
            #print("Secondary player")
            # The same next player for each move
            

            # The probabilities for having the card in hand must be taken into account when played
            total += self.do_all_actions(observation["pyhanabi"].legal_moves(), state, local_player_id, iteration_level, already_probas = False) 
            

                
            nb_moves = len(observation["pyhanabi"].legal_moves())
            if nb_moves == 0: # Is it really possible ? 
                print("In this case, the game is supposed to be finished !", local_player_id, iteration_level) 
                total = ExtensiveAgent.score_game(state)
            else:
                #total = total / float(nb_moves)
                pass

            return total
        else:
            print("Calculate_expected_value list else")
            # For each possible hand: # More optimized to iterate on hands, because it's harder to compute
            buffer_state = state.copy() # Only used to switch hands, to not touch the real state ! (Don't create weird errors)
            
            list_scores = []
            list_proba_cards, list_cards = [], []
            
            available = None
            for card_index in range(self.config["hand_size"]):
                # DON'T USE OTHER CARDS OF THE HAND (they are worth nothing) state
                available, proba_cards, cards = self.calculate_all_possible_cards_and_prob(buffer_state, card_index, local_player_id, False, available, True) 
                list_proba_cards.append(proba_cards)
                list_cards.append(cards)
            
            nb_tests = 1 # To have an idea of the time needed, and switch on another strategy if needed

            for index_hand_possibility in range(len(list_cards)):
                nb_tests *= len(list_cards[index_hand_possibility])
            res = self.enumerate_hands(list_cards)
            all_moves = observation["pyhanabi"].legal_moves()
            i_ref = 10000
            i = 0
            total = np.zeros(len(all_moves), dtype = float)
            while res is not None:
                list_indices, current_hand = res
                #print(".", end = " ", flush = True)
                #print("La liste des indices:",list_indices)
                #for i in range(5):
                #     print("la longueur des cartes possibles en ", i, ":", len(list_cards[i]), len(list_proba_cards[i]))
                if self.is_possible_hand(current_hand, available):
                    i += 1
                    if i >= i_ref:
                        i = 0
                        print("indices and hand:", list_indices, current_hand)

                    # Always the same annoying bug ...
                    for p in range(self.config["players"]):
                        buffer_state.set_hand(local_player_id, current_hand)

                    total += (self.do_all_actions(all_moves, buffer_state, local_player_id, iteration_level, return_list = True) 
                        * self.hand_probability( list_indices, list_proba_cards))

                res = self.enumerate_hands(list_cards, list_indices, nb_tests)

            print("Boucle terminée")
            nb_moves = len(all_moves)
            if nb_moves == 0: # Is it really possible ? 
                print("In this case, the game is supposed to be finished !", local_player_id, iteration_level) 
                total = []
            else:
                #total = total / float(nb_moves)
                pass

            return total 


    def act(self, observation):
        """Act based on an observation."""
        # The agent only plays on its turn
        # MEMO: produce_current_state_observation(self, local_player_id = None, state = None)
        
        if not(self.hands_initialized):
            self.saved_observation = observation
            self.hands_initialized = True
        if observation["current_player_offset"] != 0:
            return None
        print(observation)
        b = self.prepare_global_game_state(observation)
        if b == -1:
            return  -1
        chosen_random = random.randint(0,len(observation["legal_moves"]) - 1)
        #print(observation["legal_moves"])
        #return observation["legal_moves"][chosen_random]
        expected_value = self.calculate_expected_value( 0, self.global_game_state, self.local_id(0))
        print(expected_value)
        return observation["legal_moves"][np.argmax(expected_value)]


