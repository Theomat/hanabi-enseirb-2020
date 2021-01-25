from typing import Dict, List, Tuple

import numpy as np


_FULL_DECK_ = np.array([
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1],
    [3, 2, 2, 2, 1]
], dtype=np.int)

_COLOR_TO_INDEX = {
    "W": 0,
    "R": 1,
    "Y": 2,
    "G": 3,
    "B": 4,
    None: None
}


class PartialBelief:
    """
    Computes partial belief from the perspective of player ```perspective_of```.
    assuming the observations it receive are from the perspective of ```observations_from```.

    Parameters
    -----------
    - **players**: the number of players
    - **perspective_of**: the player offset whose perspective the beliefs will be build upon
    - **observation_from**: the player offset whose observations will be used to compute the beliefs
    """

    def __init__(self, players: int, perspective_of: int, observations_from: int):
        self.observation_from = observations_from
        self.perspective_of = perspective_of
        self.players = players
        self.hand_size = 5 if players <= 3 else 4
        self.hand = [{"rank": None, "color": None}] * self.hand_size
        self.possible_colors = np.ones((self.hand_size, 5), dtype=np.int)
        self.possible_ranks = np.ones((self.hand_size, 5), dtype=np.int)

    def _update_ck(self, observation: Dict):
        # Remove cards from the discard pile
        for card in observation["discard_pile"]:
            color = _COLOR_TO_INDEX[card["color"]]
            rank = card["rank"]
            self.deck[color, rank] -= 1

        # Remove cards played
        for color, n in observation["fireworks"].items():
            cindex = _COLOR_TO_INDEX[color]
            for rank in range(n):
                self.deck[cindex, rank] -= 1
        # Note: hints are CK but are treated later

    def _update_seen_hands(self, observation: Dict):
        # Update based on observed hands
        hands = observation["observed_hands"]
        offset = self.observation_from
        for hand in hands:
            # The player cannot see its own hand
            if offset == self.perspective_of or offset == self.observation_from:
                offset += 1
                continue
            for card in hand:
                color = _COLOR_TO_INDEX[card["color"]]
                rank = card["rank"]
                self.deck[color, rank] -= 1
            offset += 1
            offset %= self.players

    def update_with_action(self, action: Dict):
        """
        Update this belief with the specified action.
        It is assumed the action originates from the player 'perspective_of'.
        """
        if action["action_type"] == "PLAY" or action["action_type"] == "DISCARD":
            offset = action["card_index"]
            self.possible_ranks[offset:self.hand_size - 1, :] = self.possible_ranks[offset + 1:, :]
            self.possible_ranks[self.hand_size - 1, :] = 1
            self.possible_colors[offset:self.hand_size - 1, :] = self.possible_colors[offset + 1:, :]
            self.possible_colors[self.hand_size - 1, :] = 1

    def update_with_hint(self, action: Dict, hand: List[Dict]):
        """
        Update this belief with the specified hint.
        It is assumed the action target the player 'perspective_of'.
        """
        # Basically we update what's possible for ranks and colors for each card
        if action["action_type"] == "REVEAL_COLOR":
            color = action["color"]
            cindex = _COLOR_TO_INDEX[color]
            for offset, card in enumerate(hand):
                if card["color"] == color:
                    self.possible_colors[offset, :] = 0
                    self.possible_colors[offset, cindex] = 1
                else:
                    self.possible_colors[offset, cindex] = 0
        elif action["action_type"] == "REVEAL_RANK":
            rank = action["rank"]
            for offset, card in enumerate(hand):
                if card["rank"] == rank:
                    self.possible_ranks[offset, :] = 0
                    self.possible_ranks[offset, rank] = 1
                else:
                    self.possible_ranks[offset, rank] = 0

    def _update_hints(self, observation: Dict):
        # The information here is only used when computing probabality only on ranks or only on colors
        knowledge = observation["card_knowledge"]

        self.virtual_colors = np.zeros(5)
        self.virtual_ranks = np.zeros(5)

        if self.observation_from == self.perspective_of:
            # Observator hand knowledge
            obs_hand_kl = knowledge[0]
            self.hand = obs_hand_kl
        else:
            # Perspective of hand knowledge
            persp_hand_kl = knowledge[(self.perspective_of - self.observation_from) % self.players]
            self.hand = persp_hand_kl

            # Observator hand knowledge
            obs_hand_kl = knowledge[0]
            for card in obs_hand_kl:
                color = _COLOR_TO_INDEX[card["color"]]
                rank = card["rank"]
                if color is None and rank is None:
                    continue
                if color is None:
                    self.virtual_ranks[rank] += 1
                elif rank is None:
                    self.virtual_colors[color] += 1
                else:
                    self.deck[color, rank] -= 1

    def update(self, observation: Dict):
        """
        Update this belief according to the specified observations.

        Parameters
        -----------
        - **observation**: the observation dictionnary from the perspective of ```observations_from```.
        """
        self.deck = _FULL_DECK_.copy()

        # --------------------------------------
        # CK PART
        # --------------------------------------
        self._update_ck(observation)

        # --------------------------------------
        # Partial knowledge part
        # --------------------------------------
        self._update_seen_hands(observation)

        deck_size = np.sum(self.deck) - self.hand_size * (1 + (self.perspective_of != self.observation_from))
        deck_size = max(0, deck_size)
        real_size = observation['deck_size']
        assert deck_size == real_size, f"[Out of sync.] Invalid deck size:{deck_size} instead of {real_size}: {self.deck}"
        # --------------------------------------
        # Update based on hints
        # --------------------------------------
        # Must be done last
        self._update_hints(observation)

        assert np.sum(self.virtual_ranks) <= self.hand_size
        assert np.sum(self.virtual_colors) <= self.hand_size
        assert len(np.where(self.deck < 0)[0]) == 0, "[Counting error] a card was removed one time too many !"

    def _filtered_deck(self, offset: int) -> np.ndarray:
        """"Computes a deck of possibles card for the card in hand at offset."""
        cards = self.deck.copy()
        for c in range(5):
            cards[c, :] *= self.possible_colors[offset, c]
        for r in range(5):
            cards[:, r] *= self.possible_ranks[offset, r]
        return cards

    def _total_cards(self, offset: int, color=False, rank=False) -> int:
        """"Computes the number of possibles card for the card in hand at offset."""
        cards = self.deck.copy()

        for c in range(5):
            cards[c, :] *= self.possible_colors[offset, c]

        for r in range(5):
            cards[:, r] *= self.possible_ranks[offset, r]

        cards = np.sum(cards)
        if rank:
            cards -= np.sum(self.virtual_ranks * self.possible_ranks[offset])
        if color:
            cards -= np.sum(self.virtual_colors * self.possible_colors[offset])
        return cards

    def _prob_color(self, color: int, offset: int) -> float:
        total_cards = self._total_cards(offset, color=False)
        if total_cards == 0:
            return 0
        total_colors = np.sum(self._filtered_deck(offset)[color, :])  # - self.virtual_colors[color]
        return total_colors / total_cards

    def _prob_rank(self, rank: int,  offset: int) -> float:
        total_cards = self._total_cards(offset, rank=False)
        if total_cards == 0:
            return 0
        total_colors = np.sum(self._filtered_deck(offset)[:, rank])  # - self.virtual_ranks[rank]
        return total_colors / total_cards

    def _prob_card(self, rank: int, color: int, offset: int) -> float:
        total_cards = self._total_cards(offset, color=False, rank=False)
        if total_cards == 0:
            return 0
        total_in_deck = self.deck[color, rank]
        return total_in_deck / total_cards

    def _prob_card_knowing_color(self, rank: int, color: int, offset: int) -> float:
        p1 = self._prob_card(rank, color, offset)
        return p1

    def _prob_card_knowing_rank(self, rank: int, color: int, offset: int) -> float:
        p1 = self._prob_card(rank, color, offset)
        return p1

    def probability(self, offset: int, rank: int = -1, color: str = None) -> float:
        """
        Computes the asked probability.
        If only a ```color``` is specified, computes the probability of the card being of the specified ```color```.
        If only a ```rank``` is specified, computes the probability of the card being of the specified ```rank```.
        If no ```color``` nor ```rank``` is specified return ```1```.

        Parameters
        -----------
        - **offset**: the offset of the card in you hand that you want information about
        - **rank**: (*optional*) the rank you want information about
        - **color**: (*optional*) the color you want information about

        Return
        -----------
        A float in ```[0; 1]``` that specifies the probability of the specified card matching the parameters given.
        """
        if rank is None:
            rank = -1
        card_info = self.hand[offset]
        if rank == -1:
            if color is None:
                return 1
            return self._prob_color(_COLOR_TO_INDEX[color], offset)
        elif color is None:
            return self._prob_rank(rank, offset)
        else:
            r = card_info["rank"]
            c = card_info["color"]
            cindex = _COLOR_TO_INDEX[color]
            if r == rank and c == color:
                return 1
            elif c is None and rank == r:
                return self._prob_card_knowing_rank(rank, cindex, offset)
            elif c == color and r is None:
                return self._prob_card_knowing_color(rank, cindex, offset)
            elif c is None and r is None:
                return self._prob_card(rank, cindex, offset)
            else:
                return 0

    def probability_playable(self, offset: int, fireworks: Dict) -> float:
        """
        Return the probability that a card is playable on top of the fireworks.
        """
        probs = np.zeros(5)
        for color, n in fireworks.items():
            if n <= 4:
                probs[_COLOR_TO_INDEX[color]] = self.probability(offset, n, color)
        return np.sum(probs)

    def probability_useless(self, offset: int, fireworks: Dict) -> float:
        """
        Return the probability that a card is a card that has already been played in the fireworks.
        """
        p = 0
        for color, n in fireworks.items():
            if n == 5:
                p += self._prob_color(_COLOR_TO_INDEX[color], offset)
            for i in range(n):
                p += self.probability(offset, i, color)
        return p

    def most_informative_hint(self, actual_hand: List[Dict]) -> Tuple[Dict, float]:
        """
        Return the best hint possible and the number of bits of information that hints give to the other player.
        """
        # List possible hints
        ranks = []
        colors = []
        for card in actual_hand:
            r, c = card["rank"], card["color"]
            if r not in ranks:
                ranks.append(r)
            if c not in colors:
                colors.append(c)

        best_info = -1
        best_rank = None
        best_color = None
        # Compute hint score for ranks
        for rank in ranks:
            info = 0
            for offset, card in enumerate(actual_hand):
                r, c = card["rank"], card["color"]
                if r == rank:
                    p = self.probability(offset, r, c)
                    assert p > 0
                    info += -np.log(p)
                else:
                    p = self._prob_rank(rank, offset)
                    assert p < 1
                    info += -np.log(1 - p)
            if info > best_info:
                best_info = info
                best_rank = rank
        # Compute hint score for colors
        for color in colors:
            info = 0
            for offset, card in enumerate(actual_hand):
                r, c = card["rank"], card["color"]
                if c == color:
                    p = self.probability(offset, r, c)
                    assert p > 0
                    info += -np.log(p)
                else:
                    p = self._prob_color(_COLOR_TO_INDEX[color], offset)
                    assert p < 1
                    info += -np.log(1 - p)
            if info > best_info:
                best_info = info
                best_rank = None
                best_color = color

        best_info /= np.log(2)
        if best_rank is None:
            if best_color is None:
                raise Exception("No hint found at all !!")
            return {"action_type": "REVEAL_COLOR", "color": best_color}, best_info
        else:
            return {"action_type": "REVEAL_RANK", "rank": best_rank}, best_info
