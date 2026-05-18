"""Deal generation and hand encoding."""

import numpy as np
from dataclasses import dataclass
from endplay.types import Deal as EndplayDeal, Player, Denom, Rank, Card
from endplay.dealer import generate_deals


PLAYERS = [Player.north, Player.east, Player.south, Player.west]
SUITS = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs]

# card_index = suit_idx * 13 + rank_idx  (rank_idx: 0=2, 12=A)

_ALL_CARDS = [Card(suit=d, rank=r) for d in Denom.suits() for r in Rank]

# Persistent generator: avoids re-initialising the shuffle machinery each call.
# generate_deals() defaults to produce=40; set a large number so it never exhausts.
_DEAL_GEN = generate_deals(produce=10_000_000)


def _rank_idx(rank) -> int:
    return int(rank).bit_length() - 3


def hand_to_vector(hand) -> np.ndarray:
    """Encode a Hand as a 52-dim binary numpy array."""
    vec = np.zeros(52, dtype=np.float32)
    for card in hand:
        suit_idx = int(card.suit)  # 0=spades, 1=hearts, 2=diamonds, 3=clubs
        rank_idx = _rank_idx(card.rank)
        vec[suit_idx * 13 + rank_idx] = 1.0
    return vec


@dataclass
class BridgeDeal:
    """A bridge deal with encoded hands and the underlying endplay Deal."""
    endplay_deal: EndplayDeal
    # hands[i] for i in 0..3 → North, East, South, West; shape (52,)
    hands: np.ndarray  # shape (4, 52)
    vulnerability: str  # "none", "ns", "ew", "both"


def random_deal(vulnerability: str = "none") -> BridgeDeal:
    """Generate a single random deal."""
    ep_deal = next(_DEAL_GEN)
    hands = np.stack([hand_to_vector(ep_deal[p]) for p in PLAYERS])
    return BridgeDeal(endplay_deal=ep_deal, hands=hands, vulnerability=vulnerability)


def resample_ew(ep_deal: EndplayDeal, k: int, rng=None) -> list:
    """
    Return k new EndplayDeals with the same N-S cards but independently
    re-shuffled E-W hands.  Used for counterfactual reward estimation.
    """
    if rng is None:
        rng = np.random.default_rng()

    ns_set = set(ep_deal[Player.north]) | set(ep_deal[Player.south])
    remaining = [c for c in _ALL_CARDS if c not in ns_set]  # exactly 26 cards

    result = []
    for _ in range(k):
        perm = rng.permutation(26)
        nd = EndplayDeal()
        nd[Player.north].extend(ep_deal[Player.north])
        nd[Player.south].extend(ep_deal[Player.south])
        nd[Player.east].extend(remaining[i] for i in perm[:13])
        nd[Player.west].extend(remaining[i] for i in perm[13:])
        result.append(nd)
    return result
