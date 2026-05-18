"""Auction state machine for bridge bidding.

Bid encoding (38 actions):
  0        = Pass
  1..35    = 1C, 1D, 1H, 1S, 1NT, 2C, ..., 7NT  (7 levels × 5 denoms)
  36       = Double
  37       = Redouble

Denominations within a level: clubs=0, diamonds=1, hearts=2, spades=3, nt=4
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# --- Bid index helpers ---

PASS = 0
DOUBLE = 36
REDOUBLE = 37
NUM_BIDS = 38
MAX_AUCTION_LEN = 80  # generous upper bound; longest real auction is ~40 bids

# Denom order: clubs, diamonds, hearts, spades, NT  (ascending in bridge)
_DENOM_NAMES = ["C", "D", "H", "S", "NT"]


def contract_bid_idx(level: int, denom: int) -> int:
    """level 1..7, denom 0..4 (C,D,H,S,NT) → bid index 1..35"""
    return (level - 1) * 5 + denom + 1


def bid_to_level_denom(bid_idx: int):
    """bid index 1..35 → (level 1..7, denom 0..4)"""
    assert 1 <= bid_idx <= 35
    b = bid_idx - 1
    return b // 5 + 1, b % 5


def bid_name(bid_idx: int) -> str:
    if bid_idx == PASS:
        return "Pass"
    if bid_idx == DOUBLE:
        return "X"
    if bid_idx == REDOUBLE:
        return "XX"
    level, denom = bid_to_level_denom(bid_idx)
    return f"{level}{_DENOM_NAMES[denom]}"


@dataclass
class AuctionState:
    """
    Tracks the state of a bridge auction.

    Phase 1: E-W are forced to pass (uncontested auction).
    """
    # History of bid indices in order
    bids: list = field(default_factory=list)
    # Direction of each bid (0=N, 1=E, 2=S, 3=W)
    directions: list = field(default_factory=list)
    # Index of the dealer (first bidder); 0=N, 1=E, 2=S, 3=W
    dealer: int = 0
    # Whether E-W are forced to pass (Phase 1)
    uncontested: bool = True

    # --- derived state, updated incrementally ---
    _last_contract_bid: int = field(default=0, repr=False)  # highest bid so far (1..35), 0=none
    _last_contract_direction: int = field(default=-1, repr=False)
    _doubled: bool = field(default=False, repr=False)
    _redoubled: bool = field(default=False, repr=False)
    _consecutive_passes: int = field(default=0, repr=False)

    def current_player(self) -> int:
        """0=N, 1=E, 2=S, 3=W"""
        return (self.dealer + len(self.bids)) % 4

    def is_complete(self) -> bool:
        """Auction ends after 3 passes following the first real bid, or 4 passes at the start."""
        if self._last_contract_bid == 0:
            return self._consecutive_passes >= 4
        return self._consecutive_passes >= 3

    def valid_bids(self) -> list:
        """Return list of legal bid indices for current player."""
        player = self.current_player()
        # E-W forced to pass in uncontested mode
        if self.uncontested and player in (1, 3):
            return [PASS]

        valid = [PASS]

        if self._last_contract_bid > 0:
            # Can double if last non-pass was by an opponent and not already doubled
            if not self._doubled and not self._redoubled:
                if self._last_contract_direction % 2 != player % 2:
                    valid.append(DOUBLE)
            # Can redouble if doubled and last doubler was opponent
            if self._doubled and not self._redoubled:
                if self._last_contract_direction % 2 != player % 2:
                    valid.append(REDOUBLE)

        # Higher contract bids
        for bid_idx in range(self._last_contract_bid + 1, 36):
            valid.append(bid_idx)

        return valid

    def apply(self, bid_idx: int):
        """Apply a bid and update internal state. Raises ValueError if illegal."""
        if bid_idx not in self.valid_bids():
            raise ValueError(f"Illegal bid {bid_name(bid_idx)} by {self.current_player()}")

        player = self.current_player()
        self.bids.append(bid_idx)
        self.directions.append(player)

        if bid_idx == PASS:
            self._consecutive_passes += 1
        elif bid_idx == DOUBLE:
            self._doubled = True
            self._consecutive_passes = 0
            self._last_contract_direction = player  # doubler is now "last actor"
        elif bid_idx == REDOUBLE:
            self._redoubled = True
            self._consecutive_passes = 0
            self._last_contract_direction = player
        else:
            self._last_contract_bid = bid_idx
            self._last_contract_direction = player
            self._doubled = False
            self._redoubled = False
            self._consecutive_passes = 0

    def declarer(self) -> Optional[int]:
        """
        The declarer is the first player on the declaring side who bid the
        contract's denomination. Returns direction (0-3) or None if passed out.
        """
        if self._last_contract_bid == 0:
            return None
        _, contract_denom = bid_to_level_denom(self._last_contract_bid)
        declaring_side = self._last_contract_direction % 2  # 0=NS, 1=EW
        for bid_idx, direction in zip(self.bids, self.directions):
            if direction % 2 == declaring_side and 1 <= bid_idx <= 35:
                _, d = bid_to_level_denom(bid_idx)
                if d == contract_denom:
                    return direction
        return self._last_contract_direction

    def final_contract(self):
        """
        Returns (level, denom, declarer_direction, doubled, redoubled) or None if passed out.
        denom: 0=C, 1=D, 2=H, 3=S, 4=NT
        """
        if self._last_contract_bid == 0:
            return None
        level, denom = bid_to_level_denom(self._last_contract_bid)
        return level, denom, self.declarer(), self._doubled, self._redoubled

    def to_sequence(self) -> np.ndarray:
        """Return bid history as a numpy int array (for NN input)."""
        return np.array(self.bids, dtype=np.int64)

    def __str__(self) -> str:
        parts = [bid_name(b) for b in self.bids]
        return " - ".join(parts) if parts else "(no bids)"
