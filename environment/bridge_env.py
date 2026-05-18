"""Gymnasium-compatible bridge bidding environment.

Observation dict:
  "hand"    : float32 array of shape (52,) — current player's cards
  "auction" : int64 array of shape (MAX_AUCTION_LEN,) — bid history,
              padded with -1 for unused positions
  "player"  : int scalar — current player direction (0=N,1=E,2=S,3=W)

Action: integer in [0, 37] (see auction.py for encoding)

Reward: IMP reward delivered at the terminal step; 0 otherwise.

The environment runs a full deal per episode.  In uncontested mode (phase 1)
East and West are forced to pass, so only North and South receive non-trivial
observations.  Callers must step through all 4 seats in turn.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.deal import random_deal
from environment.auction import AuctionState, NUM_BIDS, PASS, MAX_AUCTION_LEN
from environment.scoring import solve_deal, imp_reward


class BridgeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, vulnerability: str = "none", uncontested: bool = True,
                 render_mode=None):
        super().__init__()
        self.vulnerability = vulnerability
        self.uncontested = uncontested
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "hand":    spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
            "auction": spaces.Box(-1, NUM_BIDS - 1,
                                  shape=(MAX_AUCTION_LEN,), dtype=np.int64),
            "player":  spaces.Discrete(4),
        })
        self.action_space = spaces.Discrete(NUM_BIDS)

        self._deal = None
        self._table = None
        self._auction = None

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._deal = random_deal(self.vulnerability)
        self._table = solve_deal(self._deal)
        self._auction = AuctionState(uncontested=self.uncontested)
        obs = self._obs()
        return obs, {}

    def step(self, action: int):
        assert self._auction is not None, "call reset() first"

        valid = self._auction.valid_bids()
        if action not in valid:
            # Illegal move: force a pass and penalise
            action = PASS

        self._auction.apply(action)

        done = self._auction.is_complete()
        reward = 0.0
        if done:
            reward = imp_reward(self._table, self._auction, self.vulnerability)

        obs = self._obs()
        return obs, reward, done, False, {}

    def valid_actions(self):
        """Return list of legal action indices for the current player."""
        return self._auction.valid_bids()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self):
        player = self._auction.current_player()
        hand = self._deal.hands[player]  # (52,) float32

        seq = self._auction.to_sequence()
        padded = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
        padded[: len(seq)] = seq

        return {"hand": hand, "auction": padded, "player": np.int64(player)}

    def render(self):
        if self.render_mode != "ansi":
            return
        from environment.auction import bid_name
        print("Auction:", str(self._auction))
        if self._auction.is_complete():
            cf = self._auction.final_contract()
            if cf:
                level, denom, declarer, dbl, rdbl = cf
                suffix = "XX" if rdbl else ("X" if dbl else "")
                denom_names = ["C", "D", "H", "S", "NT"]
                dirs = ["N", "E", "S", "W"]
                print(f"Final contract: {level}{denom_names[denom]}{suffix} by {dirs[declarer]}")
            else:
                print("Passed out")
