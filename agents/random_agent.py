"""Random agent: selects uniformly from legal bids."""

import random
from environment.auction import AuctionState


class RandomAgent:
    """Picks a uniformly random legal bid."""

    def act(self, hand, auction: AuctionState) -> int:
        return random.choice(auction.valid_bids())
