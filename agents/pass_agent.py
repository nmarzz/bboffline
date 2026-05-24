"""Always-pass agent: baseline for the minimum achievable score."""


class PassAgent:
    """Always passes, regardless of hand or auction state."""

    def act(self, hand, auction) -> int:
        return 0  # PASS
