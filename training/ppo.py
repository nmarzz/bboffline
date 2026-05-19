"""PPO update for the bridge bidding agent.

Uses parameter-shared weights between North and South (cooperative team).
Supports mini-batch updates over a collected rollout buffer.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import List
import numpy as np

from environment.auction import NUM_BIDS, MAX_AUCTION_LEN
from agents.nn_agent import BiddingNet


@dataclass
class Transition:
    """One bidding step from a rollout."""
    hand: np.ndarray          # (52,)
    auction_seq: np.ndarray   # (MAX_AUCTION_LEN,)
    direction: int
    valid_mask: np.ndarray    # (NUM_BIDS,) bool
    action: int
    log_prob: float
    value: float
    reward: float             # only non-zero at terminal step
    done: bool


class RolloutBuffer:
    def __init__(self):
        self.transitions: List[Transition] = []

    def push(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)


def compute_returns(transitions: List[Transition], gamma: float = 1.0) -> List[float]:
    """
    Compute discounted returns.  gamma=1 is appropriate here because bridge
    episodes are short and only have a single terminal reward.
    """
    returns = [0.0] * len(transitions)
    running = 0.0
    for i in reversed(range(len(transitions))):
        running = transitions[i].reward + gamma * running * (1 - float(transitions[i].done))
        returns[i] = running
    return returns


class PPOUpdater:
    def __init__(
        self,
        net: BiddingNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        device: str = "cpu",
    ):
        self.net = net
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.device = device
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self._lr = lr

    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._lr = lr

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on collected transitions. Returns loss stats."""
        transitions = buffer.transitions
        returns = compute_returns(transitions)

        # Pack into tensors
        hands    = torch.tensor(np.stack([t.hand for t in transitions]),
                                dtype=torch.float32, device=self.device)
        auctions = torch.tensor(np.stack([t.auction_seq for t in transitions]),
                                dtype=torch.int64, device=self.device)
        dirs     = torch.tensor([t.direction for t in transitions],
                                dtype=torch.int64, device=self.device)
        masks    = torch.tensor(np.stack([t.valid_mask for t in transitions]),
                                dtype=torch.bool, device=self.device)
        actions  = torch.tensor([t.action for t in transitions],
                                dtype=torch.int64, device=self.device)
        old_lps  = torch.tensor([t.log_prob for t in transitions],
                                dtype=torch.float32, device=self.device)
        old_vals = torch.tensor([t.value for t in transitions],
                                dtype=torch.float32, device=self.device)
        rets     = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs     = rets - old_vals
        advs     = (advs - advs.mean()) / (advs.std() + 1e-8)

        stats = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            log_probs, values = self.net(hands, auctions, dirs, masks)
            values = values.squeeze(-1)

            # Gather log probs for taken actions
            new_lps = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_lps - old_lps)

            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, rets)

            # Entropy bonus: 0*log(0) → nan; replace with 0 (limit is 0)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())

        return {k: float(np.mean(v)) for k, v in stats.items()}
