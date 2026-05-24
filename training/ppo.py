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
    ns_hands: np.ndarray      # (2, 52) — index 0=North, 1=South, always
    auction_seq: np.ndarray   # (MAX_AUCTION_LEN,)
    direction: int            # 0=North, 2=South
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


def compute_gae(
    transitions: List[Transition],
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple:
    """
    Generalised Advantage Estimation (Schulman et al. 2015).

    Returns (returns, advantages) as plain lists.

    With gamma=1 (appropriate for bridge — short episodes, single terminal
    reward) GAE reduces to an exponentially weighted sum of TD errors, trading
    off between the high-variance Monte-Carlo return (lam=1) and the
    low-variance but biased 1-step TD (lam=0).  lam≈0.95 is the standard
    default and almost always improves over raw returns.

    Episode boundaries (done=True) zero out the carry so advantages never
    bleed across episodes even when the buffer contains many episodes.
    """
    T = len(transitions)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0

    for i in reversed(range(T)):
        t        = transitions[i]
        # Bootstrap from next transition only if within the same episode
        next_val = transitions[i + 1].value if (i + 1 < T and not t.done) else 0.0
        delta    = t.reward + gamma * next_val - t.value
        last_adv = delta + gamma * lam * (1.0 - float(t.done)) * last_adv
        advantages[i] = last_adv

    values  = np.array([t.value for t in transitions], dtype=np.float32)
    returns = advantages + values
    return returns.tolist(), advantages.tolist()


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
        critic=None,
        critic_lr: float = 3e-4,
        gae_lambda: float = 0.95,
        mini_batch_size: int = 256,
    ):
        self.net             = net
        self.clip_eps        = clip_eps
        self.entropy_coef    = entropy_coef
        self.value_coef      = 0.0 if critic is not None else value_coef
        self.n_epochs        = n_epochs
        self.device          = device
        self.critic          = critic
        self.gae_lambda      = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.optimizer       = torch.optim.Adam(net.parameters(), lr=lr)
        self._lr             = lr
        self.critic_optimizer = (
            torch.optim.Adam(critic.parameters(), lr=critic_lr)
            if critic is not None else None
        )

    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        if self.critic_optimizer is not None:
            for pg in self.critic_optimizer.param_groups:
                pg["lr"] = lr
        self._lr = lr

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on collected transitions. Returns loss stats."""
        transitions = buffer.transitions
        T           = len(transitions)

        # --- GAE returns and advantages (computed once, fixed across epochs) ---
        returns, advantages = compute_gae(transitions, lam=self.gae_lambda)

        # --- Pack all tensors up-front ---
        ns_arr    = np.stack([t.ns_hands for t in transitions])        # (T, 2, 52)
        slots     = np.array([0 if t.direction == 0 else 1 for t in transitions])
        act_hands = ns_arr[np.arange(T), slots]                        # (T, 52)

        hands    = torch.tensor(act_hands, dtype=torch.float32, device=self.device)
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
        rets     = torch.tensor(returns,    dtype=torch.float32, device=self.device)
        advs     = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advs     = (advs - advs.mean()) / (advs.std() + 1e-8)

        if self.critic is not None:
            north_hands = torch.tensor(ns_arr[:, 0], dtype=torch.float32, device=self.device)
            south_hands = torch.tensor(ns_arr[:, 1], dtype=torch.float32, device=self.device)

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "critic_loss": []}

        for _ in range(self.n_epochs):
            perm = torch.randperm(T, device=self.device)

            for start in range(0, T, self.mini_batch_size):
                mb = perm[start : start + self.mini_batch_size]

                log_probs, values = self.net(
                    hands[mb], auctions[mb], dirs[mb], masks[mb]
                )
                values = values.squeeze(-1)

                new_lps = log_probs.gather(1, actions[mb].unsqueeze(1)).squeeze(1)
                ratio   = torch.exp(new_lps - old_lps[mb])

                mb_advs = advs[mb]
                surr1   = ratio * mb_advs
                surr2   = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, rets[mb]) if self.value_coef > 0 \
                             else torch.tensor(0.0, device=self.device)

                probs   = torch.exp(log_probs)
                entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                # --- Centralized critic update ---
                critic_loss_val = 0.0
                if self.critic is not None:
                    critic_vals = self.critic(
                        north_hands[mb], south_hands[mb], auctions[mb]
                    ).squeeze(-1)
                    critic_loss = nn.functional.mse_loss(critic_vals, rets[mb])
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
                    critic_loss_val = critic_loss.item()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["critic_loss"].append(critic_loss_val)

        return {k: float(np.mean(v)) for k, v in stats.items()}
