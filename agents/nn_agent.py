"""Neural network bidding agent.

Architecture (shared weights between N and S):
  - Hand encoder: MLP(52 → hidden, n_layers deep)
  - Bid embedding: Embedding(vocab → embed_dim)
  - Auction encoder: LSTM(embed_dim → hidden, n_lstm_layers)
  - Combined: concat([hand_emb, lstm_final]) → 2*hidden
  - Policy head: MLP(2*hidden → NUM_BIDS), masked log-softmax
  - Value head:  MLP(2*hidden → 1)

Direction (0-3) is prepended to the auction sequence as a special token so the
shared-weight agent knows which seat it is bidding from.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.auction import NUM_BIDS, MAX_AUCTION_LEN


_DIR_TOKEN_OFFSET = NUM_BIDS
_VOCAB_SIZE       = NUM_BIDS + 4  # bids (0-37) + 4 direction tokens


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """n_layers hidden layers of size `hidden`, then output layer."""
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class BiddingNet(nn.Module):
    """
    Shared network for all bidding seats.

    Args:
        hidden:      width of every hidden layer
        embed_dim:   bid-token embedding size
        mlp_layers:  number of hidden layers in hand encoder and output heads
        lstm_layers: number of stacked LSTM layers for auction encoding
    """

    def __init__(self, hidden: int = 128, embed_dim: int = 32,
                 mlp_layers: int = 1, lstm_layers: int = 1):
        super().__init__()
        self.hidden = hidden

        # Hand encoder: MLP with mlp_layers hidden layers
        self.hand_enc = _mlp(52, hidden, hidden, mlp_layers)

        # Auction encoder
        self.bid_emb = nn.Embedding(_VOCAB_SIZE + 1, embed_dim,
                                    padding_idx=_VOCAB_SIZE)
        self.lstm = nn.LSTM(embed_dim, hidden,
                            num_layers=lstm_layers, batch_first=True)

        # Output heads: one hidden layer each
        combined = hidden * 2
        self.policy_head = _mlp(combined, hidden, NUM_BIDS, mlp_layers)
        self.value_head  = _mlp(combined, hidden, 1,        mlp_layers)

    def _encode_auction(self, auction_seq: torch.Tensor,
                        direction: torch.Tensor) -> torch.Tensor:
        """
        auction_seq: (B, MAX_AUCTION_LEN) int64, -1 = padding
        direction:   (B,) int64
        Returns:     (B, hidden) — final LSTM hidden state
        """
        dir_token = (_DIR_TOKEN_OFFSET + direction).unsqueeze(1)  # (B, 1)

        tokens = auction_seq.clone()
        lengths = (tokens != -1).sum(dim=1) + 1   # +1 for direction token
        lengths = lengths.cpu()

        tokens[tokens == -1] = _VOCAB_SIZE         # padding_idx → zero emb
        tokens = torch.cat([dir_token, tokens], dim=1)

        embedded = self.bid_emb(tokens)            # (B, 1+MAX_LEN, embed_dim)
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]                             # top layer, (B, hidden)

    def forward(self, hand: torch.Tensor, auction_seq: torch.Tensor,
                direction: torch.Tensor,
                valid_mask: torch.Tensor = None):
        """
        hand:        (B, 52) float32
        auction_seq: (B, MAX_AUCTION_LEN) int64
        direction:   (B,) int64
        valid_mask:  (B, NUM_BIDS) bool  — True for legal actions

        Returns: log_probs (B, NUM_BIDS),  values (B, 1)
        """
        hand_emb    = self.hand_enc(hand)
        auction_emb = self._encode_auction(auction_seq, direction)
        combined    = torch.cat([hand_emb, auction_emb], dim=-1)

        logits = self.policy_head(combined)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        log_probs = torch.log_softmax(logits, dim=-1)
        values    = self.value_head(combined)
        return log_probs, values


class NNAgent:
    """Wraps BiddingNet for single-step inference (used by analysis tools)."""

    def __init__(self, net: BiddingNet, device: str = "cpu"):
        self.net    = net
        self.device = device

    @torch.no_grad()
    def act(self, hand: np.ndarray, auction_seq: np.ndarray,
            direction: int, valid_bids: list) -> int:
        hand_t = torch.tensor(hand,       dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        seq_t  = torch.tensor(auction_seq, dtype=torch.int64,
                              device=self.device).unsqueeze(0)
        dir_t  = torch.tensor([direction], dtype=torch.int64,
                              device=self.device)
        mask   = torch.zeros(1, NUM_BIDS, dtype=torch.bool, device=self.device)
        for b in valid_bids:
            mask[0, b] = True

        log_probs, _ = self.net(hand_t, seq_t, dir_t, mask)
        return Categorical(logits=log_probs[0]).sample().item()


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
