"""Neural network bidding agent.

Architecture (shared weights between N and S):
  - Hand encoder: suit-aware (default) or flat MLP
  - Bid embedding: Embedding(vocab → embed_dim)
  - Auction encoder: LSTM or Transformer (embed_dim → hidden)
  - Combined: concat([hand_emb, auction_emb]) → 2*hidden
  - Policy head: MLP(2*hidden → NUM_BIDS), masked log-softmax
  - Value head:  MLP(2*hidden → 1)

Direction (0-3) is prepended to the auction sequence as a special token.
For the LSTM it is the first input; for the Transformer it acts as a CLS token
whose output is used as the sequence representation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.auction import NUM_BIDS, MAX_AUCTION_LEN


_DIR_TOKEN_OFFSET = NUM_BIDS
_VOCAB_SIZE       = NUM_BIDS + 4  # bids (0-37) + 4 direction tokens

# Cards are ordered suit-major: indices 0-12 = clubs, 13-25 = diamonds,
# 26-38 = hearts, 39-51 = spades (rank 0=2 through rank 12=A within each suit)
_N_SUITS  = 4
_N_RANKS  = 13


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """n_layers hidden layers of size `hidden`, then output layer."""
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SuitAwareHandEncoder(nn.Module):
    """
    Encode each suit independently with shared weights, then aggregate.

    The hand (52,) is reshaped to (4, 13) — one row per suit. A single
    Linear(13 → embed_dim) is applied to every suit, giving each suit its
    own embedding that captures length and honor structure. The four suit
    embeddings are concatenated and passed through an aggregation MLP to
    produce a hidden-dimensional hand embedding.

    Shared weights across suits give the right inductive bias: the same
    features (length, voids, top-honor presence) matter in every suit.
    The concatenation rather than mean-pooling preserves which suit is
    which, so the model can learn that spades and hearts differ from
    clubs and diamonds in terms of scoring.
    """

    def __init__(self, hidden: int, embed_dim: int, mlp_layers: int):
        super().__init__()
        self.suit_linear = nn.Linear(_N_RANKS, embed_dim)
        self.agg = _mlp(_N_SUITS * embed_dim, hidden, hidden, mlp_layers)

    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        # hand: (B, 52) → (B, 4, 13)
        suits = hand.view(hand.shape[0], _N_SUITS, _N_RANKS)
        # shared linear + ReLU across all suits: (B, 4, embed_dim)
        suit_embs = torch.relu(self.suit_linear(suits))
        # flatten suits: (B, 4 * embed_dim)
        flat = suit_embs.view(hand.shape[0], -1)
        return self.agg(flat)


class BiddingNet(nn.Module):
    """
    Shared network for all bidding seats.

    Args:
        hidden:          width of every hidden layer
        embed_dim:       bid-token embedding size; also per-suit embedding size
        mlp_layers:      hidden layers in hand encoder and output heads
        lstm_layers:     LSTM layers, or Transformer encoder layers
        hand_encoder:    "suit" | "mlp"
        auction_encoder: "lstm" (default) | "transformer"
        partner_pred:    if True, add a partner hand prediction head whose output
                         is encoded and appended to the policy state. This creates
                         an explicit incentive for bids to be informative about
                         the bidder's hand (trained with an auxiliary BCE loss).
    """

    def __init__(self, hidden: int = 128, embed_dim: int = 32,
                 mlp_layers: int = 1, lstm_layers: int = 1,
                 hand_encoder: str = "suit", auction_encoder: str = "lstm",
                 partner_pred: bool = False):
        super().__init__()
        self.hidden            = hidden
        self._auction_enc_type = auction_encoder
        self.partner_pred      = partner_pred

        if hand_encoder == "suit":
            self.hand_enc = SuitAwareHandEncoder(hidden, embed_dim, mlp_layers)
        else:
            self.hand_enc = _mlp(52, hidden, hidden, mlp_layers)

        self.bid_emb = nn.Embedding(_VOCAB_SIZE + 1, embed_dim, padding_idx=_VOCAB_SIZE)

        if auction_encoder == "transformer":
            # nhead: keep at least 8 dims per head
            n_heads = max(1, embed_dim // 8)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=max(64, embed_dim * 4),
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.transformer  = nn.TransformerEncoder(enc_layer, num_layers=lstm_layers)
            self.pos_emb      = nn.Embedding(MAX_AUCTION_LEN + 2, embed_dim)
            self.auction_proj = nn.Linear(embed_dim, hidden)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden,
                                num_layers=lstm_layers, batch_first=True)

        if partner_pred:
            # Predict partner's 52-card hand from the auction embedding.
            # The prediction is re-encoded through the shared hand encoder and
            # appended to the state, giving the policy an explicit belief about
            # what partner holds. Trained with auxiliary BCE loss.
            self.partner_pred_head = nn.Linear(hidden, 52)
            combined = hidden * 3   # own_hand + auction + partner_pred
        else:
            combined = hidden * 2   # own_hand + auction

        self.policy_head = _mlp(combined, hidden, NUM_BIDS, mlp_layers)
        self.value_head  = _mlp(combined, hidden, 1,        mlp_layers)

    def _encode_auction(self, auction_seq: torch.Tensor,
                        direction: torch.Tensor) -> torch.Tensor:
        """
        auction_seq: (B, MAX_AUCTION_LEN) int64, -1 = padding
        direction:   (B,) int64
        Returns:     (B, hidden)
        """
        dir_token = (_DIR_TOKEN_OFFSET + direction).unsqueeze(1)   # (B, 1)
        tokens    = auction_seq.clone()
        lengths   = (tokens != -1).sum(dim=1) + 1                  # +1 for dir token
        tokens[tokens == -1] = _VOCAB_SIZE
        tokens    = torch.cat([dir_token, tokens], dim=1)           # (B, 1+MAX_LEN)
        embedded  = self.bid_emb(tokens)                            # (B, 1+MAX_LEN, E)

        if self._auction_enc_type == "transformer":
            B, L, _ = embedded.shape
            pos      = torch.arange(L, device=embedded.device).unsqueeze(0)
            embedded = embedded + self.pos_emb(pos)
            # True = ignore (padding positions)
            pad_mask = torch.arange(L, device=embedded.device).unsqueeze(0) \
                       >= lengths.to(embedded.device).unsqueeze(1)
            out = self.transformer(embedded, src_key_padding_mask=pad_mask)
            return self.auction_proj(out[:, 0])    # direction token = CLS
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            return h_n[-1]

    def forward(self, hand: torch.Tensor, auction_seq: torch.Tensor,
                direction: torch.Tensor,
                valid_mask: torch.Tensor = None):
        """
        hand:        (B, 52) float32
        auction_seq: (B, MAX_AUCTION_LEN) int64
        direction:   (B,) int64
        valid_mask:  (B, NUM_BIDS) bool  — True for legal actions

        Returns:
            log_probs    (B, NUM_BIDS)
            values       (B, 1)
            partner_pred (B, 52) float32 sigmoid, or None if partner_pred=False
        """
        hand_emb    = self.hand_enc(hand)
        auction_emb = self._encode_auction(auction_seq, direction)

        if self.partner_pred:
            partner_logits = self.partner_pred_head(auction_emb)    # (B, 52) raw logits
            partner_prob   = torch.sigmoid(partner_logits)           # (B, 52) for hand encoder
            partner_emb    = self.hand_enc(partner_prob)             # (B, H)
            combined = torch.cat([hand_emb, auction_emb, partner_emb], dim=-1)
        else:
            partner_logits = None
            combined       = torch.cat([hand_emb, auction_emb], dim=-1)

        logits = self.policy_head(combined)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        log_probs = torch.log_softmax(logits, dim=-1)
        values    = self.value_head(combined)
        # Return raw partner logits (not sigmoid) so the caller can use
        # binary_cross_entropy_with_logits for numerical stability.
        return log_probs, values, partner_logits


class CentralizedCritic(nn.Module):
    """
    Centralized value function for CTDE (Centralized Training, Decentralized Execution).

    Sees *both* North and South hands plus the full auction — information that
    neither player has at the table. Used only during training to produce
    lower-variance advantage estimates. Discarded at inference time.

    Architecture:
      north_hand (52,) ──► SuitAwareEncoder ──► north_emb ──┐
      south_hand (52,) ──► SuitAwareEncoder ──► south_emb ──┼──► concat → MLP → V
      auction_seq      ──► Embedding → LSTM  ──► auction_emb ┘

    The hand encoder is shared between the two seats (same weights applied
    independently), matching the inductive bias of the policy network.
    """

    def __init__(self, hidden: int = 128, embed_dim: int = 32,
                 mlp_layers: int = 1, lstm_layers: int = 1,
                 auction_encoder: str = "lstm"):
        super().__init__()
        self._auction_enc_type = auction_encoder
        self.hand_enc = SuitAwareHandEncoder(hidden, embed_dim, mlp_layers)
        self.bid_emb  = nn.Embedding(_VOCAB_SIZE + 1, embed_dim, padding_idx=_VOCAB_SIZE)

        if auction_encoder == "transformer":
            n_heads   = max(1, embed_dim // 8)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=max(64, embed_dim * 4),
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.transformer  = nn.TransformerEncoder(enc_layer, num_layers=lstm_layers)
            self.pos_emb      = nn.Embedding(MAX_AUCTION_LEN + 1, embed_dim)
            self.auction_proj = nn.Linear(embed_dim, hidden)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden, num_layers=lstm_layers, batch_first=True)

        self.value_head = _mlp(hidden * 3, hidden, 1, mlp_layers)

    def _encode_auction(self, auction_seq: torch.Tensor) -> torch.Tensor:
        tokens   = auction_seq.clone()
        lengths  = (tokens != -1).sum(dim=1).clamp(min=1)
        tokens[tokens == -1] = _VOCAB_SIZE
        embedded = self.bid_emb(tokens)                             # (B, MAX_LEN, E)

        if self._auction_enc_type == "transformer":
            B, L, _ = embedded.shape
            pos      = torch.arange(L, device=embedded.device).unsqueeze(0)
            embedded = embedded + self.pos_emb(pos)
            pad_mask = torch.arange(L, device=embedded.device).unsqueeze(0) \
                       >= lengths.to(embedded.device).unsqueeze(1)
            out      = self.transformer(embedded, src_key_padding_mask=pad_mask)
            # Mean-pool over valid positions (no CLS token in the critic)
            valid    = ~pad_mask                                     # (B, L) True=valid
            pooled   = (out * valid.unsqueeze(-1)).sum(1) \
                       / lengths.float().to(out.device).unsqueeze(1)
            return self.auction_proj(pooled)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            return h_n[-1]

    def forward(self, north_hands: torch.Tensor, south_hands: torch.Tensor,
                auction_seqs: torch.Tensor) -> torch.Tensor:
        """
        north_hands:  (B, 52) float32
        south_hands:  (B, 52) float32
        auction_seqs: (B, MAX_AUCTION_LEN) int64  (-1 = padding)
        Returns:      (B, 1) value estimates
        """
        north_emb   = self.hand_enc(north_hands)
        south_emb   = self.hand_enc(south_hands)
        auction_emb = self._encode_auction(auction_seqs)
        combined    = torch.cat([north_emb, south_emb, auction_emb], dim=-1)
        return self.value_head(combined)


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

        log_probs, _, _ = self.net(hand_t, seq_t, dir_t, mask)
        return Categorical(logits=log_probs[0]).sample().item()


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
