"""Training loop for the bridge bidding agent.

Usage:
    python -m training.train [options]

Key options:
    --episodes        Total auction episodes to train on (default 200_000)
    --batch-episodes  Auctions per PPO update batch (default 512)
    --ew-samples      EW completions used for counterfactual reward (default 10)
    --dataset         Path to pre-computed dataset dir (strongly recommended)
    --hidden          Hidden width of every MLP/LSTM layer (default 128)
    --embed-dim       Bid token embedding size (default 32)
    --mlp-layers      Hidden layers in hand-encoder and output heads (default 1)
    --lstm-layers     Stacked LSTM layers for auction encoding (default 1)

Logs a CSV of training metrics to logs/train_metrics.csv.
Saves checkpoints to checkpoints/net_<step>.pt every 10k episodes.
"""

import argparse
import os
import csv
import time
import sys
import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.deal import random_deal
from environment.auction import AuctionState, NUM_BIDS, MAX_AUCTION_LEN
from environment.scoring import batch_expected_imp_rewards, precomputed_imp_rewards
from environment.dataset import BridgeDataset
from agents.nn_agent import BiddingNet, NNAgent, count_params
from training.ppo import PPOUpdater, RolloutBuffer, Transition

_DENOM_NAMES = ["C", "D", "H", "S", "NT"]
_DIR_NAMES   = ["N", "E", "S", "W"]


# ---------------------------------------------------------------------------
# Vectorized batch collection (dataset path — GPU friendly)
# ---------------------------------------------------------------------------

def collect_batch_vectorized(
    net: BiddingNet,
    ns_hands: np.ndarray,       # (N, 2, 52) float32
    dds_tables: np.ndarray,     # (N, K, 5, 4) uint8
    vulnerability: str,
    device: str,
    ew_samples: int,
    strain_bonus: float = 0.0,
    reward_mode: str = "expected_score",
) -> tuple:
    """
    Run N auctions in lockstep: at every step each active auction takes one
    action.  All N/S decisions in a given step are batched into a single
    forward pass, giving full GPU utilisation.

    Returns (flat_transitions, exp_imps, exp_pars).
    """
    N = ns_hands.shape[0]
    auctions  = [AuctionState(uncontested=True) for _ in range(N)]
    all_trans = [[] for _ in range(N)]
    active    = np.ones(N, dtype=bool)

    while active.any():
        active_idx = np.where(active)[0]
        players    = np.array([auctions[i].current_player() for i in active_idx])

        # ---- E/W: forced pass, no NN call ----
        ew_local = np.where(players % 2 == 1)[0]   # local indices within active_idx
        for li in ew_local:
            idx = active_idx[li]
            auctions[idx].apply(0)          # PASS
            if auctions[idx].is_complete():
                active[idx] = False

        # ---- N/S: build batch, single forward pass ----
        ns_local = np.where(players % 2 == 0)[0]
        if ns_local.size == 0:
            continue

        ns_idx = active_idx[ns_local]       # global episode indices

        # Pre-allocate batch arrays (avoids per-item appends)
        B = ns_idx.size
        batch_hands  = np.empty((B, 52),              dtype=np.float32)
        batch_seqs   = np.full((B, MAX_AUCTION_LEN),  -1, dtype=np.int64)
        batch_dirs   = np.empty(B,                    dtype=np.int64)
        batch_masks  = np.zeros((B, NUM_BIDS),        dtype=bool)

        for k, idx in enumerate(ns_idx):
            player = auctions[idx].current_player()
            # hands array: index 0 = North (player 0), index 1 = South (player 2)
            hand_slot   = 0 if player == 0 else 1
            batch_hands[k]  = ns_hands[idx, hand_slot]
            batch_dirs[k]   = player

            seq = auctions[idx].to_sequence()
            batch_seqs[k, :len(seq)] = seq

            for b in auctions[idx].valid_bids():
                batch_masks[k, b] = True

        hands_t  = torch.from_numpy(batch_hands).to(device)
        seqs_t   = torch.from_numpy(batch_seqs).to(device)
        dirs_t   = torch.from_numpy(batch_dirs).to(device)
        masks_t  = torch.from_numpy(batch_masks).to(device)

        with torch.no_grad():
            log_probs_t, vals_t = net(hands_t, seqs_t, dirs_t, masks_t)

        actions_t = Categorical(logits=log_probs_t).sample()  # (B,)

        for k, idx in enumerate(ns_idx):
            action   = actions_t[k].item()
            log_prob = log_probs_t[k, action].item()
            value    = vals_t[k, 0].item()
            player   = int(batch_dirs[k])

            auctions[idx].apply(action)

            all_trans[idx].append(Transition(
                hand        = batch_hands[k],
                auction_seq = batch_seqs[k].copy(),
                direction   = player,
                valid_mask  = batch_masks[k].copy(),
                action      = action,
                log_prob    = log_prob,
                value       = value,
                reward      = 0.0,
                done        = auctions[idx].is_complete(),
            ))

            if auctions[idx].is_complete():
                active[idx] = False

    # Ensure terminal flag is set on each episode's last transition
    for trans in all_trans:
        if trans:
            trans[-1].done = True

    # Compute rewards from pre-computed tables (pure numpy, no DDS)
    exp_imps, exp_pars = precomputed_imp_rewards(
        dds_tables, auctions, vulnerability, k=ew_samples,
        strain_bonus=strain_bonus, reward_mode=reward_mode,
    )
    for trans, imp in zip(all_trans, exp_imps):
        if trans:
            trans[-1].reward = imp

    flat_trans = [t for trans in all_trans for t in trans]
    return flat_trans, list(exp_imps), list(exp_pars)


# ---------------------------------------------------------------------------
# Sequential collection (live DDS path — no pre-computed dataset needed)
# ---------------------------------------------------------------------------

def run_auction(net_agent: NNAgent, deal, device: str):
    """
    Run one uncontested auction using the NN for N and S.
    Returns (transitions, auction_state) — no DDS is called here.
    """
    auction = AuctionState(uncontested=True)
    transitions = []

    while not auction.is_complete():
        player = auction.current_player()
        valid  = auction.valid_bids()

        seq    = auction.to_sequence()
        padded = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
        padded[: len(seq)] = seq

        hand = deal.hands[player]

        if player in (1, 3):  # E/W always pass
            action, log_prob, value = 0, 0.0, 0.0
        else:
            hand_t = torch.tensor(hand,   dtype=torch.float32, device=device).unsqueeze(0)
            seq_t  = torch.tensor(padded, dtype=torch.int64,   device=device).unsqueeze(0)
            dir_t  = torch.tensor([player], dtype=torch.int64, device=device)
            mask_t = torch.zeros(1, NUM_BIDS, dtype=torch.bool, device=device)
            for b in valid:
                mask_t[0, b] = True

            with torch.no_grad():
                log_probs_t, val_t = net_agent.net(hand_t, seq_t, dir_t, mask_t)

            action   = Categorical(logits=log_probs_t[0]).sample().item()
            log_prob = log_probs_t[0, action].item()
            value    = val_t[0, 0].item()

        valid_mask = np.zeros(NUM_BIDS, dtype=bool)
        for b in valid:
            valid_mask[b] = True

        auction.apply(action)

        if player in (0, 2):
            transitions.append(Transition(
                hand        = hand,
                auction_seq = padded,
                direction   = player,
                valid_mask  = valid_mask,
                action      = action,
                log_prob    = log_prob,
                value       = value,
                reward      = 0.0,
                done        = auction.is_complete(),
            ))

    if transitions:
        transitions[-1].done = True

    return transitions, auction


def collect_batch_sequential(
    net_agent: NNAgent,
    n_episodes: int,
    vulnerability: str,
    device: str,
    ew_samples: int,
    rng,
    strain_bonus: float = 0.0,
    reward_mode: str = "expected_score",
) -> tuple:
    """Sequential rollout using live DDS calls. Slower but needs no dataset."""
    deals     = []
    all_trans = []
    auctions  = []

    for _ in range(n_episodes):
        deal = random_deal(vulnerability)
        trans, auction = run_auction(net_agent, deal, device)
        deals.append(deal)
        all_trans.append(trans)
        auctions.append(auction)

    exp_imps, exp_pars = batch_expected_imp_rewards(
        [d.endplay_deal for d in deals],
        auctions, vulnerability, k=ew_samples, rng=rng,
        strain_bonus=strain_bonus, reward_mode=reward_mode,
    )

    for trans, imp in zip(all_trans, exp_imps):
        if trans:
            trans[-1].reward = imp

    flat_trans = [t for trans in all_trans for t in trans]
    return flat_trans, list(exp_imps), list(exp_pars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def contract_str(auction: AuctionState) -> str:
    cf = auction.final_contract()
    if cf:
        level, denom_idx, declarer, dbl, rdbl = cf
        suffix = "XX" if rdbl else ("X" if dbl else "")
        return f"{level}{_DENOM_NAMES[denom_idx]}{suffix}{_DIR_NAMES[declarer]}"
    return "Pass"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = args.device
    rng    = np.random.default_rng(args.seed)

    # ---- Dataset ----
    dataset = None
    if args.dataset:
        dataset = BridgeDataset(args.dataset)
        print(f"Dataset: {dataset.n_deals:,} deals  K={dataset.ew_samples}  "
              f"vul={dataset.vulnerability}  ({args.dataset})")

    # ---- Network ----
    net = BiddingNet(
        hidden      = args.hidden,
        embed_dim   = args.embed_dim,
        mlp_layers  = args.mlp_layers,
        lstm_layers = args.lstm_layers,
    ).to(device)

    agent   = NNAgent(net, device=device)
    updater = PPOUpdater(net, lr=args.lr, entropy_coef=args.entropy_coef, device=device)
    buffer  = RolloutBuffer()

    print(f"BiddingNet  hidden={args.hidden}  embed={args.embed_dim}  "
          f"mlp_layers={args.mlp_layers}  lstm_layers={args.lstm_layers}")
    print(f"Parameters: {count_params(net):,}")
    print(f"Reward mode: {args.reward_mode}")
    print(f"EW samples per episode: {args.ew_samples}")
    print(f"Entropy coef: {args.entropy_coef} → {args.entropy_final} (linear anneal)")
    if dataset:
        print("Using vectorized dataset rollouts (GPU-efficient)")
    else:
        print("Using sequential live-DDS rollouts")

    # ---- Metrics log ----
    metrics_path = "logs/train_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "mean_imp", "mean_par",
                                 "policy_loss", "value_loss", "entropy",
                                 "elapsed_s"])

    episode     = 0
    window_imps = []
    window_pars = []
    t0          = time.time()

    while episode < args.episodes:
        # ---- Collect rollout ----
        if dataset is not None:
            indices             = dataset.sample_indices(args.batch_episodes, rng)
            ns_hands, dds_tbls  = dataset.get(indices)
            flat_trans, exp_imps, exp_pars = collect_batch_vectorized(
                net, ns_hands, dds_tbls,
                dataset.vulnerability, device, args.ew_samples,
                strain_bonus=args.strain_bonus, reward_mode=args.reward_mode,
            )
        else:
            flat_trans, exp_imps, exp_pars = collect_batch_sequential(
                agent, args.batch_episodes, args.vulnerability,
                device, args.ew_samples, rng,
                strain_bonus=args.strain_bonus, reward_mode=args.reward_mode,
            )

        # ---- PPO update ----
        frac = episode / args.episodes
        updater.set_lr(args.lr * (1.0 - frac * 2 / 3))
        updater.set_entropy_coef(
            args.entropy_coef + (args.entropy_final - args.entropy_coef) * frac
        )

        for t in flat_trans:
            buffer.push(t)
        loss_stats = updater.update(buffer)
        buffer.clear()

        episode     += args.batch_episodes
        window_imps += exp_imps
        window_pars += exp_pars

        # ---- Logging ----
        if len(window_imps) >= 500 or episode >= args.episodes:
            mean_imp = float(np.mean(window_imps))
            mean_par = float(np.mean(window_pars))
            elapsed  = time.time() - t0
            print(f"ep={episode:7d}  mean_IMP={mean_imp:+.3f}  "
                  f"mean_par={mean_par:.1f}  "
                  f"policy_loss={loss_stats['policy_loss']:.4f}  "
                  f"entropy={loss_stats['entropy']:.3f}  "
                  f"elapsed={elapsed:.0f}s")
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([episode, mean_imp, mean_par,
                                        loss_stats["policy_loss"],
                                        loss_stats["value_loss"],
                                        loss_stats["entropy"],
                                        f"{elapsed:.1f}"])
            window_imps.clear()
            window_pars.clear()

        # ---- Checkpoints ----
        if episode % 10_000 < args.batch_episodes:
            ckpt = f"checkpoints/net_{episode}.pt"
            torch.save(net.state_dict(), ckpt)
            print(f"  saved checkpoint → {ckpt}")

    torch.save(net.state_dict(), "checkpoints/net_final.pt")
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training dynamics
    parser.add_argument("--episodes",       type=int,   default=200_000,
                        help="Total auction episodes")
    parser.add_argument("--batch-episodes", type=int,   default=512,
                        help="Auctions per PPO update")
    parser.add_argument("--ew-samples",     type=int,   default=10,
                        help="EW re-deals per episode for counterfactual reward")
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--reward-mode",    default="expected_score",
                        choices=["expected_score", "optimal_contract_regret", "par_relative"],
                        help="expected_score: IMP(achieved); "
                             "optimal_contract_regret: IMP(achieved−score(C*)); "
                             "par_relative: IMP(achieved−par)")
    parser.add_argument("--strain-bonus",   type=float, default=0.0,
                        help="IMP bonus when NS bids the optimal strain (0 = off)")
    parser.add_argument("--entropy-coef",   type=float, default=0.05,
                        help="Entropy bonus coefficient at the start of training")
    parser.add_argument("--entropy-final",  type=float, default=0.01,
                        help="Entropy bonus coefficient at the end of training")
    parser.add_argument("--vulnerability",  default="none",
                        choices=["none", "ns", "ew", "both"])
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--dataset",        default=None,
                        help="Path to pre-computed dataset dir (skips live DDS)")

    # Network architecture
    parser.add_argument("--hidden",      type=int, default=128,
                        help="Hidden width for MLP layers and LSTM")
    parser.add_argument("--embed-dim",   type=int, default=32,
                        help="Bid-token embedding dimension")
    parser.add_argument("--mlp-layers",  type=int, default=1,
                        help="Hidden layers in hand encoder and output heads")
    parser.add_argument("--lstm-layers", type=int, default=1,
                        help="Stacked LSTM layers for auction encoding")

    args = parser.parse_args()
    train(args)
