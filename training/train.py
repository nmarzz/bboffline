"""Training loop for the bridge bidding agent.

Usage:
    python -m training.train [options]

Key options:
    --episodes        Total auction episodes to train on (default 200_000)
    --batch-episodes  Auctions per PPO update batch (default 512)
    --ew-samples      EW completions used for counterfactual reward (default 10)
    --dataset         Path to pre-computed dataset dir (strongly recommended)
    --eval-dataset    Path to held-out dataset for periodic greedy evaluation
    --eval-interval   Episodes between greedy evals (default 10_000)
    --wandb-project   W&B project name; omit to disable W&B logging
    --hidden          Hidden width of every MLP/LSTM layer (default 128)
    --embed-dim       Bid token embedding size (default 32)
    --mlp-layers      Hidden layers in hand-encoder and output heads (default 1)
    --lstm-layers     Stacked LSTM layers for auction encoding (default 1)

Logs a CSV of training metrics to logs/train_metrics.csv.
Saves checkpoints to checkpoints/net_<step>.pt every 10k episodes.
"""

import argparse
import math
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
from agents.nn_agent import BiddingNet, NNAgent, CentralizedCritic, count_params
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
    strain_bonus: float = 0.0,
    reward_mode: str = "expected_score",
    ew_counts: np.ndarray = None,   # (N,) per-deal EW sample counts, or None
    critic: CentralizedCritic = None,
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
            if critic is not None:
                north_t = torch.from_numpy(ns_hands[ns_idx, 0]).to(device)
                south_t = torch.from_numpy(ns_hands[ns_idx, 1]).to(device)
                vals_t  = critic(north_t, south_t, seqs_t)   # (B, 1)

        actions_t = Categorical(logits=log_probs_t).sample()  # (B,)

        for k, idx in enumerate(ns_idx):
            action   = actions_t[k].item()
            log_prob = log_probs_t[k, action].item()
            value    = vals_t[k, 0].item()
            player   = int(batch_dirs[k])

            auctions[idx].apply(action)

            all_trans[idx].append(Transition(
                ns_hands    = ns_hands[idx],        # (2, 52) — both hands
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
        dds_tables, auctions, vulnerability,
        strain_bonus=strain_bonus, reward_mode=reward_mode,
        ew_counts=ew_counts,
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
        # ns_hands: index 0=North (player 0), index 1=South (player 2), always
        ns_h = np.stack([deal.hands[0], deal.hands[2]])   # (2, 52)

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
                ns_hands    = ns_h,
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
# Greedy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_greedy(
    net: BiddingNet,
    eval_dataset: BridgeDataset,
    device: str,
    batch_size: int = 1024,
) -> dict:
    """Argmax rollouts over the full eval dataset. No sampling noise.

    Returns {"mean_imp", "std_imp", "mean_par"}.
    """
    net.eval()
    all_imps = []
    all_pars = []

    for start in range(0, eval_dataset.n_deals, batch_size):
        batch_idx  = np.arange(start, min(start + batch_size, eval_dataset.n_deals))
        ns_hands, dds_tables = eval_dataset.get(batch_idx)
        ew_counts  = eval_dataset.get_ew_counts(batch_idx)

        B        = len(batch_idx)
        auctions = [AuctionState(uncontested=True) for _ in range(B)]
        active   = np.ones(B, dtype=bool)

        while active.any():
            active_idx = np.where(active)[0]
            players    = np.array([auctions[i].current_player() for i in active_idx])

            for li in np.where(players % 2 == 1)[0]:
                idx = active_idx[li]
                auctions[idx].apply(0)
                if auctions[idx].is_complete():
                    active[idx] = False

            ns_local = np.where(players % 2 == 0)[0]
            if ns_local.size == 0:
                continue

            ns_idx      = active_idx[ns_local]
            C           = ns_idx.size
            batch_hands = np.empty((C, 52),             dtype=np.float32)
            batch_seqs  = np.full((C, MAX_AUCTION_LEN), -1, dtype=np.int64)
            batch_dirs  = np.empty(C,                   dtype=np.int64)
            batch_masks = np.zeros((C, NUM_BIDS),       dtype=bool)

            for k, idx in enumerate(ns_idx):
                player = auctions[idx].current_player()
                batch_hands[k] = ns_hands[idx, 0 if player == 0 else 1]
                batch_dirs[k]  = player
                seq = auctions[idx].to_sequence()
                batch_seqs[k, :len(seq)] = seq
                for b in auctions[idx].valid_bids():
                    batch_masks[k, b] = True

            log_probs_t, _ = net(
                torch.from_numpy(batch_hands).to(device),
                torch.from_numpy(batch_seqs).to(device),
                torch.from_numpy(batch_dirs).to(device),
                torch.from_numpy(batch_masks).to(device),
            )
            actions_t = log_probs_t.argmax(dim=-1)  # greedy

            for k, idx in enumerate(ns_idx):
                auctions[idx].apply(actions_t[k].item())
                if auctions[idx].is_complete():
                    active[idx] = False

        imps, pars = precomputed_imp_rewards(
            dds_tables, auctions, eval_dataset.vulnerability,
            ew_counts=ew_counts,
        )
        all_imps.extend(imps)
        all_pars.extend(pars)

    net.train()
    return {
        "mean_imp":     float(np.mean(all_imps)),
        "std_imp":      float(np.std(all_imps)),
        "mean_par_imp": float(np.mean(all_pars)),   # mean(IMP(par_i)), already in IMPs
    }


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

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    rng    = np.random.default_rng(args.seed)

    # ---- W&B ----
    wandb = None
    if args.wandb_project:
        import wandb as _wandb
        _wandb.init(project=args.wandb_project, config=vars(args))
        wandb = _wandb

    # ---- Dataset ----
    dataset = None
    if args.dataset:
        dataset = BridgeDataset(args.dataset)
        print(f"Dataset: {dataset.n_deals:,} deals  K={dataset.ew_samples}  "
              f"vul={dataset.vulnerability}  ({args.dataset})")

    # ---- Eval dataset ----
    eval_dataset = None
    if args.eval_dataset:
        eval_dataset = BridgeDataset(args.eval_dataset)
        print(f"Eval dataset: {eval_dataset.n_deals:,} deals  ({args.eval_dataset})")

    # ---- Network ----
    net = BiddingNet(
        hidden           = args.hidden,
        embed_dim        = args.embed_dim,
        mlp_layers       = args.mlp_layers,
        lstm_layers      = args.lstm_layers,
        hand_encoder     = args.hand_encoder,
        auction_encoder  = args.auction_encoder,
    ).to(device)

    # ---- Centralized critic (optional) ----
    critic = None
    if args.centralized_critic:
        critic = CentralizedCritic(
            hidden          = args.hidden,
            embed_dim       = args.embed_dim,
            mlp_layers      = args.mlp_layers,
            lstm_layers     = args.lstm_layers,
            auction_encoder = args.auction_encoder,
        ).to(device)
        print(f"CentralizedCritic  params: {count_params(critic):,}")

    # ---- Warm-start from checkpoint ----
    if getattr(args, "checkpoint", None):
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt)
        print(f"Loaded policy weights from {args.checkpoint}")
    if getattr(args, "critic_checkpoint", None) and critic is not None:
        ckpt = torch.load(args.critic_checkpoint, map_location=device)
        critic.load_state_dict(ckpt)
        print(f"Loaded critic weights from {args.critic_checkpoint}")

    agent   = NNAgent(net, device=device)
    updater = PPOUpdater(net, lr=args.lr, entropy_coef=args.entropy_coef, device=device,
                         critic=critic, critic_lr=args.critic_lr,
                         gae_lambda=args.gae_lambda,
                         mini_batch_size=args.mini_batch_size)
    buffer  = RolloutBuffer()

    print(f"BiddingNet  hidden={args.hidden}  embed={args.embed_dim}  "
          f"mlp_layers={args.mlp_layers}  lstm_layers={args.lstm_layers}  "
          f"hand_encoder={args.hand_encoder}")
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
        csv.writer(f).writerow(["episode", "mean_imp", "mean_par_imp",
                                 "policy_loss", "value_loss", "entropy",
                                 "elapsed_s"])

    episode     = 0
    epoch       = 0
    window_imps = []
    window_pars = []
    t0          = time.time()

    # Epoch-based iterator: shuffle the full dataset, yield batches in order,
    # re-shuffle at the start of each new epoch.  More sample-efficient than
    # random sampling with replacement (every deal seen once per epoch).
    def _epoch_batches(n: int, batch: int):
        nonlocal epoch
        while True:
            perm = rng.permutation(n)
            epoch += 1
            for start in range(0, n, batch):
                yield perm[start: start + batch]

    batch_iter = _epoch_batches(dataset.n_deals, args.batch_episodes) \
                 if dataset is not None else None

    while episode < args.episodes:
        # ---- Collect rollout ----
        if dataset is not None:
            indices             = next(batch_iter)
            ns_hands, dds_tbls  = dataset.get(indices)
            ew_counts           = dataset.get_ew_counts(indices)
            flat_trans, exp_imps, exp_pars = collect_batch_vectorized(
                net, ns_hands, dds_tbls,
                dataset.vulnerability, device,
                strain_bonus=args.strain_bonus, reward_mode=args.reward_mode,
                ew_counts=ew_counts, critic=critic,
            )
        else:
            flat_trans, exp_imps, exp_pars = collect_batch_sequential(
                agent, args.batch_episodes, args.vulnerability,
                device, args.ew_samples, rng,
                strain_bonus=args.strain_bonus, reward_mode=args.reward_mode,
            )

        # ---- PPO update ----
        frac = episode / args.episodes
        # Cosine annealing: lr_max → lr_min over training
        _lr_min = 1e-5
        _cosine_lr = _lr_min + 0.5 * (args.lr - _lr_min) * (1.0 + math.cos(math.pi * frac))
        updater.set_lr(_cosine_lr)
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
            mean_reward  = float(np.mean(window_imps))   # mode-dependent training signal
            mean_par_imp = float(np.mean(window_pars))   # always IMP(par_score)
            elapsed      = time.time() - t0
            critic_loss_str = (f"  critic_loss={loss_stats['critic_loss']:.4f}"
                               if critic is not None else "")
            # Label the training reward clearly so it is not confused with
            # eval/mean_imp (which is always absolute IMP regardless of mode).
            reward_label = {
                "expected_score":          "mean_IMP",
                "optimal_contract_regret": "mean_regret",
                "par_relative":            "mean_par_rel",
            }.get(args.reward_mode, "mean_reward")
            epoch_str = f"  epoch={epoch}" if dataset is not None else ""
            print(f"ep={episode:7d}{epoch_str}  {reward_label}={mean_reward:+.3f}  "
                  f"mean_par_IMP={mean_par_imp:+.3f}  "
                  f"policy_loss={loss_stats['policy_loss']:.4f}  "
                  f"entropy={loss_stats['entropy']:.3f}"
                  f"{critic_loss_str}  elapsed={elapsed:.0f}s")
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([episode, mean_reward, mean_par_imp,
                                        loss_stats["policy_loss"],
                                        loss_stats["value_loss"],
                                        loss_stats["entropy"],
                                        f"{elapsed:.1f}"])
            if wandb:
                log_dict = {
                    "train/mean_reward":  mean_reward,   # mode-dependent; use eval/mean_imp to compare runs
                    "train/mean_par_imp": mean_par_imp,
                    "train/policy_loss":  loss_stats["policy_loss"],
                    "train/value_loss":   loss_stats["value_loss"],
                    "train/entropy":      loss_stats["entropy"],
                    "train/lr":           updater._lr,
                    "train/epoch":        epoch,
                }
                if critic is not None:
                    log_dict["train/critic_loss"] = loss_stats["critic_loss"]
                wandb.log(log_dict, step=episode)
            window_imps.clear()
            window_pars.clear()

        # ---- Greedy eval ----
        if eval_dataset is not None and episode % args.eval_interval < args.batch_episodes:
            eval_stats = eval_greedy(net, eval_dataset, device)
            print(f"  [eval]  mean_IMP={eval_stats['mean_imp']:+.3f}  "
                  f"std={eval_stats['std_imp']:.3f}  "
                  f"mean_par_IMP={eval_stats['mean_par_imp']:+.3f}")
            if wandb:
                wandb.log({
                    "eval/mean_imp":     eval_stats["mean_imp"],
                    "eval/std_imp":      eval_stats["std_imp"],
                    "eval/mean_par_imp": eval_stats["mean_par_imp"],
                }, step=episode)

        # ---- Checkpoints ----
        if episode % 10_000 < args.batch_episodes:
            ckpt = f"checkpoints/net_{episode}.pt"
            torch.save(net.state_dict(), ckpt)
            print(f"  saved checkpoint → {ckpt}")
            if critic is not None:
                critic_ckpt = f"checkpoints/critic_{episode}.pt"
                torch.save(critic.state_dict(), critic_ckpt)
                print(f"  saved checkpoint → {critic_ckpt}")

    torch.save(net.state_dict(), "checkpoints/net_final.pt")
    if critic is not None:
        torch.save(critic.state_dict(), "checkpoints/critic_final.pt")

    # Save a permanent, run-named copy so net_final.pt can be overwritten
    # by subsequent runs without losing this checkpoint.
    if wandb:
        run_name = wandb.run.name          # e.g. "blooming-sweep-15"
        named_ckpt = f"checkpoints/net_final_{run_name}.pt"
        torch.save(net.state_dict(), named_ckpt)
        print(f"  saved named checkpoint → {named_ckpt}")
        if critic is not None:
            named_critic = f"checkpoints/critic_final_{run_name}.pt"
            torch.save(critic.state_dict(), named_critic)
            print(f"  saved named checkpoint → {named_critic}")
        wandb.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Training dynamics
    # Defaults reflect the best configuration found across sweeps 1-3:
    #   hidden=256, embed_dim=128, transformer encoder, gae_lambda=0.99   (sweeps 1-2)
    #   lr=9e-4, batch_episodes=1024, entropy_coef=0.05, mini_batch_size=256  (sweep 3)
    parser.add_argument("--episodes",       type=int,   default=200_000)
    parser.add_argument("--batch-episodes", type=int,   default=1_024)
    parser.add_argument("--ew-samples",     type=int,   default=10)
    parser.add_argument("--lr",             type=float, default=9e-4)
    parser.add_argument("--reward-mode",    default="expected_score",
                        choices=["expected_score", "optimal_contract_regret", "par_relative"])
    parser.add_argument("--strain-bonus",   type=float, default=0.0)
    parser.add_argument("--entropy-coef",   type=float, default=0.05)
    parser.add_argument("--entropy-final",  type=float, default=0.01)
    parser.add_argument("--vulnerability",  default="none",
                        choices=["none", "ns", "ew", "both"])
    parser.add_argument("--device",         default="auto",
                        help="cpu | cuda | mps | auto (default: auto-detect)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--dataset",        default=None)
    parser.add_argument("--eval-dataset",   default=None)
    parser.add_argument("--eval-interval",  type=int,   default=10_000)
    parser.add_argument("--wandb-project",      default=None)
    parser.add_argument("--centralized-critic",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-lr",          type=float, default=9e-4)
    # Network architecture
    parser.add_argument("--hidden",       type=int, default=256)
    parser.add_argument("--embed-dim",    type=int, default=128)
    parser.add_argument("--mlp-layers",   type=int, default=1)
    parser.add_argument("--lstm-layers",  type=int, default=1)
    parser.add_argument("--hand-encoder",    default="suit", choices=["suit", "mlp"])
    parser.add_argument("--auction-encoder", default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--gae-lambda",      type=float, default=0.99)
    parser.add_argument("--mini-batch-size",    type=int,   default=256)
    # Warm-start: load pre-trained weights before training begins
    parser.add_argument("--checkpoint",         default=None,
                        help="Path to a policy net .pt file to warm-start from.")
    parser.add_argument("--critic-checkpoint",  default=None,
                        help="Path to a critic net .pt file to warm-start from.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
