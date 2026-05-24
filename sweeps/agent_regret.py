"""W&B sweep agent for the regret-mode experiments.

Tests optimal_contract_regret with and without a warm-start from the
best expected_score checkpoint (checkpoints/net_final.pt).

Usage:
    # 1. Train a baseline expected_score model first:
    #    python -m training.train <best config from sweep 3>
    #    → produces checkpoints/net_final.pt and checkpoints/critic_final.pt

    # 2. Initialise this sweep (once):
    #    wandb sweep sweeps/config_regret.yaml --project bridge-bidding

    # 3. Launch agents:
    #    wandb agent <entity>/<project>/<sweep_id>
"""

import os
import sys
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import build_parser, train

# Fixed baseline checkpoints — these are never overwritten by subsequent runs.
# Produced once from the best expected_score configuration (sweep 3).
_POLICY_CKPT = "checkpoints/net_final_bestsweep.pt"
_CRITIC_CKPT = "checkpoints/critic_final_bestsweep.pt"

# Fixed across every trial — best settings from sweeps 1-3
FIXED = dict(
    dataset             = "data/train_merged",
    eval_dataset        = "data/val10000_ew20",
    episodes            = 100_000,
    eval_interval       = 10_000,
    reward_mode         = "optimal_contract_regret",
    hand_encoder        = "suit",
    vulnerability       = "none",
    entropy_final       = 0.01,
    ew_samples          = 10,
    wandb_project       = "bridge-bidding",
    device              = "auto",
    seed                = 42,
    # Architecture (fixed from sweeps 1-2)
    hidden              = 256,
    embed_dim           = 128,
    mini_batch_size     = 256,
    batch_episodes      = 1024,
    centralized_critic  = True,
    auction_encoder     = "transformer",
    gae_lambda          = 0.99,
    mlp_layers          = 1,
    lstm_layers         = 1,
)


def main():
    wandb.init()
    cfg = dict(wandb.config)

    args = build_parser().parse_args([])
    for k, v in {**FIXED, **cfg}.items():
        if k != "warm_start":          # handled below
            setattr(args, k, v)

    # critic_lr: match policy lr
    args.critic_lr = args.lr

    # Always warm-start from the expected_score baseline.
    # Cold-start with regret is a known failure mode (DESIGN.md §3.2).
    if not os.path.exists(_POLICY_CKPT):
        raise FileNotFoundError(
            f"Policy checkpoint not found: {_POLICY_CKPT}\n"
            f"Train an expected_score baseline first (see agent.py / config.yaml)."
        )
    args.checkpoint = _POLICY_CKPT
    args.critic_checkpoint = _CRITIC_CKPT if os.path.exists(_CRITIC_CKPT) else None

    train(args)


if __name__ == "__main__":
    main()
