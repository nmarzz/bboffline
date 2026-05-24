"""W&B sweep agent entry point.

Each trial is launched by `wandb agent <sweep_id>`, which calls this script.
It reads hyperparameters from wandb.config and forwards them to train(),
bypassing argparse entirely so hyphen/underscore naming is never an issue.

Usage:
    # 1. Initialise the sweep (once):
    wandb sweep sweeps/config.yaml --project bridge-bidding

    # 2. Launch agents (one per GPU, repeat on each node):
    wandb agent <entity>/<project>/<sweep_id>
"""

import os
import sys
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import build_parser, train

# Fixed across every sweep trial — edit here, not in config.yaml
FIXED = dict(
    dataset             = "data/train_merged",
    eval_dataset        = "data/val10000_ew20",
    episodes            = 100_000,
    eval_interval       = 10_000,
    reward_mode         = "expected_score",
    hand_encoder        = "suit",
    vulnerability       = "none",
    entropy_final       = 0.01,   # always anneal to low entropy
    strain_bonus        = 0.0,
    ew_samples          = 10,
    wandb_project       = "bridge-bidding",
    device              = "auto",
    seed                = 42,
    # Fixed from sweep 1
    hidden              = 256,
    centralized_critic  = True,
    # Fixed from sweep 2
    auction_encoder     = "transformer",
    gae_lambda          = 0.99,
    mlp_layers          = 1,
    lstm_layers         = 1,
)


def main():
    wandb.init()
    cfg = dict(wandb.config)   # hyperparameters chosen by the sweep controller

    # Build a default args namespace then override with fixed + swept values
    args = build_parser().parse_args([])
    for k, v in {**FIXED, **cfg}.items():
        # W&B config uses underscores; argparse dest also uses underscores
        setattr(args, k, v)

    # critic_lr: match policy lr unless explicitly swept
    if "critic_lr" not in cfg:
        args.critic_lr = args.lr

    train(args)


if __name__ == "__main__":
    main()
