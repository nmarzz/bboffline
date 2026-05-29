"""W&B sweep agent: expected_score fine-tuning from the baseline checkpoint.

All runs warm-start from checkpoints/net_final_bestsweep.pt (2.14 IMP).
Searches for hyperparameters that push past the current plateau.

Usage:
    wandb sweep sweeps/config_warmstart.yaml --project bridge-bidding
    wandb agent <entity>/bridge-bidding/<sweep_id>
"""

import os
import sys
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import build_parser, train

_POLICY_CKPT = "checkpoints/net_final_bestsweep.pt"
_CRITIC_CKPT = "checkpoints/critic_final_bestsweep.pt"

FIXED = dict(
    dataset            = "data/train_merged",
    eval_dataset       = "data/val10000_ew20",
    episodes           = 100_000,
    eval_interval      = 10_000,
    reward_mode        = "expected_score",
    hand_encoder       = "suit",
    vulnerability      = "none",
    entropy_final      = 0.005,
    strain_bonus       = 0.0,
    ew_samples         = 10,
    wandb_project      = "bridge-bidding",
    device             = "auto",
    seed               = 42,
    # Architecture fixed from sweeps 1-3
    hidden             = 256,
    embed_dim          = 128,
    mini_batch_size    = 256,
    centralized_critic = True,
    auction_encoder    = "transformer",
    gae_lambda         = 0.99,
    mlp_layers         = 1,
    lstm_layers        = 1,
    # Warm start
    checkpoint         = _POLICY_CKPT,
    critic_checkpoint  = _CRITIC_CKPT if os.path.exists(_CRITIC_CKPT) else None,
)


def main():
    if not os.path.exists(_POLICY_CKPT):
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {_POLICY_CKPT}\n"
            "Run the expected_score baseline training first."
        )

    wandb.init()
    cfg = dict(wandb.config)

    args = build_parser().parse_args([])
    for k, v in {**FIXED, **cfg}.items():
        setattr(args, k, v)

    args.critic_lr = args.lr

    train(args)


if __name__ == "__main__":
    main()
