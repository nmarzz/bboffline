# Bridge Bidding Robot

A reinforcement learning agent that learns bridge bidding from scratch — no hardcoded conventions, no supervised imitation of human systems. North and South share a single neural network and discover their own bidding language through self-play, guided only by the IMP reward relative to the double-dummy par contract.

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart

**1. Generate a dataset** (eliminates live DDS calls during training):

```bash
python3 scripts/generate_dataset.py \
    --n-deals 50000 --ew-samples 20 \
    --output data/train --workers 4
```

**2. Train**:

```bash
python3 -m training.train \
    --dataset data/train \
    --episodes 500000 \
    --batch-episodes 512
```

**3. Inspect what the agent learned**:

```bash
python3 analysis/inspect.py --checkpoint checkpoints/net_final.pt
```

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(none)* | Pre-computed dataset dir. Omit to use live DDS (slow). |
| `--episodes` | 200 000 | Total auctions to train on |
| `--batch-episodes` | 512 | Auctions per PPO update |
| `--ew-samples` | 10 | EW re-deals used to estimate counterfactual reward |
| `--hidden` | 128 | Hidden width of every MLP layer and LSTM |
| `--embed-dim` | 32 | Bid-token embedding size |
| `--mlp-layers` | 1 | Hidden layers in hand encoder and output heads |
| `--lstm-layers` | 1 | Stacked LSTM layers for auction encoding |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |

**Scaled-up network:**
```bash
python3 -m training.train \
    --dataset data/train \
    --hidden 256 --embed-dim 64 --mlp-layers 2 --lstm-layers 2 \
    --episodes 1000000 --batch-episodes 1024
```

## Outputs

| Path | Contents |
|------|----------|
| `logs/train_metrics.csv` | Episode, mean IMP, par score, losses, entropy |
| `checkpoints/net_<step>.pt` | State dict saved every 10 000 episodes |
| `checkpoints/net_final.pt` | Final weights |

## Project layout

```
environment/
  deal.py        — random deal generation, hand encoding (52-dim binary)
  auction.py     — auction state machine, bid legality
  scoring.py     — DDS oracle, par contract, IMP table
  dataset.py     — memmap loader for pre-computed DDS tables
agents/
  nn_agent.py    — BiddingNet + NNAgent wrapper
  random_agent.py
training/
  ppo.py         — PPO updater, rollout buffer, return computation
  train.py       — training loop (vectorized + sequential rollout paths)
scripts/
  generate_dataset.py — multiprocess DDS pre-computation
analysis/
  inspect.py     — sample auction printer, IMP statistics
```

## Phase 2 (planned)

Add East–West agents with the same architecture, initialized from the trained N–S weights, and train adversarially. Reward for EW is `−IMP`. Double and Redouble are already in the action space (indices 36–37) and the auction state machine handles them; only the training loop needs extending.
