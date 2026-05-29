"""Diagnostic analysis of a trained bidding agent.

Loads a checkpoint, runs greedy rollouts over the eval dataset, and reports:
  - Overall eval/mean_imp vs theoretical ceiling
  - Per-category breakdown (pass / partscore-low / partscore-high / game / slam)
  - Strain accuracy per category
  - Auction length distribution
  - Contract level distribution on game and slam hands
  - Sample missed-game auctions

Usage:
    python -m analysis.diagnose --checkpoint checkpoints/net_final.pt
    python -m analysis.diagnose --checkpoint checkpoints/net_final_dulcet-haze-245.pt \\
                                --eval-dataset data/val10000_ew20
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nn_agent import BiddingNet
from environment.dataset import BridgeDataset
from environment.scoring import _get_scoring_table, ns_par_score
from environment.auction import AuctionState, NUM_BIDS, MAX_AUCTION_LEN, bid_name
from training.train import eval_greedy

DENOM_NAMES = ["C", "D", "H", "S", "NT"]
DIR_NAMES   = ["N", "E", "S", "W"]

_IMP_TABLE = [
    (10,0),(40,1),(80,2),(120,3),(160,4),(210,5),(260,6),(310,7),(360,8),
    (420,9),(490,10),(590,11),(740,12),(890,13),(1090,14),(1290,15),(1490,16),
    (1740,17),(1990,18),(2240,19),(2490,20),(2990,21),(3490,22),(3990,23),
]

def imp_vec(scores: np.ndarray) -> np.ndarray:
    result = np.full(len(scores), 24, dtype=np.float32)
    for thresh, v in reversed(_IMP_TABLE):
        result[np.abs(scores) <= thresh] = v
    result[scores < 0] *= -1
    return result


def hand_category(best_level: int) -> str:
    if best_level == 0:  return "pass"
    if best_level <= 2:  return "partscore_low"
    if best_level == 3:  return "partscore_high"
    if best_level <= 5:  return "game"
    return "slam"


def greedy_rollout(net, eval_ds, device, batch_size=512):
    """Run greedy auctions over the full eval dataset. Returns list of record dicts."""
    scoring_table = _get_scoring_table(eval_ds.vulnerability)
    records = []

    for start in range(0, eval_ds.n_deals, batch_size):
        idx = np.arange(start, min(start + batch_size, eval_ds.n_deals))
        ns_hands, dds_tables = eval_ds.get(idx)
        B = len(idx)

        auctions = [AuctionState(uncontested=True) for _ in range(B)]
        active   = np.ones(B, dtype=bool)

        while active.any():
            ai = np.where(active)[0]
            pl = np.array([auctions[i].current_player() for i in ai])

            for li in np.where(pl % 2 == 1)[0]:
                i = ai[li]
                auctions[i].apply(0)
                if auctions[i].is_complete():
                    active[i] = False

            ns_l = np.where(pl % 2 == 0)[0]
            if not ns_l.size:
                continue

            ni = ai[ns_l]; C = ni.size
            bh = np.empty((C, 52),              dtype=np.float32)
            bs = np.full((C, MAX_AUCTION_LEN), -1, dtype=np.int64)
            bd = np.empty(C,                    dtype=np.int64)
            bm = np.zeros((C, NUM_BIDS),        dtype=bool)

            for k, i in enumerate(ni):
                p = auctions[i].current_player()
                bh[k] = ns_hands[i, 0 if p == 0 else 1]
                bd[k] = p
                seq = auctions[i].to_sequence()
                bs[k, :len(seq)] = seq
                for b in auctions[i].valid_bids():
                    bm[k, b] = True

            with torch.no_grad():
                lp, _, _ = net(torch.from_numpy(bh).to(device),
                            torch.from_numpy(bs).to(device),
                            torch.from_numpy(bd).to(device),
                            torch.from_numpy(bm).to(device))
            acts = lp.argmax(-1)

            for k, i in enumerate(ni):
                auctions[i].apply(acts[k].item())
                if auctions[i].is_complete():
                    active[i] = False

        for b, i in enumerate(idx):
            tk = dds_tables[b]
            K  = eval_ds.ew_samples
            cf = auctions[b].final_contract()
            bids_made = [bid_name(x) for x in auctions[b].bids if x != 0]

            if cf is None or cf[2] not in (0, 2):
                ach_scores = np.zeros(K, dtype=np.int32)
                agent_level, agent_denom = 0, -1
            else:
                lv, di, pi, _, _ = cf
                seat = 0 if pi == 0 else 1
                ach_scores = scoring_table[lv-1, di, seat, tk[:, di, pi].astype(np.int64)]
                agent_level, agent_denom = lv, di

            best_mean_imp = 0.0
            best_level, best_denom = 0, -1
            for level in range(1, 8):
                for d in range(5):
                    for seat2, p2 in enumerate((0, 2)):
                        sc = scoring_table[level-1, d, seat2, tk[:, d, p2].astype(np.int64)]
                        mi = float(imp_vec(sc).mean())
                        if mi > best_mean_imp:
                            best_mean_imp = mi
                            best_level, best_denom = level, d

            records.append(dict(
                cat         = hand_category(best_level),
                agent_level = agent_level,
                agent_denom = agent_denom,
                best_level  = best_level,
                best_denom  = best_denom,
                mean_ach    = float(imp_vec(ach_scores).mean()),
                ceiling     = best_mean_imp,
                bids        = bids_made,
            ))

    return records


def print_report(records, overall_stats):
    N = len(records)
    cats = ["pass", "partscore_low", "partscore_high", "game", "slam"]

    print(f"\n{'='*65}")
    print(f"OVERALL  eval/mean_imp={overall_stats['mean_imp']:.4f}  "
          f"std={overall_stats['std_imp']:.4f}")
    print(f"{'='*65}\n")

    # ── Per-category summary ─────────────────────────────────────────────────
    print(f"{'Category':<18} {'N':>5} {'%':>5}  {'Achieved':>9} {'Ceiling':>9} "
          f"{'Gap':>7} {'Captured':>9}")
    print("-" * 65)
    for cat in cats:
        sub = [r for r in records if r["cat"] == cat]
        if not sub:
            continue
        n   = len(sub)
        ach = np.mean([r["mean_ach"] for r in sub])
        cil = np.mean([r["ceiling"]  for r in sub])
        cap = ach / cil * 100 if cil > 0 else 100.0
        print(f"  {cat:<16} {n:>5} {n/N*100:>4.1f}%"
              f"  {ach:>9.3f} {cil:>9.3f} {cil-ach:>7.3f} {cap:>8.1f}%")
    total_ach = np.mean([r["mean_ach"] for r in records])
    total_cil = np.mean([r["ceiling"]  for r in records])
    print("-" * 65)
    print(f"  {'TOTAL':<16} {N:>5}      "
          f"  {total_ach:>9.3f} {total_cil:>9.3f} {total_cil-total_ach:>7.3f} "
          f"{total_ach/total_cil*100:>8.1f}%\n")

    # ── Strain accuracy ──────────────────────────────────────────────────────
    print("STRAIN ACCURACY (when agent bids)")
    for cat in cats[1:]:
        sub = [r for r in records if r["cat"] == cat and r["agent_level"] > 0
               and r["best_denom"] >= 0]
        if not sub:
            continue
        correct = sum(1 for r in sub if r["agent_denom"] == r["best_denom"])
        print(f"  {cat:<18}: {correct}/{len(sub)} = {correct/len(sub)*100:.1f}%")

    # ── Auction length ───────────────────────────────────────────────────────
    from collections import Counter
    print("\nAUCTION LENGTH (non-pass NS bids)")
    lc = Counter(len(r["bids"]) for r in records)
    for k in sorted(lc):
        print(f"  {k} bids: {lc[k]:>5,}  {lc[k]/N*100:.1f}%")

    # ── Contract level distribution ──────────────────────────────────────────
    print("\nCONTRACT LEVEL — all hands")
    lc2 = Counter(r["agent_level"] for r in records)
    for k in sorted(lc2):
        lbl = "Pass" if k == 0 else f"Level {k}"
        print(f"  {lbl}: {lc2[k]:>5,}  {lc2[k]/N*100:.1f}%")

    for cat, label in [("game", "GAME"), ("slam", "SLAM")]:
        sub = [r for r in records if r["cat"] == cat]
        if not sub:
            continue
        print(f"\nCONTRACT LEVEL — {label} hands (C*={cat})")
        lc3 = Counter(r["agent_level"] for r in sub)
        for k in sorted(lc3):
            lbl = "Pass" if k == 0 else f"Level {k}"
            print(f"  {lbl}: {lc3[k]:>5,}  {lc3[k]/len(sub)*100:.1f}%")

    # ── Sample missed games ──────────────────────────────────────────────────
    missed = [r for r in records if r["cat"] == "game" and r["agent_level"] <= 2][:10]
    if missed:
        print("\nSAMPLE MISSED GAMES (C*=game, agent bid ≤2)")
        for r in missed:
            bd = DENOM_NAMES[r["agent_denom"]] if r["agent_level"] > 0 else "-"
            print(f"  bids={r['bids']:<30}  "
                  f"agent={r['agent_level']}{bd}  "
                  f"C*={r['best_level']}{DENOM_NAMES[r['best_denom']]}  "
                  f"ach={r['mean_ach']:.2f}  ceil={r['ceiling']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/net_final.pt")
    parser.add_argument("--eval-dataset", default="data/val10000_ew20")
    parser.add_argument("--hidden",       type=int, default=256)
    parser.add_argument("--embed-dim",    type=int, default=128)
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():       device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else:                               device = "cpu"
    else:
        device = args.device

    print(f"Loading {args.checkpoint} → {device}")
    net = BiddingNet(hidden=args.hidden, embed_dim=args.embed_dim,
                     mlp_layers=1, lstm_layers=1,
                     hand_encoder="suit", auction_encoder="transformer").to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    eval_ds = BridgeDataset(args.eval_dataset)
    print(f"Eval dataset: {eval_ds.n_deals:,} deals")

    overall = eval_greedy(net, eval_ds, device)
    records = greedy_rollout(net, eval_ds, device)
    print_report(records, overall)


if __name__ == "__main__":
    main()
