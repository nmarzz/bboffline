"""Pre-compute per-deal curriculum sampling weights for a dataset.

For each deal, finds C* (the optimal expected contract level) and assigns a
sampling weight based on its category.  Game and slam hands are upweighted so
the training loop sees them more often per epoch, giving more gradient signal
where the agent currently struggles most.

Weights are saved as  <dataset_dir>/curriculum_weights.npy  (float32, shape N).

Usage:
    python scripts/make_curriculum_weights.py data/train_merged
    python scripts/make_curriculum_weights.py data/train_merged \\
        --game-weight 4 --slam-weight 6

Default weights  (tuned to findings from diagnose.py):
    pass            : 0.5   (agent overbids these; less signal needed)
    partscore_low   : 1.0   (baseline)
    partscore_high  : 2.0
    game            : 4.0   (only 47% game success; needs most signal)
    slam            : 6.0   (essentially never bid; highest priority)
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.scoring import _get_scoring_table

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


def best_contract_level(tables_k: np.ndarray, scoring_table: np.ndarray) -> int:
    """Return C* level (0=pass) that maximises E[IMP(score)] over K EW samples."""
    best_mean_imp = 0.0
    best_level    = 0
    for level in range(1, 8):
        for d in range(5):
            for seat, p in enumerate((0, 2)):
                sc = scoring_table[level-1, d, seat, tables_k[:, d, p].astype(np.int64)]
                mi = float(imp_vec(sc).mean())
                if mi > best_mean_imp:
                    best_mean_imp = mi
                    best_level    = level
    return best_level


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset", help="Path to dataset directory")
    parser.add_argument("--pass-weight",         type=float, default=0.5)
    parser.add_argument("--partscore-low-weight", type=float, default=1.0)
    parser.add_argument("--partscore-high-weight",type=float, default=2.0)
    parser.add_argument("--game-weight",          type=float, default=4.0)
    parser.add_argument("--slam-weight",          type=float, default=6.0)
    args = parser.parse_args()

    meta = np.load(os.path.join(args.dataset, "meta.npy"), allow_pickle=True).item()
    dds  = np.load(os.path.join(args.dataset, "dds_tables.npy"), mmap_mode="r")
    N, K = dds.shape[:2]

    ew_counts_path = os.path.join(args.dataset, "ew_counts.npy")
    ew_counts = np.load(ew_counts_path) if os.path.exists(ew_counts_path) else None

    vul = meta["vulnerability"]
    scoring_table = _get_scoring_table(vul)

    cat_weights = {
        0: args.pass_weight,          # pass
        1: args.partscore_low_weight, # level 1-2
        2: args.partscore_low_weight,
        3: args.partscore_high_weight,# level 3
        4: args.game_weight,          # level 4-5
        5: args.game_weight,
        6: args.slam_weight,          # level 6-7
        7: args.slam_weight,
    }

    print(f"Computing C* levels for {N:,} deals (K up to {K})...")
    weights   = np.empty(N, dtype=np.float32)
    cat_counts = {name: 0 for name in ["pass","partscore_low","partscore_high","game","slam"]}

    for i in range(N):
        k_i      = int(ew_counts[i]) if ew_counts is not None else K
        level    = best_contract_level(dds[i, :k_i], scoring_table)
        weights[i] = cat_weights[level]

        if level == 0:               cat_counts["pass"]           += 1
        elif level <= 2:             cat_counts["partscore_low"]  += 1
        elif level == 3:             cat_counts["partscore_high"] += 1
        elif level <= 5:             cat_counts["game"]           += 1
        else:                        cat_counts["slam"]           += 1

        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,} / {N:,}")

    # Normalise to sum to 1 (for use with np.random.choice p= argument)
    weights /= weights.sum()

    out_path = os.path.join(args.dataset, "curriculum_weights.npy")
    np.save(out_path, weights)
    print(f"\nSaved → {out_path}")

    print("\nCategory breakdown:")
    for cat, count in cat_counts.items():
        print(f"  {cat:<18}: {count:>7,}  {count/N*100:>5.1f}%")

    print("\nEffective sampling rates (relative to partscore_low=1.0):")
    raw = {"pass": args.pass_weight, "partscore_low": args.partscore_low_weight,
           "partscore_high": args.partscore_high_weight, "game": args.game_weight,
           "slam": args.slam_weight}
    base = args.partscore_low_weight
    for cat, w in raw.items():
        print(f"  {cat:<18}: {w/base:.1f}x")


if __name__ == "__main__":
    main()
