"""Generate a pre-computed bridge dataset.

Each worker generates a chunk of deals independently, then the main process
merges the results.

Usage:
    python scripts/generate_dataset.py --n-deals 10000 --ew-samples 20 \\
        --output data/dev --workers 4

Output directory layout:
    ns_hands.npy    uint8  (N, 2, 52)
    dds_tables.npy  uint8  (N, K, 5, 4)
    meta.npy        dict
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def _worker(args):
    """
    Generate `chunk_size` deals and solve K EW completions for each.
    Returns (ns_hands, dds_tables) as numpy arrays.
    """
    chunk_size, ew_samples, vulnerability, seed = args

    sys.path.insert(0, _PROJECT_ROOT)
    from endplay.types import Denom, Player
    import endplay.dds as dds
    from environment.deal import random_deal, resample_ew, _DEAL_GEN
    from environment.scoring import _DENOM_IDX_TO_ENDPLAY, _PLAYER_IDX_TO_ENDPLAY

    rng = np.random.default_rng(seed)

    ns_hands_out   = np.zeros((chunk_size, 2, 52),            dtype=np.uint8)
    dds_tables_out = np.zeros((chunk_size, ew_samples, 5, 4), dtype=np.uint8)

    sub_batch = max(1, 40 // ew_samples)
    i = 0
    while i < chunk_size:
        count = min(sub_batch, chunk_size - i)

        all_redeals = []
        batch_deals = []
        for _ in range(count):
            deal = random_deal(vulnerability)
            batch_deals.append(deal)
            all_redeals.extend(resample_ew(deal.endplay_deal, ew_samples, rng))

        tables = list(dds.calc_all_tables(all_redeals))

        for j, deal in enumerate(batch_deals):
            ns_hands_out[i + j, 0] = deal.hands[0]   # North
            ns_hands_out[i + j, 1] = deal.hands[2]   # South
            for k_idx in range(ew_samples):
                table = tables[j * ew_samples + k_idx]
                for d_idx, d in enumerate(_DENOM_IDX_TO_ENDPLAY):
                    for p_idx, p in enumerate(_PLAYER_IDX_TO_ENDPLAY):
                        dds_tables_out[i + j, k_idx, d_idx, p_idx] = table[d, p]

        i += count

    return ns_hands_out, dds_tables_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-deals",       type=int, default=10_000)
    parser.add_argument("--ew-samples",    type=int, default=20)
    parser.add_argument("--output",        default="data/dev")
    parser.add_argument("--workers",       type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--vulnerability", default="none",
                        choices=["none", "ns", "ew", "both"])
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    n          = args.n_deals
    K          = args.ew_samples
    n_workers  = min(args.workers, n)
    chunk_size = n // n_workers
    chunks     = [chunk_size] * (n_workers - 1) + [n - chunk_size * (n_workers - 1)]
    seeds      = [args.seed + i * 999983 for i in range(n_workers)]

    print(f"Generating {n:,} deals  K={K} EW samples  vulnerability={args.vulnerability}")
    print(f"Workers: {n_workers}  chunks: {chunks}")
    print(f"Estimated size: {(n*2*52 + n*K*5*4) / 1e6:.1f} MB")

    t0        = time.time()
    work_args = [(c, K, args.vulnerability, s) for c, s in zip(chunks, seeds)]

    if n_workers == 1:
        results = [_worker(work_args[0])]
    else:
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker, work_args),
                total=n_workers,
                desc="Generating",
                unit="chunk",
            ))

    elapsed      = time.time() - t0
    total_boards = n * K
    print(f"DDS done in {elapsed:.0f}s  ({total_boards / elapsed:.0f} boards/s)")

    # Merge chunks
    ns_hands   = np.concatenate([r[0] for r in results], axis=0)
    dds_tables = np.concatenate([r[1] for r in results], axis=0)

    # Save
    np.save(os.path.join(args.output, "ns_hands.npy"),   ns_hands)
    np.save(os.path.join(args.output, "dds_tables.npy"), dds_tables)
    np.save(os.path.join(args.output, "meta.npy"),
            {"n_deals": n, "ew_samples": K, "vulnerability": args.vulnerability})

    print(f"Saved to {args.output}/")
    print(f"  ns_hands:   {ns_hands.nbytes   / 1e6:.1f} MB  {ns_hands.shape}")
    print(f"  dds_tables: {dds_tables.nbytes / 1e6:.1f} MB  {dds_tables.shape}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
