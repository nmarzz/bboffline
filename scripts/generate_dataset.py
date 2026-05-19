"""Generate a pre-computed bridge dataset.

Usage:
    python scripts/generate_dataset.py --n-deals 10000 --ew-samples 20 \\
        --output data/dev

Output directory layout:
    ns_hands.npy    uint8  (N, 2, 52)
    dds_tables.npy  uint8  (N, K, 5, 4)
    meta.npy        dict
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def generate(n_deals, ew_samples, vulnerability, seed):
    from endplay.types import Denom, Player
    import endplay.dds as dds
    from environment.deal import random_deal, resample_ew
    from environment.scoring import _DENOM_IDX_TO_ENDPLAY, _PLAYER_IDX_TO_ENDPLAY

    rng = np.random.default_rng(seed)

    ns_hands_out   = np.zeros((n_deals, 2, 52),            dtype=np.uint8)
    dds_tables_out = np.zeros((n_deals, ew_samples, 5, 4), dtype=np.uint8)

    # Process in sub-batches that fit within the DDS 40-board limit
    sub_batch = max(1, 40 // ew_samples)
    i = 0
    with tqdm(total=n_deals, unit="deal") as pbar:
        while i < n_deals:
            count = min(sub_batch, n_deals - i)

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
            pbar.update(count)

    return ns_hands_out, dds_tables_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-deals",       type=int, default=10_000)
    parser.add_argument("--ew-samples",    type=int, default=20)
    parser.add_argument("--output",        default="data/dev")
    parser.add_argument("--vulnerability", default="none",
                        choices=["none", "ns", "ew", "both"])
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    n = args.n_deals
    K = args.ew_samples
    print(f"Generating {n:,} deals  K={K} EW samples  vulnerability={args.vulnerability}")
    print(f"Estimated size: {(n*2*52 + n*K*5*4) / 1e6:.1f} MB")

    t0 = time.time()
    ns_hands, dds_tables = generate(n, K, args.vulnerability, args.seed)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.0f}s  ({n * K / elapsed:.0f} boards/s)")

    np.save(os.path.join(args.output, "ns_hands.npy"),   ns_hands)
    np.save(os.path.join(args.output, "dds_tables.npy"), dds_tables)
    np.save(os.path.join(args.output, "meta.npy"),
            {"n_deals": n, "ew_samples": K, "vulnerability": args.vulnerability})

    print(f"Saved to {args.output}/")
    print(f"  ns_hands:   {ns_hands.nbytes   / 1e6:.1f} MB  {ns_hands.shape}")
    print(f"  dds_tables: {dds_tables.nbytes / 1e6:.1f} MB  {dds_tables.shape}")


if __name__ == "__main__":
    main()
