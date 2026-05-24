"""Merge multiple bridge datasets into one, supporting variable EW sample counts.

Datasets may have different numbers of EW samples (K). The merged dataset pads
all dds_tables arrays to max_K with zeros and stores a per-deal ew_counts.npy
so downstream code can average over only the valid samples.

Usage:
    python scripts/merge_datasets.py data/train10 data/train20 --output data/train_merged

All input datasets must share the same vulnerability setting.
"""

import argparse
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def load_dataset(data_dir):
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    ns_hands   = np.load(os.path.join(data_dir, "ns_hands.npy"))
    dds_tables = np.load(os.path.join(data_dir, "dds_tables.npy"))
    return meta, ns_hands, dds_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input dataset directories")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    datasets = []
    for d in args.inputs:
        print(f"Loading {d} ...")
        meta, ns, dds = load_dataset(d)
        print(f"  n_deals={meta['n_deals']:,}  ew_samples={meta['ew_samples']}  "
              f"vul={meta['vulnerability']}")
        datasets.append((meta, ns, dds))

    vulns = {m["vulnerability"] for m, _, _ in datasets}
    if len(vulns) > 1:
        print(f"ERROR: datasets have different vulnerability settings: {vulns}")
        sys.exit(1)
    vulnerability = vulns.pop()

    max_K     = max(m["ew_samples"] for m, _, _ in datasets)
    n_total   = sum(m["n_deals"]    for m, _, _ in datasets)

    print(f"\nMerging {len(datasets)} datasets → {n_total:,} deals  max_K={max_K}")

    ns_out   = np.empty((n_total, 2, 52),         dtype=np.uint8)
    dds_out  = np.zeros((n_total, max_K, 5, 4),   dtype=np.uint8)
    cnt_out  = np.empty(n_total,                  dtype=np.int32)

    cursor = 0
    for meta, ns, dds in datasets:
        n = meta["n_deals"]
        K = meta["ew_samples"]
        ns_out[cursor:cursor + n]        = ns
        dds_out[cursor:cursor + n, :K]   = dds
        cnt_out[cursor:cursor + n]       = K
        cursor += n

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "ns_hands.npy"),   ns_out)
    np.save(os.path.join(args.output, "dds_tables.npy"), dds_out)
    np.save(os.path.join(args.output, "ew_counts.npy"),  cnt_out)
    np.save(os.path.join(args.output, "meta.npy"),
            {"n_deals": n_total, "ew_samples": max_K, "vulnerability": vulnerability})

    print(f"Saved to {args.output}/")
    print(f"  ns_hands:   {ns_out.nbytes   / 1e6:.1f} MB  {ns_out.shape}")
    print(f"  dds_tables: {dds_out.nbytes  / 1e6:.1f} MB  {dds_out.shape}")
    print(f"  ew_counts:  {cnt_out.nbytes  / 1e6:.1f} MB  {cnt_out.shape}")
    counts, freqs = np.unique(cnt_out, return_counts=True)
    for c, f in zip(counts, freqs):
        print(f"    K={c}: {f:,} deals")


if __name__ == "__main__":
    main()
