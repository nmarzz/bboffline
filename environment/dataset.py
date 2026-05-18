"""Pre-computed bridge dataset.

Layout on disk (inside --output dir):
  ns_hands.npy    uint8  (N, 2, 52)   North and South hand vectors
  dds_tables.npy  uint8  (N, K, 5, 4) double-dummy trick counts
  meta.npy        dict   {"n_deals": N, "ew_samples": K, "vulnerability": str}

Both files are loaded as numpy memmaps so only the accessed pages are
pulled into RAM.

Table layout:
  dds_tables[i, k, denom_idx, player_idx]
  denom_idx:  0=C 1=D 2=H 3=S 4=NT
  player_idx: 0=N 1=E 2=S 3=W
"""

import os
import numpy as np


class BridgeDataset:
    def __init__(self, data_dir: str):
        meta          = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
        self.n_deals      = meta["n_deals"]
        self.ew_samples   = meta["ew_samples"]
        self.vulnerability = meta["vulnerability"]

        self.ns_hands   = np.load(os.path.join(data_dir, "ns_hands.npy"),
                                  mmap_mode="r")
        self.dds_tables = np.load(os.path.join(data_dir, "dds_tables.npy"),
                                  mmap_mode="r")

    def __len__(self):
        return self.n_deals

    def sample_indices(self, n: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, self.n_deals, size=n)

    def get(self, indices):
        """
        Return (ns_hands, dds_tables) for the given indices.
          ns_hands:   float32 (len, 2, 52)  — N hand is [0], S hand is [2]
          dds_tables: uint8   (len, K, 5, 4)
        """
        ns  = self.ns_hands[indices].astype(np.float32)
        dds = self.dds_tables[indices]
        return ns, dds
