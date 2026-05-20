"""Double dummy oracle and scoring functions."""

import numpy as np
from endplay.types import Vul, Contract, Penalty, Denom, Player
import endplay.dds as dds

from environment.deal import BridgeDeal, resample_ew
from environment.auction import AuctionState

_DDS_BATCH_LIMIT = 40  # C library hard limit for CalcAllTables


def _calc_all_tables_chunked(deals: list) -> list:
    """Call calc_all_tables in chunks of _DDS_BATCH_LIMIT to stay within the C lib limit."""
    tables = []
    for start in range(0, len(deals), _DDS_BATCH_LIMIT):
        chunk = deals[start: start + _DDS_BATCH_LIMIT]
        tables.extend(dds.calc_all_tables(chunk))
    return tables

# Vulnerability mapping
_VUL = {
    "none": Vul.none,
    "ns":   Vul.ns,
    "ew":   Vul.ew,
    "both": Vul.both,
}

# Denom order matching auction.py: clubs=0, diamonds=1, hearts=2, spades=3, nt=4
_DENOM_IDX_TO_ENDPLAY = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
_PLAYER_IDX_TO_ENDPLAY = [Player.north, Player.east, Player.south, Player.west]

# IMP conversion table: (upper_bound_inclusive, imp_value)
_IMP_TABLE = [
    (10, 0), (40, 1), (80, 2), (120, 3), (160, 4), (210, 5),
    (260, 6), (310, 7), (360, 8), (420, 9), (490, 10), (590, 11),
    (740, 12), (890, 13), (1090, 14), (1290, 15), (1490, 16),
    (1740, 17), (1990, 18), (2240, 19), (2490, 20), (2990, 21),
    (3490, 22), (3990, 23),
]


def imps(score_diff: int) -> float:
    """Convert a point difference to IMPs (signed)."""
    sign = 1 if score_diff >= 0 else -1
    abs_diff = abs(score_diff)
    for threshold, imp_val in _IMP_TABLE:
        if abs_diff <= threshold:
            return float(sign * imp_val)
    return float(sign * 24)


# ---------------------------------------------------------------------------
# Scoring lookup table (built once per vulnerability, cached)
# ---------------------------------------------------------------------------

# Shape: (7, 5, 2, 14) — [level-1, denom_idx, ns_seat, tricks]
# ns_seat: 0 = North (player_idx 0), 1 = South (player_idx 2)
_SCORING_TABLES: dict = {}


def _build_scoring_table(vulnerability: str) -> np.ndarray:
    """
    Precompute NS scores for all uncontested contracts × all possible trick counts.
    980 endplay calls, done once per vulnerability then cached.
    """
    vul = _VUL[vulnerability]
    table = np.zeros((7, 5, 2, 14), dtype=np.int32)
    for level in range(1, 8):
        for d_idx, d in enumerate(_DENOM_IDX_TO_ENDPLAY):
            for seat, p_idx in enumerate((0, 2)):       # North=0, South=2
                declarer = _PLAYER_IDX_TO_ENDPLAY[p_idx]
                for tricks in range(14):
                    result = tricks - (level + 6)
                    c = Contract(level=level, denom=d, declarer=declarer,
                                 penalty=Penalty.passed, result=result)
                    table[level - 1, d_idx, seat, tricks] = c.score(vul)
    return table


def _get_scoring_table(vulnerability: str) -> np.ndarray:
    if vulnerability not in _SCORING_TABLES:
        _SCORING_TABLES[vulnerability] = _build_scoring_table(vulnerability)
    return _SCORING_TABLES[vulnerability]


def _optimal_ns_scores_k(tables_k: np.ndarray, scoring_table: np.ndarray) -> np.ndarray:
    """
    Find C* = argmax_C E_EW[score(C, EW) | NS] and return its per-sample scores.

    Scans all 70 uncontested NS contracts (7 levels × 5 denoms × N/S declarer)
    using pure numpy indexing — no DDS calls.

    Args:
        tables_k:      (K, 5, 4) uint8 trick counts
        scoring_table: (7, 5, 2, 14) int32 from _get_scoring_table

    Returns:
        (K,) int32 — score of C* on each EW sample
    """
    best_mean   = -np.inf
    best_scores = np.zeros(tables_k.shape[0], dtype=np.int32)

    for level in range(1, 8):
        for d_idx in range(5):
            for seat, p_idx in enumerate((0, 2)):
                tricks_k  = tables_k[:, d_idx, p_idx].astype(np.int64)        # (K,)
                scores_k  = scoring_table[level - 1, d_idx, seat, tricks_k]   # (K,)
                mean_score = float(scores_k.mean())
                if mean_score > best_mean:
                    best_mean   = mean_score
                    best_scores = scores_k

    return best_scores


def _achieved_ns_scores_k(tables_k: np.ndarray, auction,
                           scoring_table: np.ndarray) -> np.ndarray:
    """
    NS score on each EW sample for the contract reached in `auction`.
    Returns zeros for a passed-out auction or EW-declared contract.

    Args:
        tables_k:      (K, 5, 4) uint8 trick counts
        auction:       completed AuctionState
        scoring_table: (7, 5, 2, 14) int32 from _get_scoring_table

    Returns:
        (K,) int32
    """
    K  = tables_k.shape[0]
    cf = auction.final_contract()
    if cf is None or cf[2] not in (0, 2):   # passout or EW declared
        return np.zeros(K, dtype=np.int32)

    level, d_idx, p_idx, _doubled, _redoubled = cf
    seat     = 0 if p_idx == 0 else 1
    tricks_k = tables_k[:, d_idx, p_idx].astype(np.int64)
    return scoring_table[level - 1, d_idx, seat, tricks_k]


def _tricks(table, denom_idx: int, player_idx: int) -> int:
    """
    Uniform trick lookup that works with both:
      - endplay DDTable objects  (table[Denom, Player])
      - plain (5, 4) numpy uint8 arrays  (table[denom_idx, player_idx])
    """
    if isinstance(table, np.ndarray):
        return int(table[denom_idx, player_idx])
    return table[_DENOM_IDX_TO_ENDPLAY[denom_idx], _PLAYER_IDX_TO_ENDPLAY[player_idx]]


def table_to_array(ep_table) -> np.ndarray:
    """Convert an endplay DDTable to a (5, 4) uint8 numpy array."""
    arr = np.zeros((5, 4), dtype=np.uint8)
    for d_idx in range(5):
        for p_idx in range(4):
            arr[d_idx, p_idx] = ep_table[
                _DENOM_IDX_TO_ENDPLAY[d_idx], _PLAYER_IDX_TO_ENDPLAY[p_idx]
            ]
    return arr


def best_ns_strain(table) -> int:
    """
    Return the denom_idx (0=C,1=D,2=H,3=S,4=NT) where NS can take the most
    double-dummy tricks (taking the better of North and South as declarer).
    Accepts either an endplay DDTable or a (5, 4) numpy array.
    """
    best_denom  = 0
    best_tricks = -1
    for d_idx in range(5):
        tricks = max(_tricks(table, d_idx, 0),   # North
                     _tricks(table, d_idx, 2))   # South
        if tricks > best_tricks:
            best_tricks = tricks
            best_denom  = d_idx
    return best_denom


def ns_par_score(table, vulnerability: str) -> int:
    """
    Best score N-S can achieve in an uncontested auction given double-dummy play.
    Returns 0 if no N-S contract makes (pass-out).
    Accepts either an endplay DDTable or a (5, 4) numpy array.
    """
    vul = _VUL[vulnerability]
    best = 0
    for level in range(1, 8):
        for denom_idx, d in enumerate(_DENOM_IDX_TO_ENDPLAY):
            for declarer_idx in (0, 2):  # North=0, South=2
                declarer = _PLAYER_IDX_TO_ENDPLAY[declarer_idx]
                tricks = _tricks(table, denom_idx, declarer_idx)
                result = tricks - (level + 6)
                if result >= 0:
                    c = Contract(level=level, denom=d, declarer=declarer,
                                 penalty=Penalty.passed, result=result)
                    score = c.score(vul)
                    if score > best:
                        best = score
    return best


def achieved_ns_score(table, auction: AuctionState, vulnerability: str) -> int:
    """
    Score N-S achieves for the final contract, from N-S perspective.
    Returns 0 for a passed-out auction.
    Accepts either an endplay DDTable or a (5, 4) numpy array.
    """
    contract_info = auction.final_contract()
    if contract_info is None:
        return 0

    level, denom_idx, declarer_dir, doubled, redoubled = contract_info
    vul = _VUL[vulnerability]
    d = _DENOM_IDX_TO_ENDPLAY[denom_idx]
    declarer = _PLAYER_IDX_TO_ENDPLAY[declarer_dir]
    tricks = _tricks(table, denom_idx, declarer_dir)
    result = tricks - (level + 6)

    penalty = Penalty.passed
    if redoubled:
        penalty = Penalty.redoubled
    elif doubled:
        penalty = Penalty.doubled

    c = Contract(level=level, denom=d, declarer=declarer, penalty=penalty, result=result)
    raw_score = c.score(vul)

    if declarer_dir in (0, 2):
        return raw_score
    else:
        return -raw_score


def _episode_reward(
    achieved: int,
    par: int,
    optimal_score: int,
    reward_mode: str,
    strain_correct: bool,
    strain_bonus: float,
) -> float:
    """Compute the scalar reward for one (auction, EW layout) sample."""
    if reward_mode == "optimal_contract_regret":
        reward = float(imps(achieved - optimal_score))
    elif reward_mode == "expected_score":
        reward = float(imps(achieved))
    else:  # "par_relative"
        reward = float(imps(achieved - par))

    if strain_bonus > 0.0 and strain_correct:
        reward += strain_bonus

    return reward


def batch_expected_imp_rewards(
    ep_deals,
    auctions: list,
    vulnerability: str,
    k: int,
    rng=None,
    strain_bonus: float = 0.0,
    reward_mode: str = "optimal_contract_regret",
) -> tuple:
    """
    Compute counterfactual expected rewards for a batch of episodes.

    For each episode, re-deals the E-W cards k times and averages the reward
    across those samples.  Because N-S can only act on their own hands,
    judging a bid sequence on a single E-W layout is noisy — averaging over
    many E-W completions reduces variance substantially.

    Uses a single calc_all_tables call for the full N*k board batch, which
    exploits DDS's internal threading rather than calling the solver N*k times.

    Args:
        ep_deals:      list of endplay Deal objects (one per episode)
        auctions:      list of completed AuctionState objects (one per episode)
        vulnerability: "none" | "ns" | "ew" | "both"
        k:             number of E-W re-deals per episode
        rng:           numpy Generator for reproducibility (created if None)
        reward_mode:   "optimal_contract_regret" — IMP(achieved − score(C*)); regret vs E-optimal
                       "expected_score"          — IMP(achieved); absolute expected IMP
                       "par_relative"            — IMP(achieved − par); vs double-dummy par

    Returns:
        (expected_rewards, expected_pars)
        expected_rewards[i]: mean reward over k E-W samples for episode i
        expected_pars[i]:    mean N-S par score over k E-W samples for episode i (diagnostic)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build all N*k re-deals in episode order
    all_redeals = []
    for ep_deal in ep_deals:
        all_redeals.extend(resample_ew(ep_deal, k, rng))

    # Chunked DDS calls — each chunk of 40 runs threaded internally
    all_ep_tables = _calc_all_tables_chunked(all_redeals)

    scoring_table = _get_scoring_table(vulnerability)
    expected_imps = []
    expected_pars = []

    for i, (ep_deal, auction) in enumerate(zip(ep_deals, auctions)):
        ep_tables   = all_ep_tables[i * k: (i + 1) * k]
        tables_k    = np.stack([table_to_array(t) for t in ep_tables])  # (K, 5, 4)

        cf          = auction.final_contract()
        ns_declared = cf is not None and cf[2] in (0, 2)
        bid_strain  = cf[1] if cf is not None else None

        if reward_mode == "optimal_contract_regret":
            optimal_scores_k  = _optimal_ns_scores_k(tables_k, scoring_table)
            achieved_scores_k = _achieved_ns_scores_k(tables_k, auction, scoring_table)

        imp_samples = []
        par_samples = []
        for k_idx, table in enumerate(ep_tables):
            par      = ns_par_score(table, vulnerability)
            achieved = (int(achieved_scores_k[k_idx]) if reward_mode == "optimal_contract_regret"
                        else achieved_ns_score(table, auction, vulnerability))
            optimal  = int(optimal_scores_k[k_idx]) if reward_mode == "optimal_contract_regret" else 0
            strain_correct = ns_declared and bid_strain == best_ns_strain(table)
            reward = _episode_reward(achieved, par, optimal, reward_mode, strain_correct, strain_bonus)
            imp_samples.append(reward)
            par_samples.append(par)

        expected_imps.append(float(np.mean(imp_samples)))
        expected_pars.append(float(np.mean(par_samples)))

    return expected_imps, expected_pars


def precomputed_imp_rewards(
    dds_tables_batch: np.ndarray,
    auctions: list,
    vulnerability: str,
    k: int = None,
    strain_bonus: float = 0.0,
    reward_mode: str = "optimal_contract_regret",
) -> tuple:
    """
    Compute expected rewards using pre-computed DDS tables from a dataset.
    No live DDS calls — pure numpy lookups.

    Args:
        dds_tables_batch: uint8 array (N, K, 5, 4) from BridgeDataset.get()
        auctions:         list of N completed AuctionState objects
        vulnerability:    "none" | "ns" | "ew" | "both"
        k:                number of EW samples to use (≤ K stored; None = use all)
        reward_mode:      see batch_expected_imp_rewards

    Returns:
        (expected_rewards, expected_pars)  — same contract as batch_expected_imp_rewards
    """
    K_stored = dds_tables_batch.shape[1]
    if k is None or k > K_stored:
        k = K_stored

    scoring_table = _get_scoring_table(vulnerability)
    expected_imps = []
    expected_pars = []

    for i, auction in enumerate(auctions):
        tables_k = dds_tables_batch[i, :k]   # (k, 5, 4)

        cf          = auction.final_contract()
        ns_declared = cf is not None and cf[2] in (0, 2)
        bid_strain  = cf[1] if cf is not None else None

        if reward_mode == "optimal_contract_regret":
            optimal_scores_k  = _optimal_ns_scores_k(tables_k, scoring_table)
            achieved_scores_k = _achieved_ns_scores_k(tables_k, auction, scoring_table)

        imp_samples = []
        par_samples = []
        for k_idx in range(k):
            table    = tables_k[k_idx]
            par      = ns_par_score(table, vulnerability)
            achieved = (int(achieved_scores_k[k_idx]) if reward_mode == "optimal_contract_regret"
                        else achieved_ns_score(table, auction, vulnerability))
            optimal  = int(optimal_scores_k[k_idx]) if reward_mode == "optimal_contract_regret" else 0
            strain_correct = ns_declared and bid_strain == best_ns_strain(table)
            reward = _episode_reward(achieved, par, optimal, reward_mode, strain_correct, strain_bonus)
            imp_samples.append(reward)
            par_samples.append(par)

        expected_imps.append(float(np.mean(imp_samples)))
        expected_pars.append(float(np.mean(par_samples)))

    return expected_imps, expected_pars
