"""FastAPI backend for the bridge bidding web app."""
import sys
import uuid
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from agents.nn_agent import BiddingNet, NNAgent
from agents.random_agent import RandomAgent
from environment.auction import AuctionState, MAX_AUCTION_LEN, NUM_BIDS, bid_name
from environment.deal import BridgeDeal, random_deal, resample_ew, hand_to_vector
from environment.scoring import achieved_ns_score, imps, _calc_all_tables_chunked

import endplay.dds as dds
from endplay.types import Contract, Deal, Denom, Penalty, Player, Vul

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
STATIC_DIR = Path(__file__).parent / "static"

_RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_RANK_CHAR_TO_IDX = {r: i for i, r in enumerate(_RANK_NAMES)}   # "2"→0 … "A"→12
_RANK_ORDER_HIGH  = list(reversed(_RANK_NAMES))                  # A K Q … 2
_SUIT_KEYS = ["S", "H", "D", "C"]          # suit_idx 0-3 in deal.py
_DIR_NAMES = ["N", "E", "S", "W"]
_DENOM_NAMES = ["C", "D", "H", "S", "NT"]  # matches auction.py denom indices

_VUL = {"none": Vul.none, "ns": Vul.ns, "ew": Vul.ew, "both": Vul.both}
_ENDPLAY_DENOMS = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
_ENDPLAY_PLAYERS = [Player.north, Player.east, Player.south, Player.west]

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
_model_cache: dict = {}
_deal_cache: dict = {}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_agent(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    if model_name == "random":
        agent = RandomAgent()
    else:
        if model_name.startswith(_CRITIC_PREFIX):
            raise HTTPException(400, f"'{model_name}' is a centralized critic — not usable for inference")
        path = CHECKPOINTS_DIR / f"{model_name}.pt"
        if not path.exists():
            raise HTTPException(404, f"Checkpoint '{model_name}' not found")
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if not any(k.startswith("policy_head") for k in state_dict):
            raise HTTPException(400, f"'{model_name}' has no policy_head — not a BiddingNet checkpoint")
        if "policy_head.weight" in state_dict:
            raise HTTPException(400,
                f"'{model_name}' uses a legacy single-Linear head format that is "
                "no longer compatible with the current architecture")
        embed_dim  = state_dict["bid_emb.weight"].shape[1]

        # --- Hand encoder ---
        if "hand_enc.suit_linear.weight" in state_dict:
            hand_encoder = "suit"
            hidden     = state_dict["hand_enc.agg.0.weight"].shape[0]
            mlp_layers = sum(1 for k in state_dict
                             if k.startswith("hand_enc.agg.") and k.endswith(".weight")) - 1
        else:
            hand_encoder = "mlp"
            hidden     = state_dict["hand_enc.0.weight"].shape[0]
            mlp_layers = sum(1 for k in state_dict
                             if k.startswith("hand_enc.") and k.endswith(".weight")) - 1

        # --- Auction encoder ---
        if any(k.startswith("transformer.") for k in state_dict):
            auction_encoder = "transformer"
            lstm_layers = sum(1 for k in state_dict
                              if k.startswith("transformer.layers.")
                              and k.endswith("self_attn.in_proj_weight"))
        else:
            auction_encoder = "lstm"
            lstm_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))

        net = BiddingNet(hidden=hidden, embed_dim=embed_dim,
                         mlp_layers=mlp_layers, lstm_layers=lstm_layers,
                         hand_encoder=hand_encoder, auction_encoder=auction_encoder)
        net.load_state_dict(state_dict)
        net.eval()
        agent = NNAgent(net)
    _model_cache[model_name] = agent
    return agent


def _hand_to_cards(vec: np.ndarray) -> dict:
    """52-dim binary → {suit: [rank_str, ...]} sorted high to low."""
    return {
        suit: [_RANK_NAMES[r] for r in range(12, -1, -1) if vec[s_idx * 13 + r]]
        for s_idx, suit in enumerate(_SUIT_KEYS)
    }


def _build_dd_dict(raw_table) -> dict:
    """endplay DDS table → JSON-serialisable nested dict."""
    return {
        d_name: {
            p_name: int(raw_table[_ENDPLAY_DENOMS[d_idx], _ENDPLAY_PLAYERS[p_idx]])
            for p_idx, p_name in enumerate(_DIR_NAMES)
        }
        for d_idx, d_name in enumerate(_DENOM_NAMES)
    }


def _best_ns_contract(raw_table, vulnerability: str):
    """Best N-S score and contract string given a DDS tricks table."""
    vul = _VUL[vulnerability]
    best_score, best_str = 0, "Pass"
    for d_idx, d in enumerate(_ENDPLAY_DENOMS):
        for dir_idx in (0, 2):  # North=0, South=2
            declarer = _ENDPLAY_PLAYERS[dir_idx]
            tricks = raw_table[d, declarer]
            for level in range(1, 8):
                if tricks - (level + 6) >= 0:
                    c = Contract(
                        level=level, denom=d, declarer=declarer,
                        penalty=Penalty.passed, result=tricks - (level + 6),
                    )
                    score = c.score(vul)
                    if score > best_score:
                        best_score = score
                        best_str = f"{level}{_DENOM_NAMES[d_idx]} by {_DIR_NAMES[dir_idx]}"
    return best_score, best_str


def _rebuild_auction(bids: list) -> AuctionState:
    a = AuctionState(dealer=0, uncontested=True)
    for b in bids:
        a.apply(b)
    return a


def _suggest_bid(agent, hand_vec: np.ndarray, auction: AuctionState, top_k: int = 3):
    """Return agent's greedy bid and the top-k distribution over valid bids.

    Returns (best_bid_idx, [(bid_idx, bid_name, probability), ...]).
    For RandomAgent the distribution is uniform over valid bids.
    """
    current = auction.current_player()
    valid = auction.valid_bids()

    if isinstance(agent, RandomAgent):
        p = 1.0 / len(valid)
        dist = [(b, bid_name(b), round(p, 4)) for b in valid[:top_k]]
        return valid[0], dist

    seq = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
    seq[: len(auction.bids)] = auction.bids

    hand_t = torch.tensor(hand_vec, dtype=torch.float32).unsqueeze(0)
    seq_t  = torch.tensor(seq,      dtype=torch.int64).unsqueeze(0)
    dir_t  = torch.tensor([current], dtype=torch.int64)
    mask   = torch.zeros(1, NUM_BIDS, dtype=torch.bool)
    for b in valid:
        mask[0, b] = True

    with torch.no_grad():
        log_probs, _ = agent.net(hand_t, seq_t, dir_t, mask)

    probs     = log_probs[0].exp()
    top       = probs.topk(min(top_k, len(valid)))
    best_idx  = int(top.indices[0].item())
    dist      = [
        (int(idx.item()), bid_name(int(idx.item())), round(float(p.item()), 4))
        for idx, p in zip(top.indices, top.values)
    ]
    return best_idx, dist


# ---------------------------------------------------------------------------
# Manual deal helpers
# ---------------------------------------------------------------------------

def _parse_hand_input(suit_dict: dict) -> np.ndarray:
    """Parse {S:str, H:str, D:str, C:str} card strings into a 52-dim binary vector.

    Rank characters: A K Q J T 9 8 7 6 5 4 3 2 (case-insensitive, spaces ignored).
    Suit order in vector: s_idx 0=S, 1=H, 2=D, 3=C (matches _SUIT_KEYS / hand_to_vector).
    """
    vec = np.zeros(52, dtype=np.float32)
    for s_idx, suit in enumerate(_SUIT_KEYS):
        ranks_str = suit_dict.get(suit, "").upper().replace(" ", "")
        for ch in ranks_str:
            if ch not in _RANK_CHAR_TO_IDX:
                raise ValueError(f"Unknown rank character '{ch}' in suit {suit}")
            bit = s_idx * 13 + _RANK_CHAR_TO_IDX[ch]
            if vec[bit]:
                raise ValueError(f"Duplicate card {ch}{suit}")
            vec[bit] = 1.0
    return vec


def _vec_to_pbn_hand(vec: np.ndarray) -> str:
    """52-dim binary vector → PBN hand string 'S.H.D.C' (ranks high→low)."""
    suits = []
    for s_idx in range(4):          # 0=S, 1=H, 2=D, 3=C
        suits.append("".join(
            r for r in _RANK_ORDER_HIGH
            if vec[s_idx * 13 + _RANK_CHAR_TO_IDX[r]]
        ))
    return ".".join(suits)


def _build_ep_deal(hand_vecs: np.ndarray) -> Deal:
    """Build an endplay Deal from (4, 52) hand vectors (N, E, S, W order)."""
    # PBN format: "N:N_hand E_hand S_hand W_hand"
    pbn = "N:" + " ".join(_vec_to_pbn_hand(hand_vecs[i]) for i in range(4))
    return Deal(pbn)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------
class DealRequest(BaseModel):
    model: str = "net_final"
    vulnerability: str = "none"


class SuggestRequest(BaseModel):
    deal_id: str
    bids: List[int] = []


class ManualDealRequest(BaseModel):
    model: str = "net_final"
    vulnerability: str = "none"
    # {N: {S: "AKQ", H: "JT", D: "98765", C: "432"}, E: ..., S: ..., W: ...}
    hands: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
_CRITIC_PREFIX = "critic_"   # CentralizedCritic checkpoints — not usable for inference


@app.get("/models")
async def list_models():
    # Name-based filter: excludes critic checkpoints without touching the files
    # (important on network filesystems where scanning all files would be slow).
    # Old-format policy nets (bare Linear heads) are still listed but _load_agent
    # will reject them with a clear error if selected.
    names = sorted(
        p.stem for p in CHECKPOINTS_DIR.glob("*.pt")
        if not p.stem.startswith(_CRITIC_PREFIX)
    )
    return {"models": ["random"] + names}


@app.post("/deal")
async def new_deal(req: DealRequest):
    if req.vulnerability not in _VUL:
        raise HTTPException(400, "Invalid vulnerability")

    deal = random_deal(req.vulnerability)
    raw_tables = dds.calc_all_tables([deal.endplay_deal])
    raw_table = raw_tables[0]

    dd_dict = _build_dd_dict(raw_table)
    par_score, par_contract = _best_ns_contract(raw_table, req.vulnerability)
    hands = {_DIR_NAMES[i]: _hand_to_cards(deal.hands[i]) for i in range(4)}

    deal_id = str(uuid.uuid4())
    _deal_cache[deal_id] = {
        "deal": deal,
        "raw_table": raw_table,
        "model": req.model,
        "vulnerability": req.vulnerability,
    }
    _load_agent(req.model)  # warm up the model

    return {
        "deal_id": deal_id,
        "vulnerability": req.vulnerability,
        "hands": hands,
        "par_score": par_score,
        "par_contract": par_contract,
        "dd_table": dd_dict,
    }


@app.post("/deal/manual")
async def manual_deal(req: ManualDealRequest):
    """Load a user-specified deal.

    Expects ``hands`` = {N: {S:str, H:str, D:str, C:str}, E:…, S:…, W:…}.
    Ranks are A K Q J T 9 8 7 6 5 4 3 2 (case-insensitive, spaces optional).
    Returns the same shape as POST /deal.
    """
    if req.vulnerability not in _VUL:
        raise HTTPException(400, "Invalid vulnerability")

    dirs = ["N", "E", "S", "W"]
    if not all(d in req.hands for d in dirs):
        raise HTTPException(400, "Must provide hands for N, E, S, and W")

    # Parse and validate individual hands
    hand_vecs = []
    for d in dirs:
        try:
            vec = _parse_hand_input(req.hands[d])
        except ValueError as exc:
            raise HTTPException(400, f"Hand {d}: {exc}") from exc
        n = int(vec.sum())
        if n != 13:
            raise HTTPException(400, f"Hand {d} has {n} cards (need 13)")
        hand_vecs.append(vec)

    # Check the 52 cards are covered exactly once
    total = np.stack(hand_vecs).sum(axis=0)
    if not np.all(total == 1):
        dupes = [
            f"{_RANK_NAMES[i % 13]}{_SUIT_KEYS[i // 13]}"
            for i in np.where(total != 1)[0]
        ]
        raise HTTPException(400, f"Duplicate or missing cards: {', '.join(dupes)}")

    hands_np = np.stack(hand_vecs)   # (4, 52)
    ep_deal  = _build_ep_deal(hands_np)

    # Re-derive hand vectors from endplay to ensure consistency with hand_to_vector
    hands_np = np.stack([hand_to_vector(ep_deal[p]) for p in _ENDPLAY_PLAYERS])
    deal = BridgeDeal(endplay_deal=ep_deal, hands=hands_np, vulnerability=req.vulnerability)

    raw_tables = dds.calc_all_tables([ep_deal])
    raw_table  = raw_tables[0]

    dd_dict = _build_dd_dict(raw_table)
    par_score, par_contract = _best_ns_contract(raw_table, req.vulnerability)
    hands_out = {_DIR_NAMES[i]: _hand_to_cards(hands_np[i]) for i in range(4)}

    deal_id = str(uuid.uuid4())
    _deal_cache[deal_id] = {
        "deal": deal,
        "raw_table": raw_table,
        "model": req.model,
        "vulnerability": req.vulnerability,
    }
    _load_agent(req.model)

    return {
        "deal_id":      deal_id,
        "vulnerability": req.vulnerability,
        "hands":        hands_out,
        "par_score":    par_score,
        "par_contract": par_contract,
        "dd_table":     dd_dict,
    }


@app.post("/suggest")
async def suggest_next(req: SuggestRequest):
    cached = _deal_cache.get(req.deal_id)
    if cached is None:
        raise HTTPException(404, "Deal not found — please request a new deal")

    deal: BridgeDeal = cached["deal"]
    raw_table = cached["raw_table"]
    vulnerability: str = cached["vulnerability"]
    agent = _load_agent(cached["model"])

    auction = _rebuild_auction(req.bids)

    if auction.is_complete():
        achieved = achieved_ns_score(raw_table, auction, vulnerability)
        par_score, par_contract = _best_ns_contract(raw_table, vulnerability)
        imp = imps(achieved - par_score)

        contract_info = auction.final_contract()
        if contract_info is None:
            contract_str = "Passed out"
        else:
            level, d_idx, dec_dir, dbl, rdbl = contract_info
            suf = "XX" if rdbl else ("X" if dbl else "")
            contract_str = f"{level}{_DENOM_NAMES[d_idx]}{suf} by {_DIR_NAMES[dec_dir]}"

        return {
            "is_complete": True,
            "final_contract": contract_str,
            "achieved_score": achieved,
            "par_score": par_score,
            "par_contract": par_contract,
            "imp": imp,
        }

    current  = auction.current_player()
    hand_vec = deal.hands[current]
    bid_idx, dist = _suggest_bid(agent, hand_vec, auction)

    return {
        "is_complete":    False,
        "current_player": _DIR_NAMES[current],
        "bid_idx":        bid_idx,
        "bid_name":       bid_name(bid_idx),
        "distribution":   [{"bid_idx": b, "bid_name": n, "prob": p} for b, n, p in dist],
    }


class ParAnalysisRequest(BaseModel):
    deal_id: str
    k: int = 20   # number of EW re-deals to sample


@app.post("/par_analysis")
async def par_analysis(req: ParAnalysisRequest):
    """Rank all NS contracts by expected score over k sampled EW completions.

    For each (level, denom, declarer∈{N,S}) we average the score across k
    re-dealt EW hands using double-dummy play, giving a robust estimate of
    how good that contract is given only the NS cards.  Returns the top 3
    contracts by expected score, plus their score on the actual deal.
    """
    cached = _deal_cache.get(req.deal_id)
    if cached is None:
        raise HTTPException(404, "Deal not found")

    deal: BridgeDeal = cached["deal"]
    raw_table        = cached["raw_table"]
    vulnerability    = cached["vulnerability"]
    vul              = _VUL[vulnerability]
    rng              = np.random.default_rng()

    # Sample k EW completions and solve them all in one batched DDS call.
    ew_deals = resample_ew(deal.endplay_deal, req.k, rng)
    sampled_tables = _calc_all_tables_chunked(ew_deals)

    # Accumulate scores for every (denom, declarer_ns, level) triple.
    # Key: (d_idx, dir_idx, level) → list of per-sample scores.
    tally: dict = {}
    for table in sampled_tables:
        for d_idx, d in enumerate(_ENDPLAY_DENOMS):
            for dir_idx in (0, 2):               # North=0, South=2
                declarer = _ENDPLAY_PLAYERS[dir_idx]
                tricks   = table[d, declarer]
                for level in range(1, 8):
                    result = tricks - (level + 6)
                    c      = Contract(level=level, denom=d, declarer=declarer,
                                      penalty=Penalty.passed, result=result)
                    score  = c.score(vul)
                    key    = (d_idx, dir_idx, level)
                    if key not in tally:
                        tally[key] = []
                    tally[key].append(score)

    # Compute expected score and score on the actual deal for every contract.
    contracts = []
    for (d_idx, dir_idx, level), scores in tally.items():
        d        = _ENDPLAY_DENOMS[d_idx]
        declarer = _ENDPLAY_PLAYERS[dir_idx]
        tricks   = raw_table[d, declarer]
        result   = tricks - (level + 6)
        c_actual = Contract(level=level, denom=d, declarer=declarer,
                            penalty=Penalty.passed, result=result)
        actual_score   = c_actual.score(vul)
        expected_score = float(np.mean(scores))
        contracts.append({
            "contract":       f"{level}{_DENOM_NAMES[d_idx]} by {_DIR_NAMES[dir_idx]}",
            "expected_score": round(expected_score, 1),
            "actual_score":   actual_score,
        })

    contracts.sort(key=lambda x: x["expected_score"], reverse=True)
    return {"contracts": contracts[:3], "k": req.k}
