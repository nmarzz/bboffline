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
from environment.deal import BridgeDeal, random_deal
from environment.scoring import achieved_ns_score, imps

import endplay.dds as dds
from endplay.types import Contract, Denom, Penalty, Player, Vul

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
STATIC_DIR = Path(__file__).parent / "static"

_RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
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
        path = CHECKPOINTS_DIR / f"{model_name}.pt"
        if not path.exists():
            raise HTTPException(404, f"Checkpoint '{model_name}' not found")
        net = BiddingNet()
        net.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
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


def _suggest_bid(agent, hand_vec: np.ndarray, auction: AuctionState) -> int:
    """Return agent's bid suggestion for the current player.

    Uses stochastic sampling (matching training), so the suggestion reflects
    the full policy distribution rather than always collapsing to the argmax.
    """
    current = auction.current_player()
    valid = auction.valid_bids()

    if isinstance(agent, RandomAgent):
        import random
        return random.choice(valid)

    # NNAgent — sample from the masked policy distribution
    seq = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
    seq[: len(auction.bids)] = auction.bids

    hand_t = torch.tensor(hand_vec, dtype=torch.float32).unsqueeze(0)
    seq_t = torch.tensor(seq, dtype=torch.int64).unsqueeze(0)
    dir_t = torch.tensor([current], dtype=torch.int64)
    mask = torch.zeros(1, NUM_BIDS, dtype=torch.bool)
    for b in valid:
        mask[0, b] = True

    with torch.no_grad():
        log_probs, _ = agent.net(hand_t, seq_t, dir_t, mask)

    from torch.distributions import Categorical
    return int(Categorical(logits=log_probs[0]).sample().item())


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------
class DealRequest(BaseModel):
    model: str = "net_final"
    vulnerability: str = "none"


class SuggestRequest(BaseModel):
    deal_id: str
    bids: List[int] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/models")
async def list_models():
    names = sorted(p.stem for p in CHECKPOINTS_DIR.glob("*.pt"))
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

    current = auction.current_player()
    hand_vec = deal.hands[current]
    bid_idx = _suggest_bid(agent, hand_vec, auction)

    return {
        "is_complete": False,
        "current_player": _DIR_NAMES[current],
        "bid_idx": bid_idx,
        "bid_name": bid_name(bid_idx),
    }
