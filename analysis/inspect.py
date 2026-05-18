"""Analysis and inspection tools.

Usage:
  python -m analysis.inspect --checkpoint checkpoints/net_final.pt --n 20
  python -m analysis.inspect --random --n 1000   # baseline IMP stats
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.deal import random_deal
from environment.auction import AuctionState, NUM_BIDS, MAX_AUCTION_LEN, bid_name
from environment.scoring import batch_expected_imp_rewards, ns_par_score, achieved_ns_score, _calc_all_tables_chunked
import endplay.dds as dds
from agents.random_agent import RandomAgent
from agents.nn_agent import BiddingNet, NNAgent

_DENOM_NAMES = ["C", "D", "H", "S", "NT"]
_DIR_NAMES = ["N", "E", "S", "W"]


def run_inspection_episode(agent, vulnerability="none", print_auction=True, ew_samples=20):
    """Run one episode and optionally print a human-readable summary."""
    deal = random_deal(vulnerability)
    auction = AuctionState(uncontested=True)

    while not auction.is_complete():
        player = auction.current_player()
        valid = auction.valid_bids()
        hand = deal.hands[player]

        seq = auction.to_sequence()
        padded = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
        padded[: len(seq)] = seq

        if isinstance(agent, NNAgent):
            action = agent.act(hand, padded, player, valid)
        else:
            action = agent.act(hand, auction)

        auction.apply(action)

    # Use the actual deal for display scores, expected reward for the IMP figure
    table = _calc_all_tables_chunked([deal.endplay_deal])[0]
    par = ns_par_score(table, vulnerability)
    achieved = achieved_ns_score(table, auction, vulnerability)
    (reward,), _ = batch_expected_imp_rewards(
        [deal.endplay_deal], [auction], vulnerability, k=ew_samples
    )

    if print_auction:
        # Print hands
        suits = ["♠", "♥", "♦", "♣"]
        for i, name in enumerate(_DIR_NAMES):
            hand_vec = deal.hands[i]
            hand_str = ""
            for s in range(4):
                cards = []
                for r in range(12, -1, -1):  # A down to 2
                    if hand_vec[s * 13 + r] == 1:
                        rank_chars = "23456789TJQKA"
                        cards.append(rank_chars[r])
                hand_str += f"{suits[s]}{''.join(cards) if cards else '-'} "
            print(f"  {name}: {hand_str}")
        print(f"  Auction: {auction}")
        cf = auction.final_contract()
        if cf:
            level, denom_idx, declarer, dbl, rdbl = cf
            suffix = "XX" if rdbl else ("X" if dbl else "")
            contract_str = f"{level}{_DENOM_NAMES[denom_idx]}{suffix} by {_DIR_NAMES[declarer]}"
        else:
            contract_str = "Passed out"
        print(f"  Contract: {contract_str}  |  N-S score: {achieved}  |  Par: {par}  |  IMP: {reward:+.0f}")
        print()

    return reward, par, achieved


def run_stats(agent, n=1000, vulnerability="none", ew_samples=20, batch_size=100):
    """Evaluate n deals in batches of batch_size, using batched DDS for speed."""
    all_imps = []
    all_pars = []

    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        deals, auctions = [], []
        for _ in range(count):
            deal = random_deal(vulnerability)
            auction = AuctionState(uncontested=True)
            while not auction.is_complete():
                player = auction.current_player()
                valid = auction.valid_bids()
                hand = deal.hands[player]
                seq = auction.to_sequence()
                padded = np.full(MAX_AUCTION_LEN, -1, dtype=np.int64)
                padded[: len(seq)] = seq
                if isinstance(agent, NNAgent):
                    action = agent.act(hand, padded, player, valid)
                else:
                    action = agent.act(hand, auction)
                auction.apply(action)
            deals.append(deal)
            auctions.append(auction)

        imps_batch, pars_batch = batch_expected_imp_rewards(
            [d.endplay_deal for d in deals], auctions, vulnerability, k=ew_samples
        )
        all_imps.extend(imps_batch)
        all_pars.extend(pars_batch)

    imps_arr = np.array(all_imps)
    pars_arr = np.array(all_pars)
    print(f"Over {n} deals (k={ew_samples} E-W samples each):")
    print(f"  Mean IMP:     {imps_arr.mean():+.3f}  (std {imps_arr.std():.2f})")
    print(f"  Median IMP:   {np.median(imps_arr):+.1f}")
    print(f"  Mean par pts: {pars_arr.mean():.1f}")
    print(f"  IMP > 0:      {(imps_arr > 0).mean()*100:.1f}%")
    print(f"  IMP == 0:     {(imps_arr == 0).mean()*100:.1f}%")
    print(f"  IMP < 0:      {(imps_arr < 0).mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--stats", action="store_true",
                        help="Print aggregate stats instead of sample auctions")
    parser.add_argument("--vulnerability", default="none",
                        choices=["none", "ns", "ew", "both"])
    args = parser.parse_args()

    if args.random or args.checkpoint is None:
        agent = RandomAgent()
        label = "Random agent"
    else:
        net = BiddingNet()
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        net.eval()
        agent = NNAgent(net)
        label = f"NN agent ({args.checkpoint})"

    print(f"=== {label} ===\n")

    if args.stats:
        run_stats(agent, n=args.n, vulnerability=args.vulnerability)
    else:
        for i in range(args.n):
            print(f"--- Deal {i+1} ---")
            run_inspection_episode(agent, args.vulnerability, print_auction=True)


if __name__ == "__main__":
    main()
