from __future__ import annotations
from typing import Dict, List, Tuple
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, random
from ..models import Card, GladiatorType
from ..observations import PlayerView

def _argmin_type(totals: Dict[GladiatorType, int]) -> GladiatorType:
    # deterministic tie-breaking by enum order
    best_t, best_v = None, None
    for t in GladiatorType:
        v = totals.get(t, 0)
        if best_v is None or v < best_v:
            best_t, best_v = t, v
    return best_t

def _open_lane_type(totals: Dict[GladiatorType, int]) -> GladiatorType:
    # choose a type with zero total if possible; else choose the least-funded
    zeros = [t for t in GladiatorType if totals.get(t, 0) == 0]
    return zeros[0] if zeros else _argmin_type(totals)

def estimate_future_representation_open_lane(
    rng: random.Random,
    base_totals: Dict[GladiatorType, int],
    existing_bet_amounts: List[int],        # actual bets already placed on the board
    remaining_behind: List[int],            # coins-behind of players yet to act (in seat order)
    my_type: GladiatorType,                 # the type we’re evaluating for our candidate 'a'
    my_a: int,                              # candidate bet amount 'a'
    mean_floor: float = 1.5,                # minimum average baseline
    std_frac: float = 0.25,                 # bell-curve std as a fraction of mean
    max_bankroll_frac: float = 0.40,        # cap predicted bet at 40% of that player’s bankroll
) -> Tuple[Dict[GladiatorType, int], Dict[GladiatorType, int], List[int]]:
    """
    Sequentially predict the remaining players' representation bets:
      - Type: always an open lane (no current bets); if none, the least-funded type.
      - Amount: random around the running average of observed bets (bell curve),
                clamped to [1, cap] where cap = max_bankroll_frac * their behind.
    Returns (added_by_others, final_totals, predicted_amounts_appended).
    """
    totals = dict(base_totals)
    totals[my_type] = totals.get(my_type, 0) + my_a

    bet_amounts = list(existing_bet_amounts)
    bet_amounts.append(my_a)

    added: Dict[GladiatorType, int] = {t: 0 for t in GladiatorType}

    for behind in remaining_behind:
        # running mean as the "most probable" amount
        mean_now = max(mean_floor, sum(bet_amounts) / max(1, len(bet_amounts)))
        sigma = max(0.5, std_frac * mean_now)  # bell curve dispersion
        cap = max(1, int(max_bankroll_frac * behind))

        # sample a normal around mean; clamp and round
        a_i = int(round(rng.gauss(mean_now, sigma)))
        a_i = max(1, min(cap, a_i))

        t_i = _open_lane_type(totals)

        totals[t_i] = totals.get(t_i, 0) + a_i
        added[t_i] += a_i
        bet_amounts.append(a_i)

    return added, totals, bet_amounts
