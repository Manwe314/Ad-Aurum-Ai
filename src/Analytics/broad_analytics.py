# BBB/src/broad_analytics.py
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import random

# --- Engine / helpers imports (robust) ----------------------------------------
try:
    # your sim helpers (parallel game runner + traits builder from analytics.py)
    from Analytics.simple_analitics import run_many_games_avg_round_coins_parallel  # type: ignore
except Exception:
    from .analytics import run_many_games_avg_round_coins_parallel  # type: ignore

try:
    from bbb.brains.base import Traits  # type: ignore
except Exception:
    from .brains.base import Traits  # type: ignore

# Quiet engine-side logging/printing if present
try:
    from bbb.globals import LOGGING, GAME_ENGINE_PIRINTS  # type: ignore
    LOGGING = False  # type: ignore
    GAME_ENGINE_PIRINTS = False  # type: ignore
except Exception:
    pass

# ---------- Trait utils -------------------------------------------------------

TRAIT_ORDER: List[str] = [
    "aggressiveness",
    "risk_tolerance",
    "tempo",
    "bluffiness",
    "stubbornness",
    "domination_drive",
    "herding",
    "ev_adherence",
    "exploration",
]

def _traits_field_names() -> set:
    try:
        return {f.name for f in fields(Traits)}
    except Exception:
        return set(TRAIT_ORDER)

def _vec_to_traits(vec9: np.ndarray, round_to: int = 5) -> Traits:
    """
    Map 9 floats in [0,1] -> integer trait values.
      - All traits except 'exploration' are in 0..100 and rounded to nearest `round_to` (default 5).
      - 'exploration' is capped to 0..10 and rounded to nearest integer.
    """
    if len(vec9) != len(TRAIT_ORDER):
        raise ValueError(f"vec9 must have length {len(TRAIT_ORDER)}; got {len(vec9)}")

    vals: Dict[str, int] = {}
    for i, name in enumerate(TRAIT_ORDER):
        x = float(vec9[i])
        # clamp input to [0,1]
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0

        if name == "exploration":
            # scale to 0..10, round to nearest int, clamp
            v = int(round(x * 10.0))
            v = max(0, min(10, v))
        else:
            # scale to 0..100, round to nearest `round_to`, clamp
            v = int(round(x * 100.0))
            v = max(0, min(100, v))
            if round_to and round_to > 1:
                v = int(round(v / round_to) * round_to)
                v = max(0, min(100, v))

        vals[name] = v

    # ensure compatibility if Traits has extra/missing fields
    payload = {k: vals.get(k, 50) for k in _traits_field_names()}
    return Traits(**payload)  # type: ignore


# ---------- Sobol sampling ----------------------------------------------------

def _sobol(n: int, d: int, seed: Optional[int]) -> np.ndarray:
    """
    Return n x d matrix in [0,1). Uses SciPy Sobol if available; falls back to RNG.
    """
    try:
        from scipy.stats.qmc import Sobol  # type: ignore
        eng = Sobol(d=d, scramble=True, seed=seed)
        try:
            x = eng.random(n)                          # SciPy >=1.9 allows any n
        except ValueError:
            # older SciPy requires power-of-two; pad then truncate
            m = 1
            while m < n:
                m <<= 1
            x = eng.random(m)[:n]
        return x
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, d))

# ---------- Core per-point evaluator -----------------------------------------

def _wins_for_point(
    traits4: List[Traits],
    *,
    repeats: int,
    games_per_repeat: int,
    starting_coins: int,
    num_battles: int,
    seed: Optional[int],
    workers: int,
    chunk_size: int,
) -> Tuple[List[int], int]:
    """
    Run `repeats` replicates for this 4-player trait set.
    Each replicate permutes seating, then runs `games_per_repeat` games in parallel.
    Returns:
        wins_per_player_index[0..3], total_games_per_player
    """
    rng = random.Random(seed)
    wins_per_player = [0, 0, 0, 0]
    total_games = 0

    for r in range(repeats):
        # permute seating to reduce seat bias
        perm = list(range(4))
        rng.shuffle(perm)
        seats = [traits4[perm[0]], traits4[perm[1]], traits4[perm[2]], traits4[perm[3]]]

        avgs, wins, ties = run_many_games_avg_round_coins_parallel(
            traits_by_seat=seats,
            games=games_per_repeat,
            starting_coins=starting_coins,
            num_battles=num_battles,
            seed=rng.randint(0, 2**31 - 1),
            workers=workers if workers and workers > 1 else 1,
            chunk_size=chunk_size if chunk_size and chunk_size > 0 else None,
        )
        # map seat wins back to the original player indices
        for s in range(4):
            pid = perm[s]
            wins_per_player[pid] += int(wins.get(f"P{s+1}", 0))
        total_games += games_per_repeat

    return wins_per_player, total_games

# ---------- CSV writer --------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def _write_header(w: csv.writer) -> None:
    cols = [
        "design_id",
        "player_id",
        "repeats",
        "games_per_repeat",
        "total_games",
        "num_battles",
        # traits (P_i)
    ]
    cols.extend([f"trait_{name}" for name in TRAIT_ORDER])
    cols.append("winrate_percent")
    cols.append("wins")
    w.writerow(cols)

def _write_rows_for_point(
    w: csv.writer,
    design_id: int,
    traits4: List[Traits],
    wins_per_player: List[int],
    total_games_per_player: int,
    *,
    repeats: int,
    games_per_repeat: int,
    num_battles: int,
) -> None:
    for pid in range(4):
        t = traits4[pid]
        wins = wins_per_player[pid]
        wr = (100.0 * wins / total_games_per_player) if total_games_per_player > 0 else 0.0
        row = [
            design_id,
            pid,
            repeats,
            games_per_repeat,
            total_games_per_player,
            num_battles,
        ]
        row.extend([getattr(t, name, 50) for name in TRAIT_ORDER])
        row.extend([f"{wr:.3f}", wins])
        w.writerow(row)

# ---------- CLI command -------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    points = int(args.points)
    if points <= 0:
        raise ValueError("--points must be > 0")
    repeats = int(args.repeats)
    games_per_repeat = int(args.games)
    round_to = int(args.round_to)

    # sample 36-D points in [0,1)
    X = _sobol(points, 36, seed=args.seed)

    _ensure_dir(args.outcsv)
    with open(args.outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        _write_header(w)

        parent_rng = random.Random(args.seed)

        for i in range(points):
            # split into 4 chunks of 9 dims → Traits
            p = X[i]
            traits4 = [
                _vec_to_traits(p[0:9], round_to),
                _vec_to_traits(p[9:18], round_to),
                _vec_to_traits(p[18:27], round_to),
                _vec_to_traits(p[27:36], round_to),
            ]

            wins_per_player, total_games = _wins_for_point(
                traits4,
                repeats=repeats,
                games_per_repeat=games_per_repeat,
                starting_coins=args.starting_coins,
                num_battles=args.num_battles,
                seed=parent_rng.randint(0, 2**31 - 1),
                workers=args.workers,
                chunk_size=args.chunk_size,
            )

            _write_rows_for_point(
                w,
                design_id=i + 1,
                traits4=traits4,
                wins_per_player=wins_per_player,
                total_games_per_player=repeats * games_per_repeat,
                repeats=repeats,
                games_per_repeat=games_per_repeat,
                num_battles=args.num_battles,
            )

            if args.verbose:
                print(f"[{i+1}/{points}] wrote rows for design {i+1}")

    print(f"Saved: {os.path.abspath(args.outcsv)}")
    return 0

# ---------- Parser / Main -----------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BBB.broad_analytics",
        description="Broad trait sampling via Sobol; outputs per-player traits and winrates."
    )
    p.add_argument("--points", type=int, required=True, help="Number of Sobol design points to sample (each is 4 players).")
    p.add_argument("--repeats", type=int, default=4, help="Replicates per point (different RNG + permuted seating).")
    p.add_argument("--games", type=int, required=True, help="Games per replicate (parallelized internally).")
    p.add_argument("--starting-coins", type=int, required=True, help="Starting coins per player.")
    p.add_argument("--num-battles", type=int, default=3, help="Battles per round (default: 3).")
    p.add_argument("--round-to", type=int, default=5, help="Round trait values to nearest k (default: 5).")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    p.add_argument("--workers", type=int, default=0, help="Processes for inner game sims (0/1 = no parallelism).")
    p.add_argument("--chunk-size", type=int, default=0, help="Games per task (0 = auto ~8 chunks/worker).")
    p.add_argument("--outcsv", default="broad_samples.csv", help="Output CSV path.")
    p.add_argument("--verbose", action="store_true")
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_sample(args)

if __name__ == "__main__":
    raise SystemExit(main())


