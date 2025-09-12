# BBB/src/train_pool.py
from __future__ import annotations

import argparse, json, os, random, copy
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# reuse your training constants
from bbb.globals import WW_VARS_TRAINING
from Training.experiments import run_delta_parallel  # adjust import path if needed

# -------- Search space (copied from your trainer) ----------
# Build bounds from current WW_VARS_TRAINING keys
BOUNDS: Dict[str, Tuple[float, float]] = {k: (0.0, 2.0) for k in WW_VARS_TRAINING.keys()}
DIMENSIONS = [Real(lo, hi, name=k) for k, (lo, hi) in BOUNDS.items()]
PARAM_NAMES = list(BOUNDS.keys())

# -------- Pool I/O ----------------------------------------
def load_pool(path: str) -> List[Dict[str, float]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data
    return []

def save_pool(path: str, pool: Sequence[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(pool), f, indent=2)

# -------- One BO run against a pool ------------------------
def bo_once_against_pool(
    *,
    pool: Sequence[Dict[str, float]],
    pool_sample: int,
    n_calls: int,
    n_init: int,
    seed: Optional[int],
    # delta params
    randomizations: int,
    games_per_alpha: int,
    cycles_per_position: int,
    num_battles: int,
    starting_coins: int,
    workers: Optional[int],
) -> Tuple[Dict[str, float], float]:
    """
    Run one BO session. Objective samples 'pool_sample' opponent weight sets
    from the pool and averages their delta scores for robustness.
    Returns (best_params, best_score).
    """
    rng = random.Random(seed)

    # shared mutable copy the objective will write
    ww_training = copy.deepcopy(WW_VARS_TRAINING)

    @use_named_args(dimensions=DIMENSIONS)
    def objective(**params):
        # update training weights for tracked player
        ww_training.update(params)

        if not pool:
            # no opponents in pool yet -> run vs default (None)
            res = run_delta_parallel(
                randomizations=randomizations,
                games_per_alpha=games_per_alpha,
                cycles_per_position=cycles_per_position,
                num_battles=num_battles,
                starting_coins=starting_coins,
                workers=workers,
                opp_pool=None,  # default opponents
            )
            return -float(res.score)

        # average across a few random opponents from pool
        k = min(max(1, pool_sample), len(pool))
        choices = [pool[rng.randrange(len(pool))] for _ in range(k)]
        scores = []
        for opp_w in choices:
            res = run_delta_parallel(
                randomizations=randomizations,
                games_per_alpha=games_per_alpha,
                cycles_per_position=cycles_per_position,
                num_battles=num_battles,
                starting_coins=starting_coins,
                workers=workers,
                opp_weights= opp_w,  # pick this one in workers
            )
            scores.append(float(res.score))
        mean_score = sum(scores) / len(scores)
        return -mean_score

    result = gp_minimize(
        func=objective,
        dimensions=DIMENSIONS,
        acq_func="EI",
        n_calls=n_calls,
        n_initial_points=n_init,
        random_state=seed,
        verbose=True,
    )
    best_score = -float(result.fun)
    best_params = dict(zip(PARAM_NAMES, result.x))
    return best_params, best_score

# -------- CLI loop: repeat BO and grow the pool ------------
def cmd_train_pool(args: argparse.Namespace) -> int:
    pool = load_pool(args.pool)

    for it in range(1, args.iters + 1):
        print(f"\n=== BO iteration {it}/{args.iters} (pool size={len(pool)}) ===")
        best_params, best_score = bo_once_against_pool(
            pool=pool,
            pool_sample=args.pool_sample,
            n_calls=args.n_calls,
            n_init=args.n_init,
            seed=(args.seed + it if args.seed is not None else None),
            randomizations=args.randomizations,
            games_per_alpha=args.games_per_alpha,
            cycles_per_position=args.cycles_per_position,
            num_battles=args.num_battles,
            starting_coins=args.starting_coins,
            workers=args.workers if args.workers and args.workers > 1 else None,
        )
        # add to pool and keep only top-K if requested
        entry = dict(best_params)  # weights only
        entry["_score"] = best_score
        pool.append(entry)
        # keep the best 'pool_limit' by score
        if args.pool_limit and len(pool) > args.pool_limit:
            pool.sort(key=lambda d: d.get("_score", 0.0), reverse=True)
            pool = pool[: args.pool_limit]
        save_pool(args.pool, pool)
        # also save the best weights of this iteration
        out_dir = os.path.dirname(args.pool) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"best_iter_{it}.json"), "w", encoding="utf-8") as f:
            json.dump({"score": best_score, "weights": best_params}, f, indent=2)
        print(f"  -> best score {best_score:.4f}; pool now {len(pool)} entries")

    print(f"\nDone. Pool saved to {os.path.abspath(args.pool)} (size={len(pool)}).")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BBB.train_pool",
        description="Repeat BO training and build an opponent weight pool."
    )
    p.add_argument("--iters", type=int, required=True, help="How many BO sessions to run.")
    p.add_argument("--pool", required=True, help="Path to JSON weight pool file.")
    p.add_argument("--pool-sample", type=int, default=3, help="Opponents sampled per objective eval (robust score).")
    p.add_argument("--pool-limit", type=int, default=0, help="If >0, keep only top-K weights in pool.")
    # BO sizes
    p.add_argument("--n-calls", type=int, default=100)
    p.add_argument("--n-init", type=int, default=45)
    p.add_argument("--seed", type=int, default=None)
    # delta config
    p.add_argument("--randomizations", type=int, default=10)
    p.add_argument("--games-per-alpha", type=int, default=15)
    p.add_argument("--cycles-per-position", type=int, default=1)
    p.add_argument("--num-battles", type=int, default=3)
    p.add_argument("--starting-coins", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_train_pool(args)

if __name__ == "__main__":
    raise SystemExit(main())
