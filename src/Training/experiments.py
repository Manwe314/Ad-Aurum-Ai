# BBB/src/experiments.py
from __future__ import annotations

import random
import time
import statistics
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
import os
import concurrent.futures
try:
    import psutil  # pip install psutil
except Exception:
    psutil = None

# --- imports from your package (same package level as simulate_game) ---
try:
    # match your current modules
    from bbb.models import Player, Deck, Battle
    from bbb.board import BettingBoard
    from bbb.decisions import choose_favored_faction, choose_betting_type
    from bbb.phase4 import play_battle_phase
    from bbb.rounds import determine_round_winner, evaluate_favored_factions
    from bbb.brains.base import PlayerBrain, Traits, DeckMemory  # as used in your entrypoint
except Exception:
    # fallback if your package is singular `brain` not `brains`
    from .brain.base import PlayerBrain, Traits  # type: ignore
    from .models import Player, Deck, Battle  # type: ignore
    from .board import BettingBoard  # type: ignore
    from .decisions import choose_favored_faction, choose_betting_type  # type: ignore
    from .phase4 import play_battle_phase  # type: ignore
    from .rounds import determine_round_winner, evaluate_favored_factions  # type: ignore
    try:
        from .brains.base import DeckMemory  # type: ignore
    except Exception:
        DeckMemory = None  # not strictly required; guarded below

# Optional engine globals (gracefully ignored if absent)
try:
    from bbb.globals import (
        ADDITIONAL_INFO,
        TARGET_PLAYER,
        GAME_ENGINE_PIRINTS,
        LOGGER,
        LOGGING,
        PARALEL_LOGGING,
        WW_VARS
    )
except Exception:
    ADDITIONAL_INFO = ""
    TARGET_PLAYER = ""
    GAME_ENGINE_PIRINTS = False
    LOGGING = False
    LOGGER = None
    WW_VARS = {}

# ---------------------------------------------------------------------
# Experiment configuration (tweak freely)
# ---------------------------------------------------------------------
# if True, use NORMAL_PRIORITY_CLASS; else ABOVE_NORMAL_PRIORITY_CLASS
CPU_NORMAL = True  

# amount of games = games per alpha * cycles per position * gamma randomizations * 24 default strategies * 4 per cycle
# Example: 1 * 1 * 1 * 24 * 4 = 96 games total
# X: games per alpha run
GAMES_PER_ALPHA: int = 15

# Y: how many full cycles of starting positions to run in beta
CYCLES_PER_POSITION: int = 1

# number of independent opponent randomizations per gamma
GAMMA_RANDOMIZATIONS: int = 10

# fixed game parameters (you can override per call)
NUM_BATTLES: int = 3
STARTING_COINS: int = 10

# tie handling: how much credit for a tied game
TIE_VALUE: float = 0.5

# exploration constant used in experiments (tracked and opponents)
EXPLORATION_CONST: int = 0.5

# delta scoring: score = mean_win_rate - STD_PENALTY * std_across_strategies
STD_PENALTY: float = 0.33

# RNG base seed (for reproducibility)
BASE_SEED: int = 42


# ---------------------------------------------------------------------
# Traits helpers
# ---------------------------------------------------------------------

def _traits_field_names() -> set:
    # make robust to your evolving Traits dataclass
    try:
        return {f.name for f in fields(Traits)}
    except Exception:
        # fallback if Traits is a simple class
        return {
            "aggressiveness", "risk_tolerance", "tempo", "bluffiness",
            "stubbornness", "domination_drive", "herding", "ev_adherence",
            "exploration"
        }

def _make_traits(**kwargs) -> Traits:
    """Safely construct Traits with only known fields (avoids version drift)."""

    allowed = _traits_field_names()
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    # fill any missing required fields with sensible defaults
    defaults = dict(
        aggressiveness=50, risk_tolerance=50, tempo=50, bluffiness=50,
        stubbornness=50, domination_drive=50, herding=50, ev_adherence=50,
        exploration=EXPLORATION_CONST,
    )
    for k in allowed:
        filtered.setdefault(k, defaults.get(k, 50))
    return Traits(**filtered)  # type: ignore

def tracked_baseline_traits() -> Traits:
    """Tracked player stays at 50 for all (except exploration)."""
    return _make_traits(
        aggressiveness=50, risk_tolerance=50, tempo=50, bluffiness=50,
        stubbornness=50, domination_drive=50, herding=50, ev_adherence=50,
        exploration=EXPLORATION_CONST,
    )


# ---------------------------------------------------------------------
# Strategy space (gamma/delta)
# ---------------------------------------------------------------------

Level = Literal["low", "mid", "high"]
LEVEL_RANGES: Dict[Level, Tuple[int, int]] = {
    "low": (0, 33),
    "mid": (33, 66),
    "high": (66, 100),
}

GROUPS = {
    # 1) aggressiveness, tempo, stubbornness
    "g1": ("aggressiveness", "tempo", "stubbornness"),
    # 2) bluffiness, herding
    "g2": ("bluffiness", "herding"),
    # 3) ev_adherence, risk_tolerance, domination_drive
    "g3": ("ev_adherence", "risk_tolerance", "domination_drive"),
}

@dataclass(frozen=True)
class Strategy:
    """A strategy defines (g1, g2, g3) levels; gamma randomizes within ranges."""
    g1: Level
    g2: Level
    g3: Level

    @property
    def name(self) -> str:
        return f"G1-{self.g1}_G2-{self.g2}_G3-{self.g3}"

# All 9 combos
ALL_STRATEGIES: List[Strategy] = [
    Strategy(g1, g2, g3)
    for g1 in ("low", "mid", "high")
    for g2 in ("low", "mid", "high")
    for g3 in ("low", "mid", "high")
]

# Default 6-of-9 subset; hold out the diagonals for post-training tests
DEFAULT_DELTA_STRATEGIES: List[Strategy] = [
    s for s in ALL_STRATEGIES
    if s not in {
        Strategy("low", "low", "low"),
        Strategy("mid", "mid", "mid"),
        Strategy("high", "high", "high"),
    }
]


def _sample_level(rng: random.Random, level: Level) -> int:
    lo, hi = LEVEL_RANGES[level]
    # inclusive ranges as requested (0–33, 33–66, 66–100)
    return rng.randint(lo, hi)


def _randomize_opponent_traits_for_strategy(
    rng: random.Random, strategy: Strategy
) -> Traits:
    """Randomize a single opponent's traits according to a strategy."""
    vals: Dict[str, int] = {}

    for group_key, level in zip(("g1", "g2", "g3"), (strategy.g1, strategy.g2, strategy.g3)):
        for fname in GROUPS[group_key]:
            vals[fname] = _sample_level(rng, level)

    # exploration is constant
    vals["exploration"] = EXPLORATION_CONST
    return _make_traits(**vals)


# ---------------------------------------------------------------------
# Core game runner (single game, fully parameterized)
# ---------------------------------------------------------------------

def _rotate_players(players: List[Player]) -> List[Player]:
    return players[1:] + players[:1] if players else players

def _play_one_full_game(
    player_traits: Sequence[Traits],
    tracked_index: int,
    *,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    base_seed: int = BASE_SEED,
) -> Tuple[float, Dict[str, int]]:
    """
    Run one *full* game (same structure as your simulate_game), return:
    - win_value for tracked player (1, TIE_VALUE, or 0)
    - rounds_won per player by name

    A "win" is defined as strictly highest rounds_won at game end.
    Ties award TIE_VALUE credit.
    """
    assert len(player_traits) == 4, "This framework assumes 4 players."

    # Build players with requested seating order
    players: List[Player] = [Player(f"P{i+1}", starting_coins) for i in range(4)]
    players[tracked_index].training_target = True  
    board = BettingBoard()
    rng = random.Random(base_seed)

    # Attach brains with their traits
    for i, p in enumerate(players):
        p.brain = PlayerBrain(rng=random.Random(base_seed + 1000 + i), traits=player_traits[i])  # type: ignore

    # Tournament: one round per player (mirrors your simulate_game)
    for round_num in range(1, 4 + 1):
        board.reset()
        deck = Deck()
        deck.shuffle()

        for p in players:
            p.reset_round()

        # Phase 1: favored faction
        for p in players:
            choose_favored_faction(p, players)

        # Phase 2: draw cards
        cards_per_player = num_battles * 3 + 1
        for p in players:
            p.cards = deck.draw(cards_per_player)
            if DeckMemory is not None and hasattr(p.brain, "brain_memory"):
                p.brain.brain_memory = DeckMemory()  # type: ignore
                p.brain.brain_memory.remove_cards(p.cards)  # type: ignore

        # Phase 3: choose betting types
        for p in players:
            choose_betting_type(p, players, board)

        # Phase 4: schedule all pairwise battles, then play N battles
        battles: List[Battle] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                battle = Battle(players[i], players[j])
                players[i].battles.append(battle)
                players[j].battles.append(battle)
                battles.append(battle)

        for battle_index in range(num_battles):
            play_battle_phase(players, battles, board, round_num, battle_index)

        # Phase 5: determine round winner
        determine_round_winner(players, board)

        # Phase 6: cleanup, eval favored factions, rotate seating
        evaluate_favored_factions(players, round_num)
        players = _rotate_players(players)

    for p in players:
        p.reset_round()

    name_by_idx = {i: f"P{i+1}" for i in range(4)}
    tracked_name = name_by_idx[tracked_index]
    rounds_won = {p.name: p.rounds_won for p in players}
    
    winners = [p for p in players if p.coins == max(pl.coins for pl in players)]
    if len(winners) == 1:
        winner = winners[0]
        if winner.name == tracked_name:
            return 1.0, rounds_won
        else:
            return 0.0, rounds_won
    else:
        tiebreak = [p for p in winners if p.rounds_won == max(pl.rounds_won for pl in winners)]
        if len(tiebreak) == 1:
            winner = tiebreak[0]
            if winner.name == tracked_name:
                return 1.0, rounds_won
            else:
                return 0.0, rounds_won
        else:
            return TIE_VALUE, rounds_won


# ---------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------

@dataclass
class AlphaResult:
    wins: float
    games: int
    win_rate: float

def run_alpha(
    *,
    opponent_traits: Sequence[Union[Traits, Dict[str, int]]],
    tracked_index: int,
    games: int = GAMES_PER_ALPHA,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    seed: int = BASE_SEED,
) -> AlphaResult:
    """
    Alpha: run X games with fixed opponents. Tracked player at tracked_index.
    Opponents must be 3 entries (P1..P4 are filled accordingly).
    Tracked player is always the baseline (50s, exploration=0.5).
    """
    assert len(opponent_traits) == 3, "Provide traits for exactly 3 opponents."
    # Build the 4-seat trait array in seating order with tracked seat at tracked_index
    base = [None, None, None, None]  # type: ignore
    base[tracked_index] = tracked_baseline_traits()

    # Fill other seats left-to-right with provided opponent traits
    opp_iter = iter(opponent_traits)
    for i in range(4):
        if base[i] is None:
            t = next(opp_iter)
            base[i] = t if isinstance(t, Traits) else _make_traits(**t)

    rng = random.Random(seed)
    wins = 0.0
    for g in range(games):
        # vary seed per game for reproducibility with diversity
        game_seed = rng.randint(0, 2**31 - 1)
        w, _ = _play_one_full_game(
            base, tracked_index,
            num_battles=num_battles, starting_coins=starting_coins, base_seed=game_seed
        )
        wins += w

    win_rate = wins / games if games > 0 else 0.0
    return AlphaResult(wins=wins, games=games, win_rate=win_rate)


@dataclass
class BetaResult:
    per_position: List[AlphaResult]
    total_games: int
    overall_win_rate: float

def run_beta(
    *,
    opponent_traits: Sequence[Union[Traits, Dict[str, int]]],
    cycles: int = CYCLES_PER_POSITION,
    games_per_alpha: int = GAMES_PER_ALPHA,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    seed: int = BASE_SEED,
) -> BetaResult:
    """
    Beta: for the same opponent traits, cycle seating so that the tracked player
    sits at each position (P1..P4). For each seating, run Alpha (X games).
    Repeat the whole cycle `cycles` times (Y).
    """
    rng = random.Random(seed)
    per_position: List[AlphaResult] = [AlphaResult(0, 0, 0.0) for _ in range(4)]

    for c in range(cycles):
        for tracked_index in range(4):
            alpha_seed = rng.randint(0, 2**31 - 1)
            ares = run_alpha(
                opponent_traits=opponent_traits,
                tracked_index=tracked_index,
                games=games_per_alpha,
                num_battles=num_battles,
                starting_coins=starting_coins,
                seed=alpha_seed,
            )
            prev = per_position[tracked_index]
            # accumulate
            total_wins = prev.wins + ares.wins
            total_games = prev.games + ares.games
            per_position[tracked_index] = AlphaResult(
                wins=total_wins,
                games=total_games,
                win_rate=(total_wins / total_games) if total_games else 0.0,
            )

    total_games = sum(p.games for p in per_position)
    overall_wr = (sum(p.wins for p in per_position) / total_games) if total_games else 0.0
    return BetaResult(per_position=per_position, total_games=total_games, overall_win_rate=overall_wr)


@dataclass
class GammaResult:
    strategy: Strategy
    beta: BetaResult

def run_gamma(
    *,
    strategy: Strategy,
    randomizations: int = GAMMA_RANDOMIZATIONS,
    games_per_alpha: int = GAMES_PER_ALPHA,
    cycles_per_position: int = CYCLES_PER_POSITION,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    seed: int = BASE_SEED,
) -> GammaResult:
    """
    Gamma: given a strategy (levels for the 3 groups), create `randomizations`
    opponent triplets by sampling traits inside the bins, and for each
    randomization run Beta (which itself cycles seating and runs Alpha).
    Returns aggregated Beta across all randomizations.
    """
    rng = random.Random(seed)
    # accumulate across randomizations
    agg_per_pos: List[AlphaResult] = [AlphaResult(0, 0, 0.0) for _ in range(4)]

    for _ in range(randomizations):
        # build 3 opponents via strategy
        opps = [
            _randomize_opponent_traits_for_strategy(rng, strategy)
            for _ in range(3)
        ]
        beta_seed = rng.randint(0, 2**31 - 1)
        bres = run_beta(
            opponent_traits=opps,
            cycles=cycles_per_position,
            games_per_alpha=games_per_alpha,
            num_battles=num_battles,
            starting_coins=starting_coins,
            seed=beta_seed,
        )
        # accumulate per position
        for i, ar in enumerate(bres.per_position):
            prev = agg_per_pos[i]
            wins = prev.wins + ar.wins
            games = prev.games + ar.games
            agg_per_pos[i] = AlphaResult(wins, games, (wins / games) if games else 0.0)

    total_games = sum(p.games for p in agg_per_pos)
    overall_wr = (sum(p.wins for p in agg_per_pos) / total_games) if total_games else 0.0
    return GammaResult(strategy=strategy, beta=BetaResult(agg_per_pos, total_games, overall_wr))


@dataclass
class DeltaResult:
    overall_win_rate: float
    score: float
    per_strategy: List[Tuple[str, float]]
    details: Dict[str, Dict[str, float]]

def run_delta(
    *,
    strategies: Optional[Sequence[Strategy]] = None,
    randomizations: int = GAMMA_RANDOMIZATIONS,
    games_per_alpha: int = GAMES_PER_ALPHA,
    cycles_per_position: int = CYCLES_PER_POSITION,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    seed: int = BASE_SEED,
    std_penalty: float = STD_PENALTY,
) -> DeltaResult:
    """
    Delta: run Gamma for each strategy in `strategies` (default: 6-of-9 subset),
    then compute:
      - overall_win_rate: mean across all strategies weighted by their games
      - score: mean_win_rate - std_penalty * std( per-strategy win rates )
    Returns both the score and raw per-strategy win rates.
    """
    if strategies is None:
        strategies = list(DEFAULT_DELTA_STRATEGIES)

    rng = random.Random(seed)
    gamma_results: List[GammaResult] = []

    for strat in strategies:
        gseed = rng.randint(0, 2**31 - 1)
        gres = run_gamma(
            strategy=strat,
            randomizations=randomizations,
            games_per_alpha=games_per_alpha,
            cycles_per_position=cycles_per_position,
            num_battles=num_battles,
            starting_coins=starting_coins,
            seed=gseed,
        )
        gamma_results.append(gres)

    per_strategy_wr = [(g.strategy.name, g.beta.overall_win_rate) for g in gamma_results]
    wr_values = [wr for _, wr in per_strategy_wr]
    mean_wr = sum(wr_values) / len(wr_values) if wr_values else 0.0
    std_wr = statistics.pstdev(wr_values) if len(wr_values) > 1 else 0.0
    score = mean_wr - std_penalty * std_wr

    # weighted overall (identical to mean_wr here since each Gamma uses same config)
    overall_wr = mean_wr

    details = {
        name: {
            "win_rate": wr,
        }
        for name, wr in per_strategy_wr
    }

    return DeltaResult(
        overall_win_rate=overall_wr,
        score=score,
        per_strategy=per_strategy_wr,
        details=details,
    )


# ---------------------------------------------------------------------
# Convenience: quick alpha/beta with explicit traits
# ---------------------------------------------------------------------

def make_traits(**kwargs) -> Traits:
    """Public helper to create Traits safely (respects current dataclass fields)."""
    return _make_traits(**kwargs)





def _gamma_worker(args) -> "GammaResult":
    if psutil:
        p = psutil.Process(os.getpid())
        try:
            p.cpu_affinity(list(range(os.cpu_count() or 1)))
        except Exception:
            pass
        try:
            if os.name == "nt":
                if CPU_NORMAL:
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(-5)
        except Exception:
            pass

    (
        strategy,
        randomizations,
        games_per_alpha,
        cycles_per_position,
        num_battles,
        starting_coins,
        seed,
        opp_weights,
    ) = args

    # --- set opponents' weights for THIS process only
    try:
        if opp_weights:
            WW_VARS.clear()
            WW_VARS.update(opp_weights)
    except Exception:
        pass

    return run_gamma(
        strategy=strategy,
        randomizations=randomizations,
        games_per_alpha=games_per_alpha,
        cycles_per_position=cycles_per_position,
        num_battles=num_battles,
        starting_coins=starting_coins,
        seed=seed,
    )


def run_delta_parallel(
    *,
    strategies: Optional[Sequence[Strategy]] = None,
    randomizations: int = GAMMA_RANDOMIZATIONS,
    games_per_alpha: int = GAMES_PER_ALPHA,
    cycles_per_position: int = CYCLES_PER_POSITION,
    num_battles: int = NUM_BATTLES,
    starting_coins: int = STARTING_COINS,
    seed: int = BASE_SEED,
    std_penalty: float = STD_PENALTY,
    workers: Optional[int] = None,
    opp_weights: Optional[Dict[str, float]] = None,
) -> DeltaResult:
    """
    Parallel delta: run one Gamma per strategy in separate processes.

    Notes:
      - For CPU-bound sims, processes beat threads (GIL).
      - Disable or isolate logging; multiple processes writing the same file can clash.
    """
    global LOGGING, PARALEL_LOGGING
    LOGGING = PARALEL_LOGGING = False  # disable logging in workers
    if strategies is None:
        strategies = list(DEFAULT_DELTA_STRATEGIES)

    # determine pool size
    if workers is None or workers <= 0:
        cpu = os.cpu_count() or 1
        workers = min(len(strategies), cpu)

    # pre-generate per-strategy seeds in the parent for reproducibility
    parent_rng = random.Random(seed)
    tasks = []
    for strat in strategies:
        s = parent_rng.randint(0, 2**31 - 1)
        tasks.append((
            strat,
            randomizations,
            games_per_alpha,
            cycles_per_position,
            num_battles,
            starting_coins,
            s,
            opp_weights,
        ))

    # launch workers
    results: List[GammaResult] = [None] * len(tasks)  # type: ignore
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_gamma_worker, t): i
            for i, t in enumerate(tasks)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()

    # aggregate exactly like run_delta() does
    per_strategy = [(r.strategy.name, r.beta.overall_win_rate) for r in results]  # type: ignore
    wr_values = [wr for _, wr in per_strategy]
    mean_wr = sum(wr_values) / len(wr_values) if wr_values else 0.0
    std_wr = statistics.pstdev(wr_values) if len(wr_values) > 1 else 0.0
    score = mean_wr - std_penalty * std_wr
    overall_wr = mean_wr

    details = {name: {"win_rate": wr} for name, wr in per_strategy}

    return DeltaResult(
        overall_win_rate=overall_wr,
        score=score,
        per_strategy=per_strategy,
        details=details,
    )


# ---------------------------------------------------------------------
# CLI demo (optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    res = run_delta()  # uses DEFAULT_DELTA_STRATEGIES and global defaults
    elapsed = time.time() - t0
    print(f"Delta experiment completed in {elapsed:.1f} seconds")
    print(f"overall_win_rate={res.overall_win_rate:.4f}")
    print(f"score={res.score:.4f}")
    for name, wr in res.per_strategy:
        print(f"{name}\t{wr:.4f}")
