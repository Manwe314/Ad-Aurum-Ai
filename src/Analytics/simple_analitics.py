# BBB/src/analytics.py
from __future__ import annotations

import argparse
import json
import os, uuid
from dataclasses import fields, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from .analytics_logger import (
    aggregate_both,
    plot_play_per_play_beats_losses, plot_play_per_play_certainty, plot_play_per_play_winrate,
    plot_play_eval_avg_wins, plot_play_eval_certainty, plot_play_eval_winrate,
    plot_bet_eval_avg_wins, plot_bet_eval_certainty, plot_bet_eval_winrate, plot_play_eval_avg_losses
)
from .analytics_logger import (
    aggregate_outcomes,
    write_card_outcome_winrates_csv,
    plot_outcome_winrates_with_concede,
    aggregate_new_coins, plot_new_coins_per_round_stacked,
    average_new_coins_by_round
)

# --- Engine imports (robust to minor layout differences) ---
try:
    from bbb.models import Player, Deck, Battle
    from bbb.board import BettingBoard
    from bbb.decisions import choose_favored_faction, choose_betting_type
    from bbb.phase4 import play_battle_phase
    from bbb.rounds import determine_round_winner, evaluate_favored_factions
    from bbb.brains.base import PlayerBrain, Traits, DeckMemory
except Exception:
    from bbb.models import Player, Deck, Battle  # type: ignore
    from bbb.board import BettingBoard  # type: ignore
    from bbb.decisions import choose_favored_faction, choose_betting_type  # type: ignore
    from bbb.phase4 import play_battle_phase  # type: ignore
    from bbb.rounds import determine_round_winner, evaluate_favored_factions  # type: ignore
    from bbb.brain.base import PlayerBrain, Traits  # type: ignore
    try:
        from .brains.base import DeckMemory  # type: ignore
    except Exception:
        DeckMemory = None  # optional

# Quiet engine-side logging/printing if present
try:
    from bbb.globals import LOGGING, GAME_ENGINE_PIRINTS, PARALEL_LOGGING
except Exception:
    LOGGING = False  # type: ignore
    GAME_ENGINE_PIRINTS = False  # type: ignore

# --- Plotting (matplotlib only; no seaborn) ---
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # for color validation
from matplotlib.patches import Patch  

import csv
import random
import math
import concurrent.futures
import multiprocessing as mp

# --------------------------
# Utilities: Traits loading
# --------------------------

def _traits_field_names() -> set:
    try:
        return {f.name for f in fields(Traits)}
    except Exception:
        return {
            "aggressiveness", "risk_tolerance", "tempo", "bluffiness",
            "stubbornness", "domination_drive", "herding", "ev_adherence", "exploration"
        }

def make_traits_safe(d: Dict) -> Traits:
    """
    Construct a Traits object from a dict, ignoring unknown keys
    and defaulting missing fields sensibly (50s; exploration defaults to 2).
    """
    allowed = _traits_field_names()
    defaults = dict(
        aggressiveness=50, risk_tolerance=50, tempo=50, bluffiness=50,
        stubbornness=50, domination_drive=50, herding=50, ev_adherence=50,
        exploration=d.get("exploration", 0.5),
    )
    payload = {k: d.get(k, defaults.get(k, 50)) for k in allowed}
    return Traits(**payload)  # type: ignore

def _validate_color(c: str) -> str:
    """Return the color if valid for matplotlib, else raise ValueError."""
    try:
        mcolors.to_rgba(c)  # will raise if invalid
        return c
    except Exception:
        raise ValueError(
            f"Invalid color '{c}'. Use a Matplotlib color name (e.g. 'red') "
            f"or hex (e.g. '#ff0000'). See https://matplotlib.org/stable/gallery/color/named_colors.html"
        )
    
# put this helper near your other small utils
def _wrap_legend_text(s: str, max_chars: int = 80) -> str:
    """Wrap 'Fixed: a=50, b=50, ...' into lines no longer than ~max_chars."""
    parts = s.split(", ")
    lines, cur = [], ""
    for p in parts:
        nxt = (cur + ", " if cur else "") + p
        if len(nxt) > max_chars and cur:
            lines.append(cur)
            cur = p
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def load_traits_json(path: str) -> tuple[List[Traits], List[str]]:
    """
    Expected JSON schema:
    {
      "players": [
        {"traits": { ... }, "color": "#ff0000"},
        {"traits": { ... }, "color": "blue"},
        {"traits": { ... }, "color": "#00aa88"},
        {"traits": { ... }, "color": "orange"}
      ]
    }
    Colors are required and must be Matplotlib-valid (name or hex).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    players = data.get("players", [])
    if len(players) != 4:
        raise ValueError(f"Traits JSON must contain exactly 4 players; got {len(players)}")

    traits_list: List[Traits] = []
    colors: List[str] = []
    for i, entry in enumerate(players):
        tdict = entry.get("traits", {})
        color = entry.get("color", None)
        if not color:
            raise ValueError(f"Player {i+1} is missing 'color' in JSON.")
        traits_list.append(make_traits_safe(tdict))
        colors.append(_validate_color(color))
    return traits_list, colors

# --- add near your other imports/helpers in analytics.py ---
def load_single_player_traits(path: str) -> Traits:
    """
    Expected JSON:
    { "traits": { ... } }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tdict = data.get("traits", {})
    return make_traits_safe(tdict)

def assert_valid_trait_name(name: str) -> str:
    fields_set = _traits_field_names()
    if name not in fields_set:
        raise ValueError(f"--target-trait '{name}' is not a valid Traits field. Valid: {sorted(fields_set)}")
    return name

def random_opponent_traits(seed: int) -> Traits:
    rng = random.Random(seed)
    names = _traits_field_names()
    # exploration stays modest (2) unless the dataclass default is different
    payload = {n: (0.5 if n == "exploration" else rng.randint(0, 100)) for n in names}
    return make_traits_safe(payload)


# -----------------------------------------
# Core: run one game and collect per-round
# -----------------------------------------

def _rotate_players(players: List[Player]) -> List[Player]:
    return players[1:] + players[:1] if players else players

def run_one_game_collect_round_end_coins(
    traits_by_seat: Sequence[Traits],
    *,
    starting_coins: int,
    num_battles: int = 3,
    base_seed: Optional[int] = None,
) -> Dict[str, List[int]]:
    """
    Runs a full 4-round game and records each player's coins
    at the end of each round.
    Returns: { "P1": [c1,c2,c3,c4], "P2": [...], ... }
    """

    assert len(traits_by_seat) == 4, "This framework assumes 4 players."

    if base_seed is None:
        base_seed = random.randint(0, 2**31-1)
    rng = random.Random(base_seed)

    # Silence engine chatter

    players: List[Player] = [Player(f"P{i+1}", starting_coins) for i in range(4)]
    for i, p in enumerate(players):
        p.brain = PlayerBrain(rng=random.Random(base_seed + 1000 + i), traits=traits_by_seat[i])  # type: ignore

    board = BettingBoard()
    round_end_coins: Dict[str, List[int]] = {p.name: [] for p in players}

    for round_num in range(1, 4 + 1):
        board.reset()
        deck = Deck()
        deck.shuffle()

        for p in players:
            p.reset_round()

        # Phase 1
        for p in players:
            choose_favored_faction(p, players)

        # Phase 2
        cards_per_player = num_battles * 3 + 1
        for p in players:
            p.cards = deck.draw(cards_per_player)
            if DeckMemory is not None and hasattr(p.brain, "brain_memory"):
                p.brain.brain_memory = DeckMemory()  # type: ignore
                p.brain.brain_memory.remove_cards(p.cards)  # type: ignore

        # Phase 3
        for p in players:
            choose_betting_type(p, players, board)

        # Phase 4: schedule & play battles
        battles: List[Battle] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                b = Battle(players[i], players[j])
                players[i].battles.append(b)
                players[j].battles.append(b)
                battles.append(b)

        for battle_index in range(num_battles):
            play_battle_phase(players, battles, board, round_num, battle_index)

        # Phase 5
        determine_round_winner(players, board)

        # Phase 6: evaluate favored factions; record coins; rotate seats
        evaluate_favored_factions(players, round_num)

        for p in players:
            round_end_coins[p.name].append(p.coins + p.front_coins)

        players = _rotate_players(players)

    for p in players:
        p.reset_round()

    winners = [p for p in players if p.coins == max(pl.coins for pl in players)]
    if len(winners) == 1:
        winner = winners[0]
    else:
        tiebreak = [p for p in winners if p.rounds_won == max(pl.rounds_won for pl in winners)]
        if len(tiebreak) == 1:
            winner = tiebreak[0]
        else:
            winner = None 

    return round_end_coins, winner.name if winner else None

# --------------------------
# Parallel helpers
# --------------------------

def _worker_init_quiet():
    """Initializer for child processes: silence engine chatter."""
    try:
        from bbb.globals import LOGGING, GAME_ENGINE_PIRINTS  # type: ignore
        LOGGING = False
        GAME_ENGINE_PIRINTS = False
    except Exception:
        pass

def _traits_to_dicts(traits_by_seat: Sequence[Traits]) -> List[Dict]:
    """Convert Traits objects to plain dicts for robust pickling across processes."""
    result: List[Dict] = []
    for t in traits_by_seat:
        try:
            result.append(asdict(t))  # dataclass → dict
        except Exception:
            # fallback: build from known fields
            names = _traits_field_names()
            result.append({k: getattr(t, k, 50) for k in names})
    return result

def _dicts_to_traits(trait_dicts: Sequence[Dict]) -> List[Traits]:
    return [make_traits_safe(d) for d in trait_dicts]

def _run_chunk(args) -> Tuple[Dict[str, List[int]], Dict[str, int], int]:
    """
    Worker task: run N games and return partial sums and wins.
    Returns:
      - sums: {P#: [sum_r1..r4]}
      - wins: {P#: wins}
      - ties: int
    """
    (trait_dicts, starting_coins, num_battles, seed, n_games) = args
    traits = _dicts_to_traits(trait_dicts)
    rng = random.Random(seed)

    sums: Dict[str, List[int]] = {f"P{i+1}": [0, 0, 0, 0] for i in range(4)}
    wins: Dict[str, int] = {f"P{i+1}": 0 for i in range(4)}
    ties = 0

    for _ in range(n_games):
        base_seed = rng.randint(0, 2**31 - 1)
        per_round, winner = run_one_game_collect_round_end_coins(
            traits_by_seat=traits,
            starting_coins=starting_coins,
            num_battles=num_battles,
            base_seed=base_seed,
        )
        for name, coins_list in per_round.items():
            # coins_list length is 4 (rounds)
            s = sums[name]
            s[0] += int(coins_list[0]); s[1] += int(coins_list[1])
            s[2] += int(coins_list[2]); s[3] += int(coins_list[3])
        if winner is None:
            ties += 1
        else:
            wins[winner] += 1

    return sums, wins, ties


def run_many_games_avg_round_coins(
    traits_by_seat: Sequence[Traits],
    *,
    games: int,
    starting_coins: int,
    num_battles: int = 3,
    seed: Optional[int] = None,
) -> tuple[Dict[str, List[float]], Dict[str, int], int]:
    """
    Sequential version (kept for small runs or workers=1).
    Returns (avgs, wins, ties).
    """
    if games <= 0:
        raise ValueError("games must be > 0")
    parent_rng = random.Random(seed)
    sums: Dict[str, List[int]] = {f"P{i+1}": [0, 0, 0, 0] for i in range(4)}
    wins: Dict[str, int] = {f"P{i+1}": 0 for i in range(4)}
    ties = 0

    for _ in range(games):
        base_seed = parent_rng.randint(0, 2**31 - 1)
        per_round, winner = run_one_game_collect_round_end_coins(
            traits_by_seat, starting_coins=starting_coins, num_battles=num_battles, base_seed=base_seed
        )
        for name, coins_list in per_round.items():
            for r_idx, c in enumerate(coins_list):
                sums[name][r_idx] += int(c)
        if winner is None:
            ties += 1
        else:
            wins[winner] += 1

    avgs: Dict[str, List[float]] = {name: [t / games for t in totals] for name, totals in sums.items()}
    return avgs, wins, ties

def run_many_games_avg_round_coins_parallel(
    traits_by_seat: Sequence[Traits],
    *,
    games: int,
    starting_coins: int,
    num_battles: int = 3,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> tuple[Dict[str, List[float]], Dict[str, int], int]:
    """
    Parallel version: splits total `games` into chunks and aggregates.
    Returns (avgs, wins, ties).
    """
    if games <= 0:
        raise ValueError("games must be > 0")

    # Decide worker count
    if not workers or workers <= 1:
        return run_many_games_avg_round_coins(
            traits_by_seat=traits_by_seat,
            games=games,
            starting_coins=starting_coins,
            num_battles=num_battles,
            seed=seed,
        )

    max_workers = os.cpu_count() or 1
    workers = max(1, min(workers, max_workers))

    # Chunking: default ~8 chunks per worker
    if not chunk_size or chunk_size <= 0:
        chunk_size = max(1, games // (workers * 8))
    n_chunks = math.ceil(games / chunk_size)


    if PARALEL_LOGGING:
        run_dir = os.path.join("src/Analytics/analytics_shards", f"run_{uuid.uuid4().hex}")
        os.makedirs(run_dir, exist_ok=True)
        os.environ["BBB_ANALYTICS_DIR"] = run_dir

    # Build tasks with independent seeds so results are reproducible but varied
    parent_rng = random.Random(seed)
    trait_dicts = _traits_to_dicts(traits_by_seat)
    tasks = []
    remaining = games
    for _ in range(n_chunks):
        n = min(chunk_size, remaining)
        if n <= 0:
            break
        remaining -= n
        task_seed = parent_rng.randint(0, 2**31 - 1)
        tasks.append((trait_dicts, starting_coins, num_battles, task_seed, n))

    # Aggregate partials
    sums_total: Dict[str, List[int]] = {f"P{i+1}": [0, 0, 0, 0] for i in range(4)}
    wins_total: Dict[str, int] = {f"P{i+1}": 0 for i in range(4)}
    ties_total = 0

    ctx = mp.get_context("spawn")  # explicit for Windows
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_worker_init_quiet
    ) as pool:
        for sums_part, wins_part, ties_part in pool.map(_run_chunk, tasks):
            for name in sums_total.keys():
                st = sums_total[name]; sp = sums_part[name]
                st[0] += sp[0]; st[1] += sp[1]; st[2] += sp[2]; st[3] += sp[3]
                wins_total[name] += wins_part[name]
            ties_total += ties_part

    # Compute averages and logging output
    avgs: Dict[str, List[float]] = {name: [t / games for t in totals] for name, totals in sums_total.items()}
    if PARALEL_LOGGING:
        play_agg, bet_agg = aggregate_both(run_dir)
        outcome_agg = aggregate_outcomes(run_dir)
        coin_outcome_agg = aggregate_new_coins(run_dir)
        coins_outcome = average_new_coins_by_round(coin_outcome_agg, games=games)

        # write_card_outcomes_csv(os.path.join(run_dir, "play_card_outcomes.csv"), play_agg)
        # write_card_outcomes_csv(os.path.join(run_dir, "bet_card_outcomes.csv"),  bet_agg)

        # Charts (top_k optional if you have many cards)
        plot_new_coins_per_round_stacked(os.path.join(run_dir, "coins/new_coins_per_round_stacked.png"), coins_outcome, title_prefix="new coins avarage")

        plot_outcome_winrates_with_concede(os.path.join(run_dir, "actual_winrate/card_outcome_winrates.png"), outcome_agg)

        plot_play_per_play_beats_losses(os.path.join(run_dir, "per_plays/play_per_play_beats_losses.png"), play_agg)
        plot_play_per_play_certainty(   os.path.join(run_dir, "per_plays/play_per_play_certainty.png"),    play_agg)
        plot_play_per_play_winrate(     os.path.join(run_dir, "per_plays/play_per_play_winrate.png"),      play_agg)

        plot_play_eval_avg_wins(        os.path.join(run_dir, "per_evaluation/card_picking/play_eval_avg_wins.png"),         play_agg)
        plot_play_eval_avg_losses(os.path.join(run_dir, "per_evaluation/card_picking/play_eval_avg_losses.png"), play_agg)
        plot_play_eval_certainty(       os.path.join(run_dir, "per_evaluation/card_picking/play_eval_certainty.png"),        play_agg)
        plot_play_eval_winrate(         os.path.join(run_dir, "per_evaluation/card_picking/play_eval_winrate.png"),          play_agg)

        plot_bet_eval_avg_wins(         os.path.join(run_dir, "per_evaluation/card_betting/bet_eval_avg_wins.png"),          bet_agg)
        plot_bet_eval_certainty(        os.path.join(run_dir, "per_evaluation/card_betting/bet_eval_certainty.png"),         bet_agg)
        plot_bet_eval_winrate(          os.path.join(run_dir, "per_evaluation/card_betting/bet_eval_winrate.png"),           bet_agg)
    return avgs, wins_total, ties_total

def winrate_for_trait_value(
    *,
    base_traits: Traits,
    target_trait: str,
    trait_value: int,
    runs: int,
    games_per_run: int,
    starting_coins: int,
    num_battles: int,
    seed: Optional[int],
    workers: int,
    chunk_size: int,
) -> float:
    """
    Returns overall winrate (%) for tracked player (seat P1) when its `target_trait`
    is set to `trait_value`, averaging across `runs`. Each run randomizes opponents.
    """
    # build tracked player's traits for this point
    bt = base_traits
    # create a copy with the override
    t_kwargs = {k: getattr(bt, k) for k in _traits_field_names()}
    t_kwargs[target_trait] = max(0, min(100, int(trait_value)))
    tracked = Traits(**t_kwargs)  # type: ignore

    parent_rng = random.Random(seed)
    total_wins = 0
    total_games = 0

    for r in range(runs):
        # randomize opponents for this run
        seed_run = parent_rng.randint(0, 2**31 - 1)
        opp1 = random_opponent_traits(seed_run + 1)
        opp2 = random_opponent_traits(seed_run + 2)
        opp3 = random_opponent_traits(seed_run + 3)

        avgs, wins, ties = run_many_games_avg_round_coins_parallel(
            traits_by_seat=[tracked, opp1, opp2, opp3],
            games=games_per_run,
            starting_coins=starting_coins,
            num_battles=num_battles,
            seed=seed_run,
            workers=workers,
            chunk_size=chunk_size if chunk_size and chunk_size > 0 else None,
        )
        total_wins += int(wins.get("P1", 0))
        total_games += games_per_run

    return (100.0 * total_wins / total_games) if total_games else 0.0


# --------------------------
# Minimal CSV + plotting
# --------------------------

def _format_traits_legend(tr: Traits) -> str:
    """Compact per-player traits text for legend."""
    order = [
        "aggressiveness", "tempo", "stubbornness",
        "bluffiness", "herding",
        "ev_adherence", "risk_tolerance", "domination_drive",
        "exploration",
    ]
    short = {
        "aggressiveness": "agg",
        "tempo": "tempo",
        "stubbornness": "stub",
        "bluffiness": "bluff",
        "herding": "herd",
        "ev_adherence": "ev",
        "risk_tolerance": "risk",
        "domination_drive": "dom",
        "exploration": "explr",
    }
    parts = []
    for k in order:
        if hasattr(tr, k):
            parts.append(f"{short.get(k,k)}={getattr(tr, k)}")
    return ", ".join(parts)

def save_iterate_trait_csv(out_csv: str, xs: List[int], ys: List[float], meta: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_trait", meta.get("target_trait", "")])
        w.writerow(["step", meta.get("step", "")])
        w.writerow(["runs", meta.get("runs", "")])
        w.writerow(["games_per_run", meta.get("games_per_run", "")])
        w.writerow([])
        w.writerow(["trait_value", "winrate_percent"])
        for x, y in zip(xs, ys):
            w.writerow([x, f"{y:.3f}"])

def plot_iterate_trait(out_png: str, xs: List[int], ys: List[float], legend_text: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    # line + markers
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Trait value")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # value labels on each point
    for x, y in zip(xs, ys):
        dy = -10 if y >= 95 else 5
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8)

    # title as a figure-level suptitle
    fig.suptitle(title, y=0.98)

    # put fixed traits as a wrapped subtitle (figure text) above the axes
    wrapped = _wrap_legend_text(legend_text, max_chars=80)
    fig.text(0.5, 0.92, wrapped, ha="center", va="top", fontsize=8)

    # leave room at the top for the subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)




def save_csv_summary(out_csv: str, avgs: Dict[str, List[float]]) -> None:
    names = [f"P{i+1}" for i in range(4)]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Round"] + names)
        for r in range(4):
            row = [r + 1] + [f"{avgs[n][r]:.3f}" for n in names]
            w.writerow(row)

def save_csv_win_rates(out_csv: str, wins: Dict[str, int], games: int, ties: int) -> None:
    names = [f"P{i+1}" for i in range(4)]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Player", "Wins", "WinRatePercent", "TotalGames", "Ties"])
        for n in names:
            wr = (wins[n] / games) * 100.0 if games else 0.0
            w.writerow([n, wins[n], f"{wr:.3f}", games, ties])

def plot_all_players_one_chart(out_png: str, avgs: Dict[str, List[float]], games: int, colors: List[str]) -> None:
    rounds = [1, 2, 3, 4]
    plt.figure()
    for i in range(4):
        name = f"P{i+1}"
        y = avgs[name]
        plt.plot(rounds, y, marker="o", label=name, color=colors[i])  # user-specified colors
    plt.title(f"Average coins per round (n={games})")
    plt.xlabel("Round")
    plt.ylabel("Coins")
    plt.ylim(bottom=0)  # Y axis from 0
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def plot_win_rates_bar(
    out_png: str,
    wins: Dict[str, int],
    games: int,
    colors: List[str],
    traits_by_seat: Sequence[Traits],
) -> None:
    names = [f"P{i+1}" for i in range(4)]
    xs = list(range(4))
    heights = [(wins[n] / games) * 100.0 if games else 0.0 for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(xs, heights, color=colors)
    plt.xticks(xs, names)
    plt.ylabel("Win rate (%)")
    plt.ylim(0, 100)
    plt.title(f"Win rate per player (n={games})")

    # value labels on bars
    for idx, h in enumerate(heights):
        plt.text(idx, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    # legend with trait summaries per player (color-matched)
    handles = [
        Patch(
            facecolor=colors[i],
            label=f"{names[i]}: {_format_traits_legend(traits_by_seat[i])}"
        )
        for i in range(4)
    ]
    plt.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# --------------------------
# CLI (subcommands)
# --------------------------

def cmd_iterate_trait(args: argparse.Namespace) -> int:
    global PARALEL_LOGGING
    PARALEL_LOGGING = False  # silence detailed logging for this mode
    base_traits = load_single_player_traits(args.traits)
    target_trait = assert_valid_trait_name(args.target_trait)

    # fixed traits (legend): everything except the target
    fixed_pairs = []
    for name in sorted(_traits_field_names()):
        if name == target_trait:
            continue
        fixed_pairs.append(f"{name}={getattr(base_traits, name)}")
    legend_text = "Fixed: " + ", ".join(fixed_pairs)

    # build X values: 0 .. 100 (ensure 100 included)
    step = max(1, int(args.step))
    xs: List[int] = list(range(0, 101, step))
    if xs[-1] != 100:
        xs.append(100)

    ys: List[float] = []
    parent_rng = random.Random(args.seed)
    # we’ll vary the *outer* seed per point for reproducibility
    for x in xs:
        seed_point = parent_rng.randint(0, 2**31 - 1)
        wr = winrate_for_trait_value(
            base_traits=base_traits,
            target_trait=target_trait,
            trait_value=x,
            runs=args.runs,
            games_per_run=args.games,
            starting_coins=args.starting_coins,
            num_battles=args.num_battles,
            seed=seed_point,
            workers=args.workers if args.workers and args.workers > 1 else 1,
            chunk_size=args.chunk_size if args.chunk_size and args.chunk_size > 0 else 0,
        )
        ys.append(wr)

    os.makedirs(args.outdir, exist_ok=True)

    # save CSV
    save_iterate_trait_csv(
        os.path.join(args.outdir, "iterate_trait_curve.csv"),
        xs, ys,
        meta=dict(
            target_trait=target_trait,
            step=str(step),
            runs=str(args.runs),
            games_per_run=str(args.games),
        ),
    )

    # plot
    model_note = f" • {args.model_name}" if args.model_name else ""
    title = f"Winrate vs {target_trait}{model_note} (runs={args.runs}, games/run={args.games})"
    plot_iterate_trait(
        os.path.join(args.outdir, "iterate_trait_curve.png"),
        xs, ys, legend_text, title
    )

    print(f"Saved iterate-trait CSV/PNG to: {os.path.abspath(args.outdir)}")
    return 0


def cmd_coins_per_round(args: argparse.Namespace) -> int:
    traits, colors = load_traits_json(args.traits)

    # Debug (optional): confirm what we're about to use
    # print(f"[analytics] workers={args.workers} chunk_size={args.chunk_size}")

    avgs, wins, ties = run_many_games_avg_round_coins_parallel(
        traits_by_seat=traits,
        games=args.games,
        starting_coins=args.starting_coins,
        num_battles=args.num_battles,
        seed=args.seed,
        workers=args.workers,          # << pass it
        chunk_size=args.chunk_size,    # << pass it
    )

    os.makedirs(args.outdir, exist_ok=True)
    save_csv_summary(os.path.join(args.outdir, f"avg_coins_summary_{args.model_name}.csv"), avgs)
    save_csv_win_rates(os.path.join(args.outdir, f"win_rates_{args.model_name}.csv"), wins, args.games, ties)

    plot_all_players_one_chart(
        out_png=os.path.join(args.outdir, f"avg_coins_all_players_{args.model_name}.png"),
        avgs=avgs,
        games=args.games,
        colors=colors,
    )
    plot_win_rates_bar(
    out_png=os.path.join(args.outdir, f"win_rates_bars_{args.model_name}.png"),
    wins=wins,
    games=args.games,
    colors=colors,
    traits_by_seat=traits,  # <-- add this
)

    print(f"Saved CSV and charts to: {os.path.abspath(args.outdir)}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="BBB.analytics",
        description="Analytics tools for BBB simulator (coins per round, etc.)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser(
        "coins-per-round",
        help="Run N games with given traits and plot average end-of-round coins per player (single chart, colored lines)."
    )
    q.add_argument("--traits", required=True, help="Path to JSON with 4 players' traits and colors.")
    q.add_argument("--games", type=int, required=True, help="Number of games to simulate.")
    q.add_argument("--starting-coins", type=int, required=True, help="Starting coins per player.")
    q.add_argument("--num-battles", type=int, default=3, help="Battles per round (default: 3).")
    q.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    q.add_argument("--outdir", default="analytics_out", help="Output directory for CSV/PNG.")
    q.add_argument("--workers", type=int, default=0, help="Number of processes to use (0/1 = no parallelism; recommend os.cpu_count()).")
    q.add_argument("--chunk-size", type=int, default=0, help="Games per task (0 = auto ~8 chunks/worker).")
    q.add_argument("--model-name", default="AAA", help="Model name of used weights to name by")

    it = sub.add_parser(
        "iterate-trait",
        help="Sweep a single trait (0..100) for the tracked player and plot winrate vs trait value."
    )
    it.add_argument("--traits", required=True, help="Path to JSON with ONE player's baseline traits: {\"traits\": {...}}")
    it.add_argument("--target-trait", required=True, help="Trait field to sweep, e.g. aggressiveness")
    it.add_argument("--step", type=int, default=10, help="Step size between 0 and 100 (default: 10)")
    it.add_argument("--runs", type=int, required=True, help="Independent runs per trait value (opponents randomized each run)")
    it.add_argument("--games", type=int, required=True, help="Games per run (run in parallel if --workers>1)")
    it.add_argument("--starting-coins", type=int, required=True)
    it.add_argument("--num-battles", type=int, default=3)
    it.add_argument("--seed", type=int, default=None)
    it.add_argument("--workers", type=int, default=0, help="Processes for inner game sims (per run)")
    it.add_argument("--chunk-size", type=int, default=0)
    it.add_argument("--model-name", default="", help="Optional label to show on the plot title")
    it.add_argument("--outdir", default="iterate_out")

    return p

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "coins-per-round":
        return cmd_coins_per_round(args)
    elif args.cmd == "iterate-trait":
        return cmd_iterate_trait(args)
    else:
        parser.error(f"Unknown cmd {args.cmd}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
