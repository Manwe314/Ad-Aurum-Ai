from __future__ import annotations

import atexit
import json
import os
import threading
import time
import uuid
from typing import Dict, Optional, Sequence, Tuple, List
from bbb.models import Card

import matplotlib.pyplot as plt  
import csv

# ---------- helpers ----------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

def _run_dir_from_env() -> str:
    """
    Directory where all processes write their shards.
    Parent should set BBB_ANALYTICS_DIR to a fresh directory per run.
    """
    d = os.environ.get("BBB_ANALYTICS_DIR")
    if not d:
        d = os.path.join("analytics_shards", f"run_{_now_iso()}_{uuid.uuid4().hex[:6]}")
    os.makedirs(d, exist_ok=True)
    return d

def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # os.replace is atomic on POSIX & Windows
    os.replace(tmp, path)


# ---------- logger ----------

class CardOutcomeLogger:
    """
    Per-process analytics logger.

    Log from your engine whenever the brain evaluates the *chosen* card
    (and optionally other candidates). We keep *counters* in memory and
    write a small shard file at process exit (atexit) or when you call dump().

    Stats per card_key:
        evals                -> total log_eval calls
        sum_wins_evals       -> sum of wins counts (all evals)
        sum_losses_evals     -> sum of losses counts (all evals)
        plays                -> how many times this card_was played
        sum_wins_played      -> sum wins counts on the specific play
        sum_losses_played    -> sum losses counts on the specific play
        certain_evals        -> evals where outcome was "certain"
        certain_plays        -> plays where outcome was "certain"
    """
    def __init__(self, run_dir: Optional[str] = None) -> None:
        self.run_dir = run_dir or _run_dir_from_env()
        self.pid = os.getpid()
        self.shard_path = os.path.join(self.run_dir, f"card_outcomes_shard_{self.pid}.json")
        self._lock = threading.Lock()
        self._stats: Dict[Card, Dict[str, int]] = {}
        atexit.register(self.dump)

    def log_eval(
        self,
        *,
        card: Card,
        wins_against: int,
        losses_against: int,
        total_candidates: Optional[int] = None,
        played: bool = True,
        certain: Optional[bool] = None,
    ) -> None:
        """
        Call this when you have evaluated a card (ideally the chosen/played one).

        - card_key: stable identifier for the card (e.g., card.id or str(card))
        - wins_against / losses_against: counts from your brain's calc
        - total_candidates: optional, for certainty heuristic
        - played: True if this eval corresponds to the card that was actually played
        - certain: optionally pass your own certainty boolean; if None we infer:
              (wins==0 or losses==0) or (total_candidates is not None and wins+losses==total_candidates)
        """
        if certain is None:
            if total_candidates is not None:
                certain = (wins_against == 0 or losses_against == 0) or \
                          ((wins_against + losses_against) == total_candidates)
            else:
                certain = (wins_against == 0 or losses_against == 0)

        with self._lock:
            s = self._stats.get(card)
            if s is None:
                s = {
                    "evals": 0,
                    "sum_wins_evals": 0,
                    "sum_losses_evals": 0,
                    "plays": 0,
                    "sum_wins_played": 0,
                    "sum_losses_played": 0,
                    "certain_evals": 0,
                    "certain_plays": 0,
                }
                self._stats[card] = s

            s["evals"] += 1
            s["sum_wins_evals"] += int(wins_against)
            s["sum_losses_evals"] += int(losses_against)
            if certain:
                s["certain_evals"] += 1

            if played:
                s["plays"] += 1
                s["sum_wins_played"] += int(wins_against)
                s["sum_losses_played"] += int(losses_against)
                if certain:
                    s["certain_plays"] += 1

    def dump(self) -> None:
        """Write this process's shard as JSON (atomic replace)."""
        with self._lock:
            payload = {
                "schema": 1,
                "pid": self.pid,
                "written_at": _now_iso(),
                "stats": self._stats,
            }
        _atomic_write_json(self.shard_path, payload)


# Singleton helper per process
_logger_singleton: Optional[CardOutcomeLogger] = None

def get_card_logger(run_dir: Optional[str] = None) -> CardOutcomeLogger:
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = CardOutcomeLogger(run_dir=run_dir)
    return _logger_singleton


# ---------- reducer / reports ----------

def _merge_stats(dst: Dict[Card, Dict[str, int]], src: Dict[Card, Dict[str, int]]) -> None:
    for k, s in src.items():
        d = dst.setdefault(k, {
            "evals": 0, "sum_wins_evals": 0, "sum_losses_evals": 0,
            "plays": 0, "sum_wins_played": 0, "sum_losses_played": 0,
            "certain_evals": 0, "certain_plays": 0,
        })
        for field in d.keys():
            d[field] += int(s.get(field, 0))

def aggregate_card_outcomes(shards_dir: str) -> Dict[Card, Dict[str, float]]:
    """
    Read all shard files in `shards_dir`, merge, and compute derived metrics.
    Returns dict keyed by card_key with:
      - plays, evals, certain_plays, certain_rate
      - avg_beats_per_play, avg_losses_per_play
    """
    merged: Dict[Card, Dict[str, int]] = {}
    for name in os.listdir(shards_dir):
        if not name.startswith("card_outcomes_shard_") or not name.endswith(".json"):
            continue
        path = os.path.join(shards_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            stats = payload.get("stats", {})
            _merge_stats(merged, stats)
        except Exception:
            # skip corrupt file
            continue

    result: Dict[Card, Dict[str, float]] = {}
    for card, s in merged.items():
        plays = int(s.get("plays", 0))
        wins_played = int(s.get("sum_wins_played", 0))
        losses_played = int(s.get("sum_losses_played", 0))
        certain_plays = int(s.get("certain_plays", 0))
        evals = int(s.get("evals", 0))

        avg_beats = (wins_played / plays) if plays else 0.0
        avg_losses = (losses_played / plays) if plays else 0.0
        certain_rate = (certain_plays / plays) if plays else 0.0

        result[card] = {
            "plays": float(plays),
            "evals": float(evals),
            "avg_beats_per_play": avg_beats,
            "avg_losses_per_play": avg_losses,
            "certain_plays": float(certain_plays),
            "certain_rate": certain_rate,
        }
    return result

def write_card_outcomes_csv(out_csv: str, agg: Dict[Card, Dict[str, float]]) -> None:
    cols = ["card_key", "plays", "evals", "avg_beats_per_play", "avg_losses_per_play", "certain_plays", "certain_rate"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for card, row in sorted(agg.items()):
            w.writerow([card] + [row[c] for c in cols if c != "card_key"])

def plot_card_outcomes_bars(out_png: str, agg: Dict[Card, Dict[str, float]], top_k: Optional[int] = None) -> None:
    """
    Grouped bars per card: avg_beats_per_play vs avg_losses_per_play.
    If there are many cards, you can pass top_k to limit by plays (most-played first).
    """
    items = list(agg.items())
    # sort by plays desc
    items.sort(key=lambda kv: kv[1].get("plays", 0.0), reverse=True)
    if top_k:
        items = items[:top_k]

    cards = [k for k, _ in items]
    beats = [v["avg_beats_per_play"] for _, v in items]
    losses = [v["avg_losses_per_play"] for _, v in items]

    import numpy as np  # only used here; safe dependency
    x = np.arange(len(cards))
    width = 0.4

    plt.figure(figsize=(max(8, len(cards) * 0.4), 6))
    plt.bar(x - width/2, beats, width, label="Avg beats/play")
    plt.bar(x + width/2, losses, width, label="Avg losses/play")
    plt.ylabel("Cards")
    plt.title("Per-card outcomes (averaged over plays)")
    plt.xticks(x, cards, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_card_certainty_bars(out_png: str, agg: Dict[Card, Dict[str, float]], top_k: Optional[int] = None) -> None:
    """One bar per card: certain_rate (%) with value labels."""
    items = list(agg.items())
    items.sort(key=lambda kv: kv[1].get("plays", 0.0), reverse=True)
    if top_k:
        items = items[:top_k]
    cards = [k for k, _ in items]
    rates = [v["certain_rate"] * 100.0 for _, v in items]
    x = range(len(cards))
    plt.figure(figsize=(max(8, len(cards) * 0.4), 6))
    bars = plt.bar(x, rates)
    for i, h in enumerate(rates):
        plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Certain on play (%)")
    plt.title("Per-card certainty rate")
    plt.xticks(list(x), cards, rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
