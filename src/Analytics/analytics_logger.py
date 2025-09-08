# BBB/src/analytics_logger.py
from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
import uuid
from typing import Dict, Optional, Sequence, Tuple, List
from bbb.models import Card

import matplotlib.pyplot as plt
import numpy as np
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
    os.replace(tmp, path)  # atomic on POSIX & Windows


# ---------- logger ----------

# Internal stat template per card
def _empty_card_stats() -> Dict[str, int]:
    return {
        # evaluations (all times we computed wins/loses for this card in this stage)
        "evals": 0,
        "sum_wins_evals": 0,
        "sum_losses_evals": 0,
        "sum_total_candidates_evals": 0,
        "certain_evals": 0,

        # "played" (or "bet-placed") events in this stage
        "plays": 0,
        "sum_wins_played": 0,
        "sum_losses_played": 0,
        "sum_total_candidates_played": 0,
        "certain_plays": 0,
    }

class CardOutcomeLogger:
    """
    Per-process analytics logger with two separate channels:
      - 'play' : card selection (which card to play)
      - 'bet'  : additional bets evaluation/placement stage

    Use log_eval_play(...) during the PLAY stage, and log_eval_bet(...) during the BET stage.

    We keep *counters* in memory and write a small shard file at process exit (atexit)
    so multiprocessing has zero contention.

    Shard schema (v2):
    {
      "schema": 2,
      "pid": <int>,
      "written_at": "<iso>",
      "stats": {
        "play": { "<card_key>": {...}, ... },
        "bet":  { "<card_key>": {...}, ... }
      }
    }
    """

    def __init__(self, run_dir: Optional[str] = None) -> None:
        self.run_dir = run_dir or _run_dir_from_env()
        self.pid = os.getpid()
        self.shard_path = os.path.join(self.run_dir, f"card_outcomes_shard_{self.pid}.json")
        self._lock = threading.Lock()
        # two namespaces: play & bet
        self._stats: Dict[str, Dict[str, Dict[str, int]]] = {"play": {}, "bet": {}}
        atexit.register(self.dump)

    # -------- public API (two channels) --------

    def log_eval_play(
        self,
        *,
        card_key: str,
        wins_against: int,
        losses_against: int,
        total_candidates: Optional[int] = None,
        played: bool = False,
        certain: Optional[bool] = None,
    ) -> None:
        """Record evaluation for the PLAY stage."""
        self._log(
            kind="play",
            card_key=card_key,
            wins_against=wins_against,
            losses_against=losses_against,
            total_candidates=total_candidates,
            acted=played,
            certain=certain,
        )

    def log_eval_bet(
        self,
        *,
        card_key: str,
        wins_against: int,
        losses_against: int,
        total_candidates: Optional[int] = None,
        bet_placed: bool = False,
        certain: Optional[bool] = None,
    ) -> None:
        """Record evaluation for the BET stage."""
        self._log(
            kind="bet",
            card_key=card_key,
            wins_against=wins_against,
            losses_against=losses_against,
            total_candidates=total_candidates,
            acted=bet_placed,
            certain=certain,
        )

    # -------- core implementation --------

    def _log(
        self,
        *,
        kind: str,  # "play" or "bet"
        card_key: str,
        wins_against: int,
        losses_against: int,
        total_candidates: Optional[int],
        acted: bool,   # played OR bet_placed
        certain: Optional[bool],
    ) -> None:
        with self._lock:
            ns = self._stats[kind]
            s = ns.get(card_key)
            if s is None:
                s = _empty_card_stats()
                ns[card_key] = s

            # evaluations
            s["evals"] += 1
            s["sum_wins_evals"] += int(wins_against)
            s["sum_losses_evals"] += int(losses_against)
            if total_candidates is not None:
                s["sum_total_candidates_evals"] += int(total_candidates)
            if certain:
                s["certain_evals"] += 1

            # acted (played or bet_placed)
            if acted:
                s["plays"] += 1
                s["sum_wins_played"] += int(wins_against)
                s["sum_losses_played"] += int(losses_against)
                if total_candidates is not None:
                    s["sum_total_candidates_played"] += int(total_candidates)
                if certain:
                    s["certain_plays"] += 1

    def dump(self) -> None:
        """Write this process's shard as JSON (atomic replace)."""
        with self._lock:
            payload = {
                "schema": 2,  # bumped from v1
                "pid": self.pid,
                "written_at": _now_iso(),
                "stats": self._stats,
            }
        _atomic_write_json(self.shard_path, payload)


# Singleton per process
_logger_singleton: Optional[CardOutcomeLogger] = None

def get_card_logger(run_dir: Optional[str] = None) -> CardOutcomeLogger:
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = CardOutcomeLogger(run_dir=run_dir)
    return _logger_singleton


# ---------- reducer / reports ----------

def _merge_card_maps(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]) -> None:
    for k, s in src.items():
        d = dst.setdefault(k, _empty_card_stats())
        for field in d.keys():
            d[field] += int(s.get(field, 0))

def _read_shard(path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Returns a dict with keys 'play' and 'bet' mapping to their card maps.
    Supports schema v2 (preferred) and v1 (legacy: only a flat map -> treated as 'play').
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    schema = int(payload.get("schema", 1))
    stats = payload.get("stats", {})
    if schema >= 2 and isinstance(stats, dict) and "play" in stats and "bet" in stats:
        play_map = stats.get("play", {}) or {}
        bet_map = stats.get("bet", {}) or {}
        return {"play": play_map, "bet": bet_map}
    else:
        # legacy shard (v1): interpret as 'play' only
        return {"play": stats, "bet": {}}

def aggregate_card_outcomes(shards_dir: str, *, kind: str = "play") -> Dict[str, Dict[str, float]]:
    """
    Merge shards for a given kind ('play' or 'bet') and compute derived metrics per card:

      plays, evals, certain_plays, certain_rate,
      avg_beats_per_play, avg_losses_per_play,
      avg_total_candidates_per_eval, avg_total_candidates_per_play

    Returns: { card_key: {...metrics...}, ... }
    """
    assert kind in ("play", "bet")
    merged: Dict[str, Dict[str, int]] = {}

    for name in os.listdir(shards_dir):
        if not name.startswith("card_outcomes_shard_") or not name.endswith(".json"):
            continue
        path = os.path.join(shards_dir, name)
        try:
            shard = _read_shard(path)
            _merge_card_maps(merged, shard.get(kind, {}) or {})
        except Exception:
            continue  # skip unreadable shard

    result: Dict[str, Dict[str, float]] = {}
    for card, s in merged.items():
        plays = int(s.get("plays", 0))
        evals = int(s.get("evals", 0))
        wins_played = int(s.get("sum_wins_played", 0))
        losses_played = int(s.get("sum_losses_played", 0))
        certain_plays = int(s.get("certain_plays", 0))
        sum_tot_eval = int(s.get("sum_total_candidates_evals", 0))
        sum_tot_play = int(s.get("sum_total_candidates_played", 0))

        avg_beats = (wins_played / plays) if plays else 0.0
        avg_losses = (losses_played / plays) if plays else 0.0
        avg_tot_eval = (sum_tot_eval / evals) if evals else 0.0
        avg_tot_play = (sum_tot_play / plays) if plays else 0.0

        # sums used to compute per-eval averages and win-rates
        sum_wins_evals = int(s.get("sum_wins_evals", 0))
        sum_losses_evals = int(s.get("sum_losses_evals", 0))
        avg_wins_per_eval = (sum_wins_evals / evals) if evals else 0.0
        avg_losses_per_eval = (sum_losses_evals / evals) if evals else 0.0

        certain_evals = int(s.get("certain_evals", 0))
        certain_rate_play = (certain_plays / plays) if plays else 0.0
        certain_rate_eval = (certain_evals / evals) if evals else 0.0

        # win-rate (%), computed from sums (not averages) to be precise
        total_eval_comparisons = (sum_wins_evals + sum_losses_evals)
        winrate_eval_percent = (100.0 * sum_wins_evals / total_eval_comparisons) if total_eval_comparisons > 0 else 0.0

        total_play_comparisons = (wins_played + losses_played)
        winrate_play_percent = (100.0 * wins_played / total_play_comparisons) if total_play_comparisons > 0 else 0.0

        result[card] = {
            # counts
            "plays": float(plays),
            "evals": float(evals),
            "certain_plays": float(certain_plays),
            "certain_evals": float(certain_evals),

            # per-play
            "avg_beats_per_play": avg_beats,
            "avg_losses_per_play": avg_losses,
            "certain_rate_play": certain_rate_play,
            "avg_total_candidates_per_play": avg_tot_play,
            "winrate_play_percent": winrate_play_percent,

            # per-eval
            "avg_wins_per_eval": avg_wins_per_eval,
            "avg_losses_per_eval": avg_losses_per_eval,
            "certain_rate_eval": certain_rate_eval,
            "avg_total_candidates_per_eval": avg_tot_eval,
            "winrate_eval_percent": winrate_eval_percent,
        }
    return result

def aggregate_both(shards_dir: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Convenience: returns (play_metrics, bet_metrics)."""
    return aggregate_card_outcomes(shards_dir, kind="play"), aggregate_card_outcomes(shards_dir, kind="bet")


_card_re = re.compile(r"^Card\(\s*([^,]+)\s*,\s*([0-9]+)\s*\)$")
def _parse_card_key(card_key: str) -> tuple[str, int, str]:
    """
    Extract (type_name, number, original_key) from keys like 'Card(Type, 7)'.
    Falls back to (card_key, very_large_int, card_key) if it doesn't match.
    """
    m = _card_re.match(card_key)
    if not m:
        return (card_key, 10**9, card_key)
    type_name = m.group(1).strip()
    number = int(m.group(2))
    return (type_name, number, card_key)

def _sorted_items_by_type_number(agg: dict) -> list[tuple[str, dict]]:
    items = list(agg.items())
    items.sort(key=lambda kv: _parse_card_key(kv[0]))
    return items


def write_card_outcomes_csv(out_csv: str, agg: Dict[str, Dict[str, float]]) -> None:
    """
    Write a wide CSV reflecting all plotted metrics.
    Rows are sorted by card type (X) then number (Y) for keys like 'Card(X, Y)'.
    Percent fields are numeric (0..100), not strings, so they’re easy to chart in Excel.
    """
    # Columns mirror the aggregator’s result dict
    cols = [
        "card_key",
        # counts
        "plays", "evals", "certain_plays", "certain_evals",
        # per-play
        "avg_beats_per_play", "avg_losses_per_play",
        "certain_rate_play_percent", "winrate_play_percent",
        "avg_total_candidates_per_play",
        # per-eval
        "avg_wins_per_eval", "avg_losses_per_eval",
        "certain_rate_eval_percent", "winrate_eval_percent",
        "avg_total_candidates_per_eval",
    ]

    # Prepare sorted rows
    items = _sorted_items_by_type_number(agg)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for card_key, row in items:
            plays  = float(row.get("plays", 0.0))
            evals  = float(row.get("evals", 0.0))

            # per-play metrics
            avg_beats_play  = float(row.get("avg_beats_per_play", 0.0))
            avg_losses_play = float(row.get("avg_losses_per_play", 0.0))
            certain_rate_play_percent = float(row.get("certain_rate_play", 0.0)) * 100.0
            winrate_play_percent      = float(row.get("winrate_play_percent", 0.0))
            avg_tot_play   = float(row.get("avg_total_candidates_per_play", 0.0))
            certain_plays  = float(row.get("certain_plays", 0.0))

            # per-eval metrics
            avg_wins_eval  = float(row.get("avg_wins_per_eval", 0.0))
            avg_losses_eval= float(row.get("avg_losses_per_eval", 0.0))
            certain_rate_eval_percent = float(row.get("certain_rate_eval", 0.0)) * 100.0
            winrate_eval_percent      = float(row.get("winrate_eval_percent", 0.0))
            avg_tot_eval   = float(row.get("avg_total_candidates_per_eval", 0.0))
            certain_evals  = float(row.get("certain_evals", 0.0))

            w.writerow([
                card_key,
                plays, evals, certain_plays, certain_evals,
                avg_beats_play, avg_losses_play,
                certain_rate_play_percent, winrate_play_percent,
                avg_tot_play,
                avg_wins_eval, avg_losses_eval,
                certain_rate_eval_percent, winrate_eval_percent,
                avg_tot_eval,
            ])


def _save_empty(out_png: str, msg: str = "No data") -> None:
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# --- PLAY: per-play beats & losses ---
def plot_play_per_play_beats_losses(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("plays", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No plays logged for PLAY stage")

    cards = [k for k, _ in items]
    left  = [v["avg_beats_per_play"]  for _, v in items]
    right = [v["avg_losses_per_play"] for _, v in items]

    x = np.arange(len(cards)); w = 0.4
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x - w/2, left,  w, label="Avg beats / play")
    plt.bar(x + w/2, right, w, label="Avg losses / play")
    plt.ylabel("Cards"); plt.title("Phase 4 | Choose Card - per-play outcomes")
    plt.xticks(x, cards, rotation=45, ha="right"); plt.legend(); plt.tight_layout()

    # value labels
    ymax = max(left + right) if (left or right) else 0.0
    dy = 0.01 * ymax if ymax > 0 else 0.05  # tiny offset
    for i, v in enumerate(left):
        plt.text(x[i] - w/2, v + dy, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(right):
        plt.text(x[i] + w/2, v + dy, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# --- PLAY: per-play certainty % ---
def plot_play_per_play_certainty(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("plays", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No plays logged for PLAY stage")
    cards = [k for k, _ in items]
    heights = [v["certain_rate_play"] * 100.0 for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Certain on play (%)"); plt.title("Phase 4 | Choose Card - certainty rate per play")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- PLAY: per-play win-rate % ---
def plot_play_per_play_winrate(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("plays", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No plays logged for PLAY stage")
    cards = [k for k, _ in items]
    heights = [v["winrate_play_percent"] for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Win rate on play (%)"); plt.title("Phase 4 | Choose Card - win rate (plays)")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- EVAL: average wins per eval (PLAY) ---
def plot_play_eval_avg_wins(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for PLAY stage")

    cards = [k for k, _ in items]
    heights = [v["avg_wins_per_eval"] for _, v in items]
    x = np.arange(len(cards))

    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Avg wins / eval"); plt.title("Phase 4 | Choose Card - average wins per eval")
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()

    # value labels
    ymax = max(heights) if heights else 0.0
    dy = 0.01 * ymax if ymax > 0 else 0.05
    for i, h in enumerate(heights):
        plt.text(i, h + dy, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# --- EVAL: certainty % per eval (PLAY) ---
def plot_play_eval_certainty(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for PLAY stage")
    cards = [k for k, _ in items]
    heights = [v["certain_rate_eval"] * 100.0 for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Certain on eval (%)"); plt.title("Phase 4 | Choose Card - certainty rate per eval")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- EVAL: win-rate % per eval (PLAY) ---
def plot_play_eval_winrate(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for PLAY stage")
    cards = [k for k, _ in items]
    heights = [v["winrate_eval_percent"] for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Win rate on eval (%)"); plt.title("Phase 4 | Choose Card - win rate (evals)")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- EVAL: average wins per eval (BET) ---
def plot_bet_eval_avg_wins(out_png: str, agg_bet: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_bet) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for BET stage")
    cards = [k for k, _ in items]
    heights = [v["avg_wins_per_eval"] for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Avg wins / eval"); plt.title("Phase 4 | Place Additional Bets - average wins per eval")
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    ymax = max(heights) if heights else 0.0
    dy = 0.01 * ymax if ymax > 0 else 0.05
    for i, h in enumerate(heights):
        plt.text(i, h + dy, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- EVAL: average losses per eval (PLAY) — NEW (with value labels) ---
def plot_play_eval_avg_losses(out_png: str, agg_play: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_play) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for PLAY stage")

    cards = [k for k, _ in items]
    heights = [v["avg_losses_per_eval"] for _, v in items]
    x = np.arange(len(cards))

    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Avg losses / eval"); plt.title("Phase 4 | Choose Card - average losses per eval")
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()

    # value labels
    ymax = max(heights) if heights else 0.0
    dy = 0.01 * ymax if ymax > 0 else 0.05
    for i, h in enumerate(heights):
        plt.text(i, h + dy, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# --- EVAL: certainty % per eval (BET) ---
def plot_bet_eval_certainty(out_png: str, agg_bet: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_bet) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for BET stage")
    cards = [k for k, _ in items]
    heights = [v["certain_rate_eval"] * 100.0 for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Certain on eval (%)"); plt.title("Phase 4 | Place Additional Bets - certainty rate per eval")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()

# --- EVAL: win-rate % per eval (BET) ---
def plot_bet_eval_winrate(out_png: str, agg_bet: dict, top_k: int | None = None) -> None:
    items = [kv for kv in _sorted_items_by_type_number(agg_bet) if kv[1].get("evals", 0) > 0]
    if top_k: items = items[:top_k]
    if not items: return _save_empty(out_png, "No evals logged for BET stage")
    cards = [k for k, _ in items]
    heights = [v["winrate_eval_percent"] for _, v in items]
    x = np.arange(len(cards))
    plt.figure(figsize=(max(8, len(cards)*0.45), 6))
    plt.bar(x, heights)
    plt.ylabel("Win rate on eval (%)"); plt.title("Phase 4 | Place Additional Bets - win rate (evals)")
    for i, h in enumerate(heights): plt.text(i, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, cards, rotation=45, ha="right"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=160); plt.close()