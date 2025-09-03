from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, random
from ..models import Card, GladiatorType, Battle
from ..observations import PlayerView
from colorama import Fore, Back, Style
from .utils import estimate_future_representation_open_lane
from ..globals import ADDITIONAL_INFO, TARGET_PLAYER, NUMBER_OF_BATTLES, NUM_PLAYERS, FOCUS_ON_BET_SIZING, FOCUS_ON_CARD_PLAY,FOCUS_ON_BATTLE_INITIAL_BET, FOCUS_ON_ADDITIONAL_BETS, GAME_ENGINE_PIRINTS, FOCUS_ON_EQUALIZING_BETS, WW_VARS, WW_VARS_TRAINING
from collections import Counter
from typing import Iterable
from enum import Enum as _Enum

class TDIntent(_Enum):
    NOW = 1
    LATER = 2
    NONE = 3

# -------- Traits --------
@dataclass(frozen=True)
class Traits:
    aggressiveness: int = 50       # bet sizing tendency (0..100)
    risk_tolerance: int = 50       # accept negative EV for variance
    tempo: int = 50                # play good cards early
    bluffiness: int = 50           # prefer misleading reveals / thin bets
    stubbornness: int = 50         # willingness to match during equalization
    domination_drive: int = 50     # weight of 3/3 dominance goal
    herding: int = 50              # tendency to follow board multipliers
    ev_adherence: int = 50         # 0: vibes, 100: strict EV
    exploration: int = 3          # random exploration

THREE_X_CHAIN = [GladiatorType.A, GladiatorType.B, GladiatorType.C, GladiatorType.D, GladiatorType.E]
TWO_X_CHAIN   = [GladiatorType.A, GladiatorType.C, GladiatorType.E, GladiatorType.B, GladiatorType.D]

def _dom_mult(a: GladiatorType, b: GladiatorType) -> int:
    """Return 3 if a strongly > b, 2 if a weakly > b, else 1."""
    i3 = THREE_X_CHAIN.index(a)
    if THREE_X_CHAIN[(i3 + 1) % 5] == b:
        return 3
    i2 = TWO_X_CHAIN.index(a)
    if TWO_X_CHAIN[(i2 + 1) % 5] == b:
        return 2
    return 1

def _softmax_pick(scores: Dict[GladiatorType, float], rng: random.Random, temperature: float) -> GladiatorType:
    # avoid explosions, shift by max
    mx = max(scores.values()) if scores else 0.0
    exps = {t: math.exp((s - mx) / max(1e-6, temperature)) for t, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    r = rng.random()
    acc = 0.0
    for t, w in exps.items():
        acc += w / Z
        if r <= acc:
            return t
    return next(iter(scores))  # fallback


class DeckMemory:
    """
    Tracks remaining unseen cards globally from this player's perspective.
    Start with full deck; remove your hand and any revealed cards as game proceeds.
    """
    ALL_VALUES = [1,3,5,5,7,7,9,10,11]

    def __init__(self):
        self.remaining = Counter()
        for t in GladiatorType:
            for v in self.ALL_VALUES:
                self.remaining[(t, v)] += 1

    def remove_cards(self, cards: Iterable[Card]):
        for c in cards:
            if c is None:
                continue
            self.remaining[(c.type, c.number)] = max(0, self.remaining[(c.type, c.number)] - 1)

    def count_total(self) -> int:
        return sum(self.remaining.values())

    def possible_given_shown(self, *, shown_type: GladiatorType | None, shown_number: int | None) -> list[tuple[GladiatorType,int,int]]:
        """
        Returns list of (type, number, count) that match the partial info:
        - if shown_type is set, restrict to that type
        - if shown_number is set, restrict to that number
        """
        out = []
        for (t, v), cnt in self.remaining.items():
            if cnt <= 0:
                continue
            if shown_type is not None and t != shown_type:
                continue
            if shown_number is not None and v != shown_number:
                continue
            out.append((t, v, cnt))
        return out



# -------- Brain interface --------
class PlayerBrain:
    """
    Pluggable decision-maker. All methods are *pure* in the sense they return
    intents; the engine applies side effects.
    """

    def __init__(self, rng: Optional[random.Random] = None, traits: Optional[Traits] = None):
        self.rng = rng or random.Random()
        self.traits = traits or Traits()
        self.brain_memory = DeckMemory()

    # Phase 1
    def pick_favored_faction(self, view: "PlayerView") -> str:
        """
        Decide favored faction (an opponent's name) using a single weight + risk_tolerance:
          - Risky route: favor the last-to-act player this round (latest in turn order).
          - Safe route: favor the richest opponent; if tie, pick the one latest in turn order.
        """
        # Opponents in the current turn order (view.players is ordered)
        others = [name for name in view.players if name != view.me]
        if not others:
            return ""  # no opponents (edge case)

        # --- Identify candidates ---
        # Last to act = last name in 'others' (since 'players' is in turn order)
        last_to_act = others[-1]

        # Richest total = behind + front
        # Build a list of (name, total) respecting turn order in 'others'
        totals = [(name, view.others_bankrolls.get(name).behind) for name in others if name in view.others_bankrolls]
        if not totals:
            # If bankroll info missing, default to last_to_act
            return last_to_act

        max_total = max(t for _, t in totals)
        # Keep only tied richest, preserving order; then take the latest (last) among them
        richest_tied_in_order = [name for (name, t) in totals if t == max_total]
        richest_latest = richest_tied_in_order[-1]

        # --- Blend via risk tolerance and a single weight ---
        risk = (getattr(self, "traits", None).risk_tolerance / 100.0) if getattr(self, "traits", None) else 0.5
        ww_last_bias = WW_VARS.get("ww_last_bias") if not view.training_target else WW_VARS_TRAINING.get("ww_last_bias")

        # Logistic switch centered at risk=0.5
        x = (risk - 0.5) * 2.0  # map risk to [-1, +1]
        p_choose_last = 1.0 / (1.0 + math.exp(-ww_last_bias * x))

        # Sample the route
        if self.rng.random() < p_choose_last:
            return last_to_act
        else:
            return richest_latest

    # Phase 3 (representation)
    def pick_representation_bet(self, view: PlayerView) -> Tuple[GladiatorType, int]:
        rng = self.rng
        traits = self.traits

        behind = view.my_bankroll.behind
        if behind <= 0:
            # Must bet at least 1 by rules; if broke, still return minimal legal shape
            # (Engine can clamp later if needed.)
            return rng.choice(list(GladiatorType)), 1

        # --- 1) Hand support per type ---
        hand_counts: Dict[GladiatorType, int] = {t: 0 for t in GladiatorType}
        hand_strength: Dict[GladiatorType, int] = {t: 0 for t in GladiatorType}
        for c in view.my_hand.cards:
            hand_counts[c.type] += 1
            hand_strength[c.type] += c.number

        # Normalize hand support: combine count and value
        # Weight count slightly more to reflect multiple deployments across battles
        ww_hand_count = WW_VARS.get("ww_hand_count") if not view.training_target else WW_VARS_TRAINING.get("ww_hand_count")
        ww_hand_strength = WW_VARS.get("ww_hand_strength") if not view.training_target else WW_VARS_TRAINING.get("ww_hand_strength")
        hand_support: Dict[GladiatorType, float] = {}
        for t in GladiatorType:
            hand_support[t] = ww_hand_count * hand_counts[t] + ww_hand_strength * hand_strength[t] 

        # --- 2) Counter power versus existing board bets (public) ---
        # If others have bet on types U with amounts a_U, favor types T that beat U
        counter_power: Dict[GladiatorType, float] = {t: 0.0 for t in GladiatorType}
        for u_type, entries in view.board.raw_bets.items():
            total_amt = sum(a for _, a in entries)
            if total_amt <= 0:
                continue
            for t in GladiatorType:
                mult = _dom_mult(t, u_type)  # 3, 2, or 1
                # Only reward superiority over u_type (mult > 1)
                counter_power[t] += (mult - 1) * total_amt

        # --- 3) Open-lane bonus (favor strong hand types with few/no current bets) ---
        totals = view.board.total_bets_by_type
        max_total = max(totals.values(), default=0)
        # crowding in [0,1]; if board is empty, crowding=0 for all types
        crowding = {
            t: (totals.get(t, 0) / max(1, max_total)) if max_total > 0 else 0.0
            for t in GladiatorType
        }
        # Open lane = 1 - crowding. Scale by our hand strength for that type.
        open_lane_bonus = {t: (1.0 - crowding[t]) * hand_support[t] for t in GladiatorType}

        # Weight: stronger when herding is low, weaker when herding is high
        # herding=0  -> w_open ≈ 0.8 (strongly prefer empty lanes)
        # herding=100-> w_open ≈ 0.0 (ignore open lanes; just follow the board)

        # --- 4) Deception pressure: penalize "most obvious" type if bluffiness high ---
        # Find our currently strongest-by-hand type
        best_hand_type = max(GladiatorType, key=lambda t: hand_support[t]) if view.my_hand.cards else rng.choice(list(GladiatorType))
        deception_penalty: Dict[GladiatorType, float] = {t: 0.0 for t in GladiatorType}
        # Apply a small penalty to best_hand_type proportional to bluffiness
        deception_penalty[best_hand_type] = 0.5 * (traits.bluffiness / 100.0) # hard coded weight

        # --- Combine scores with trait-driven weights ---
        ww_hand   = WW_VARS.get("ww_hand")  if not view.training_target else WW_VARS_TRAINING.get("ww_hand")
        ww_open = WW_VARS.get("ww_open") if not view.training_target else WW_VARS_TRAINING.get("ww_open")
        ww_countp = WW_VARS.get("ww_countp")  if not view.training_target else WW_VARS_TRAINING.get("ww_countp")
        # Tilt countering more for higher risk_tolerance (seek edges vs others)
        counter = ww_countp * (0.5 + 0.5 * (traits.risk_tolerance / 100.0))
        ww_bluff = WW_VARS.get("ww_bluff") if not view.training_target else WW_VARS_TRAINING.get("ww_bluff")

        scores: Dict[GladiatorType, float] = {}
        for t in GladiatorType:
            s = 0.0
            s += ww_hand   * hand_support[t]
            s += counter * counter_power[t]
            s += (ww_open * (1.0 - traits.herding / 100.0)) * open_lane_bonus[t]
            s -= (ww_bluff * (traits.bluffiness / 100.0))  * deception_penalty[t]
            scores[t] = s
        if FOCUS_ON_BET_SIZING:
            print(f"Scores for {view.me}: {scores}")

        # --- Exploration: sometimes try something else entirely ---
        if rng.random() < (traits.exploration / 100.0):
            chosen_type = rng.choice(list(GladiatorType))
        else:
            # Temperature shaped by EV adherence (higher adherence = lower temperature)
            temperature = 1.5 - 1.3 * (traits.ev_adherence / 100.0)  # range ~[0.2, 1.5]
            temperature = max(0.15, min(2.5, temperature))
            chosen_type = _softmax_pick(scores, rng, temperature)

        # --- Bet sizing via EV-like scoring over a in [1..behind] ---
        # 1) who has already placed a representation bet?
        already_betters = {name for entries in view.board.raw_bets.values() for (name, _) in entries}

        # 2) seat order after me (wrap-around)
        try:
            my_idx = view.players.index(view.me)
        except ValueError:
            my_idx = 0
        ordered_after_me = view.players[my_idx+1:] + view.players[:my_idx]

        # 3) remaining players who haven't bet yet, in order
        remaining_names = [n for n in ordered_after_me if n not in already_betters]

        # 4) coins-behind for those remaining players
        remaining_behind = []
        for n in remaining_names:
            if n == view.me:
                # shouldn't happen, but guard anyway
                remaining_behind.append(view.my_bankroll.behind)
            else:
                pb = view.others_bankrolls.get(n)
                if pb is not None:
                    remaining_behind.append(pb.behind)
                else:
                    # if not visible, assume a small baseline so estimator stays sane
                    remaining_behind.append(3)

        # 5) list of actual bets already placed on the board (amounts only)
        existing_bet_amounts = [amt for entries in view.board.raw_bets.values() for (_, amt) in entries]

        # 6) choose amount using the imagined final board with open-lane/avg estimator
        amount = self._choose_bet_amount(
            view=view,
            chosen_type=chosen_type,
            remaining_behind=remaining_behind,
            existing_bet_amounts=existing_bet_amounts,
        )

        return chosen_type, amount


    def _choose_bet_amount(self, view: "PlayerView", chosen_type: "GladiatorType", remaining_behind: List[int] | None = None, existing_bet_amounts: List[int] | None = None) -> int:
        """
        Score a in [1..behind] using:
          Score(a) = (Btot* - BT*) + BattleEV(a) + DominationEV(a) - Liquidity(a) + AggBoost(a)
        where * uses an imagined final board built by predicting remaining players:
          - each picks an open lane type
          - amount ~ bell curve around running average
        """
        behind = view.my_bankroll.behind
        if behind <= 0:
            return 1

        base_totals = dict(view.board.total_bets_by_type)
        if existing_bet_amounts is None:
            # Extract actual representation bet amounts currently on board
            existing_bet_amounts = [amt for entries in view.board.raw_bets.values() for _, amt in entries]
        if remaining_behind is None:
            # If you don't have seat-order info handy, you can pass an estimate or an empty list.
            remaining_behind = []

        # Hand stats for chosen type
        T_cards = [c.number for c in view.my_hand.cards if c.type == chosen_type]
        NT = len(T_cards)
        Vavg = (sum(T_cards) / NT) if NT > 0 else 0.0

        scores: Dict[int, float] = {}
        for a in range(1, behind + 1):
            # Predict remaining bets to build a complete board for this candidate 'a'
            added, final_totals, bet_amounts = estimate_future_representation_open_lane(
                rng=self.rng,
                base_totals=base_totals,
                existing_bet_amounts=existing_bet_amounts,
                remaining_behind=remaining_behind,
                my_type=chosen_type,
                my_a=a,
                mean_floor=1.5,
                std_frac=0.25,
                max_bankroll_frac=0.40,
            )
            # ========= 1) ROUND PRIZE (shaped) =========
            # Raw: (Btot* - BT*) can explode and dominate. Use a concave transform to tame it.
            BT_star = final_totals.get(chosen_type, 0) 
            Btot_star = sum(final_totals.values())
            gain =  Btot_star - BT_star - a
            ww_round = WW_VARS.get("ww_round") if not view.training_target else WW_VARS_TRAINING.get("ww_round")
            s = ww_round * gain
            if FOCUS_ON_BET_SIZING:
                print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: gain={gain}" + Style.RESET_ALL)

            # ========= 2) BATTLE EV (inverted-U via concede) =========
            # Inputs for this type in our hand
            T_cards = [c.number for c in view.my_hand.cards if c.type == chosen_type]
            NT = len(T_cards)
            Vavg = (sum(T_cards) / NT) if NT > 0 else 0.0
            NT_eff = min(NT, 3)  # at most 3 battles

            # Compare our final multiplier to others' average multiplier
            other_types = [t for t in final_totals.keys() if t != chosen_type]
            Bavg_other = (sum(final_totals.get(t, 0) for t in other_types) / max(1, len(other_types))) if other_types else 0.0
            Δ = BT_star - Bavg_other

            # Win tendency increases with Δ (logistic), but opponents' concede probability also increases with Δ
            k_win = 0.5
            k_concede = 0.5
            Δ0_win = 0.0      # center (tune)
            Δ0_conc = 2.0     # concede starts to bite when you're clearly ahead (tune)

            p_win = 1.0 / (1.0 + math.exp(-k_win * (Δ - Δ0_win)))       # 0..1 increasing
            p_concede = 1.0 / (1.0 + math.exp(-k_concede * (Δ - Δ0_conc)))  # 0..1 increasing

            # Expected battle pot when not conceded: use a small baseline (tunable)
            # You can also tie this to average equalization capacity; keep simple for now.
            baseline_pot = 1.5

            # Inverted-U: EV ∝ win_prob * (1 - concede_prob)
            battle_ev = NT_eff * Vavg * (p_win * (1.0 - p_concede)) * baseline_pot
            ww_battle = WW_VARS.get("ww_battle")  if not view.training_target else WW_VARS_TRAINING.get("ww_battle")
            s += ww_battle * battle_ev
            if FOCUS_ON_BET_SIZING:
                print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: Δ={Δ:.2f} p_win={p_win:.2f} p_conc={p_concede:.2f} battleEV={battle_ev:.2f}" + Style.RESET_ALL)

            # ========= 3) DOMINATION EV =========
            # How many domination opportunities do we plausibly have this round from this type?
            opportunities = NT // 3  # as you suggested; simple and conservative
            if opportunities > 0:
                Mt = BT_star
                # per-battle success factor: allow concessions to help domination a bit,
                # but not fully (so overshooting still limited by S2)
                per_battle_success = min(1.0, p_win + 0.6 * p_concede)
                # If 3 battles exist, domination chance approx = per_battle_success^3 per opportunity group
                p3 = per_battle_success ** 3

                # S2 ≈ second-highest total; crude proxy: half of non-our-type money
                max_bet = 0
                second_max_bet = 0
                for amount in bet_amounts:
                    if amount >= max_bet:
                        second_max_bet = max_bet
                        max_bet = amount
                    elif amount > second_max_bet:
                        second_max_bet = amount
                S2 = second_max_bet
                dom_ev = opportunities * p3 * S2
                ww_dom = WW_VARS.get("ww_dom")  if not view.training_target else WW_VARS_TRAINING.get("ww_dom")
                s += ww_dom * dom_ev
                if FOCUS_ON_BET_SIZING:
                    print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: dom opp={opportunities} p3={p3:.2f} S2={S2:.2f} domEV={dom_ev:.2f}" + Style.RESET_ALL)

            # ========= 4) COSTS: explicit spend + reserve penalty =========
            #  Liquidity/reserve: steep penalty if 'a' starves equalization/battle bets
            lam = 0.6 + 0.4 * (self.traits.ev_adherence / 100.0)  # 0.6..1.0
            reserve_base = 1
            reserve_per_battle = 1
            Rmin = reserve_base + reserve_per_battle * NUMBER_OF_BATTLES   # assume 3 battles; wire your setting if available
            short = max(0, a - max(0, behind - Rmin))
            reserve_penalty = (2.0 + 2.0 * (self.traits.ev_adherence / 100.0)) * short
            ww_liquidty = WW_VARS.get("ww_liquidty")  if not view.training_target else WW_VARS_TRAINING.get("ww_liquidty")
            s -= (lam * a + reserve_penalty) * ww_liquidty
            if FOCUS_ON_BET_SIZING:
                print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: liq={lam*a:.2f} reserve_penalty={reserve_penalty:.2f}" + Style.RESET_ALL)

            # ========= 5) Aggressiveness: concave boost (optional, small) =========
            ww_theta = WW_VARS.get("ww_theta")  if not view.training_target else WW_VARS_TRAINING.get("ww_theta")
            s += ww_theta * (self.traits.aggressiveness / 100.0) * math.sqrt(a)
            if FOCUS_ON_BET_SIZING:
                print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: final score={s:.2f}" + Style.RESET_ALL)

            scores[a] = s

        # Exploration?
        if FOCUS_ON_BET_SIZING:
            print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"Scores for {view.me} with type {chosen_type}: {scores}" + Style.RESET_ALL)
        if self.rng.random() < (self.traits.exploration / 100.0):
            return self.rng.randint(1, behind)

        temperature = 1.5 - 1.3 * (self.traits.ev_adherence / 100.0)
        temperature = max(0.15, min(2.5, temperature))
        a_choice = _softmax_pick(scores, self.rng, temperature)
        return max(1, min(behind, int(a_choice)))

    def _multiplier(self, a: GladiatorType, b: GladiatorType) -> int:
        i3 = THREE_X_CHAIN.index(a)
        if THREE_X_CHAIN[(i3 + 1) % 5] == b:
            return 3
        i2 = TWO_X_CHAIN.index(a)
        if TWO_X_CHAIN[(i2 + 1) % 5] == b:
            return 2
        return 1

    def _card_strength(self, card: Card, board_totals: dict[GladiatorType,int]) -> int:
        return card.number * board_totals.get(card.type, 0)

    def _win_lose_stats(
        self,
        my_card: Card,
        opp_candidates: list[tuple[GladiatorType,int,int]],
        board_totals: dict[GladiatorType,int]
    ) -> tuple[int,int,int,int]:
        """
        Returns (wins, losses, total, sum_winning_values).
        - wins/losses/total count opponent-card multiplicities (cnt).
        - sum_winning_values accumulates the *opponent card numbers we beat* (weighted by cnt).
          This will let us compute avg value captured when we win.
        """
        wins = losses = total = 0
        sum_win_vals = 0
        my_base = self._card_strength(my_card, board_totals)

        for (t,v,cnt) in opp_candidates:
            opp_base = v * board_totals.get(t, 0)
            dom_me   = self._multiplier(my_card.type, t)
            dom_opp  = self._multiplier(t, my_card.type)
            s_me  = my_base * max(dom_me, 1)
            s_opp = opp_base * max(dom_opp, 1)

            if s_me > s_opp:
                wins += cnt
                sum_win_vals += v * cnt  # value of card we would take when we win
            elif s_opp > s_me:
                losses += cnt
            elif s_me == s_opp:
                if dom_me > 1 and dom_opp == 1:
                    wins += cnt
                    sum_win_vals += v * cnt
                elif dom_opp > 1 and dom_me == 1:
                    losses += cnt
            total += cnt

        return wins, losses, total, sum_win_vals

    def _ev_card_value(self, my_card: Card, opp_candidates, board_totals) -> float:
        """
        EV ≈ p_win * avg(value of cards we beat)  -  p_lose * my_card.number
        (ties ignored)
        """
        wins, losses, total, sum_win_vals = self._win_lose_stats(my_card, opp_candidates, board_totals)
        if FOCUS_ON_CARD_PLAY:
            print(Back.WHITE + Fore.LIGHTBLACK_EX + f"for {my_card} wins: {wins} losses: {losses} total: {total}" + Style.RESET_ALL)
        if total == 0:
            return 0.0
        p_win  = wins   / total
        p_lose = losses / total
        avg_win_val = (sum_win_vals / wins) if wins > 0 else 0.0
        return (p_win * avg_win_val) - (p_lose * my_card.number)

    def _opponent_rep_type(self, opp_name: str, board_raw_bets: dict[GladiatorType, list[tuple[str,int]]]) -> GladiatorType | None:
        # Scan board for (player_name, amount) entries; return the type they bet on.
        for gtype, entries in board_raw_bets.items():
            for pname, _amt in entries:
                if pname == opp_name:
                    return gtype
        return None

    def _decide_td_intent(self, view: "PlayerView", battles: list["Battle"], memory: "DeckMemory") -> tuple[TDIntent, float]:
        """
        Score feasibility to 3-0 with current board and partial info.
        Return (intent, intent_strength in [0,1]).
        """
        board_totals = view.board.total_bets_by_type
        my_cards = list(view.my_hand.cards)

        # For each battle, compute the best EV if we play NOW
        battle_scores: list[float] = []
        for b in battles:
            opp_shown_type = b.opp_card.type if (b.opp_card is not None and b.opp_faceup == 'type') else None
            opp_shown_value = b.opp_card.number if (b.opp_card is not None and b.opp_faceup == 'number') else None

            opp_cands = memory.possible_given_shown(shown_type=opp_shown_type, shown_number=opp_shown_value)

            best = 0.0
            for c in my_cards:
                ev = self._ev_card_value(c, opp_cands, board_totals)
                if ev > best:
                    best = ev
            battle_scores.append(best)

        top3 = sorted(battle_scores, reverse=True)[:3]
        raw = sum(max(0.0, x) for x in top3)
        norm = min(1.0, raw / 30.0)  # scale to [0,1] roughly

        # --- weights (ww_*) ---
        ww_td_norm_now   = WW_VARS.get("ww_td_norm_now") if not view.training_target else WW_VARS_TRAINING.get("ww_td_norm_now")
        ww_td_drive      = WW_VARS.get("ww_td_drive") if not view.training_target else WW_VARS_TRAINING.get("ww_td_drive")
        ww_td_tempo      = WW_VARS.get("ww_td_tempo") if not view.training_target else WW_VARS_TRAINING.get("ww_td_tempo")
        ww_td_norm_later = WW_VARS.get("ww_td_norm_later") if not view.training_target else WW_VARS_TRAINING.get("ww_td_norm_later")
        ww_td_drive_lat  = WW_VARS.get("ww_td_drive_lat") if not view.training_target else WW_VARS_TRAINING.get("ww_td_drive_lat")
        ww_td_tempo_lat  = WW_VARS.get("ww_td_tempo_lat") if not view.training_target else WW_VARS_TRAINING.get("ww_td_tempo_lat")
        ww_td_none_k     = WW_VARS.get("ww_td_none_k")  if not view.training_target else WW_VARS_TRAINING.get("ww_td_none_k")

        drive = self.traits.domination_drive / 100.0
        tempo = self.traits.tempo / 100.0

        logit_now   =  ww_td_norm_now * norm + ww_td_drive * drive + ww_td_tempo * tempo
        logit_later =  ww_td_norm_later * norm + ww_td_drive_lat * drive + ww_td_tempo_lat * (1.0 - tempo)
        logit_none  =  ww_td_none_k * (1.0 - norm)

        m = max(logit_now, logit_later, logit_none)
        ex = [math.exp(logit_now - m), math.exp(logit_later - m), math.exp(logit_none - m)]
        Z = sum(ex)
        p_now, p_later, p_none = (ex[0]/Z, ex[1]/Z, ex[2]/Z)

        r = self.rng.random()
        if r < p_now:
            return TDIntent.NOW, norm
        elif r < p_now + p_later:
            return TDIntent.LATER, norm
        else:
            return TDIntent.NONE, norm

    def _score_card_for_battle(
        self,
        view: "PlayerView",
        my_card: Card,
        opp_name: str,
        opp_cands,
        board_totals,
        td_intent: TDIntent,
        td_strength: float
    ) -> float:
        """
        Base EV + TD‑intent shaping + stubbornness‑weighted belief that opponent plays their represented type
        when their TYPE is unknown.
        """
        # --- base EV (value-aware) ---
        base_ev = self._ev_card_value(my_card, opp_cands, board_totals)

        if FOCUS_ON_CARD_PLAY:
            print(Back.WHITE + Fore.LIGHTBLACK_EX + f"for {my_card} base_ev={base_ev:.2f}" + Style.RESET_ALL)
        # --- TD shaping weights (ww_*) ---
        ww_td_now_boost     = WW_VARS.get("ww_td_now_boost") if not view.training_target else WW_VARS_TRAINING.get("ww_td_now_boost") # 0.02  # boost ~ v^2
        ww_td_later_penalty = WW_VARS.get("ww_td_later_penalty") if not view.training_target else WW_VARS_TRAINING.get("ww_td_later_penalty") # 0.05  # penalty ~ v
        ww_none_center_bias = WW_VARS.get("ww_none_center_bias") if not view.training_target else WW_VARS_TRAINING.get("ww_none_center_bias") # 0.05  # bias ~ center 
        
        v = my_card.number
        if td_intent == TDIntent.NOW:
            base_ev += td_strength * ww_td_now_boost * (v * v)
        elif td_intent == TDIntent.LATER:
            base_ev -= td_strength * ww_td_later_penalty * v
        else:
            base_ev += ww_none_center_bias * (11 - abs(6 - v))

        if FOCUS_ON_CARD_PLAY:
            print(Back.WHITE + Fore.LIGHTBLACK_EX + f"for {my_card} base_ev after TD={base_ev:.2f} ~ TD_intend={td_intent} TD_strength={td_strength}" + Style.RESET_ALL)

        # --- stubbornness belief vector (only if opponent TYPE unknown) ---
        # If we don't know opp type (because they showed number or haven't played),
        # assume they play their represented type on the board; score how our card fares vs that.
        ww_belief = WW_VARS.get("ww_belief")  if not view.training_target else WW_VARS_TRAINING.get("ww_belief")
        stub = self.traits.stubbornness / 100.0

        # detect if TYPE is unknown: opp_cands contains multiple types; if all same type -> known type
        types_in_cands = {t for (t, _v, _cnt) in opp_cands}
        if len(types_in_cands) > 1:
            opp_rep = self._opponent_rep_type(opp_name, view.board.raw_bets)
            if opp_rep is not None:
                # Build hypothetical candidate set constrained to opp_rep type
                # (respect any known number constraints in opp_cands)
                # Count remaining by number for that type using current candidates
                by_val = {}
                total_cnt = 0
                for (t,v,cnt) in opp_cands:
                    if t == opp_rep:
                        by_val[v] = by_val.get(v, 0) + cnt
                        total_cnt += cnt
                if total_cnt > 0:
                    # compute p_win, p_lose against *only* opp_rep candidates
                    wins = losses = 0
                    my_base = self._card_strength(my_card, board_totals)
                    for v, cnt in by_val.items():
                        opp_base = v * board_totals.get(opp_rep, 0)
                        dom_me   = self._multiplier(my_card.type, opp_rep)
                        dom_opp  = self._multiplier(opp_rep, my_card.type)
                        s_me  = my_base * max(dom_me, 1)
                        s_opp = opp_base * max(dom_opp, 1)
                        if s_me > s_opp: wins += cnt
                        elif s_opp > s_me: losses += cnt
                    p_win = wins / total_cnt
                    p_lose = losses / total_cnt
                    belief_score = (p_win - p_lose) * stub
                    if FOCUS_ON_CARD_PLAY:
                        print(Back.WHITE + Fore.LIGHTBLACK_EX + f"for {my_card} Stubborness addage{ww_belief * belief_score:.2f}" + Style.RESET_ALL)
                    base_ev += ww_belief * belief_score
                # else: no remaining cards of that type -> add 0
        return base_ev

    def _choose_show_side(self, my_card: Card, opp_cands, board_totals) -> bool:
        """
        Decide to show TYPE (True) or NUMBER (False) by maximizing opponent uncertainty.
        """
        def entropy_if_show(show_type: bool) -> float:
            wins = losses = total = 0
            for (t,v,cnt) in opp_cands:
                dom_me   = self._multiplier(my_card.type, t)
                dom_opp  = self._multiplier(t, my_card.type)
                s_me  = my_card.number * board_totals.get(my_card.type, 0) * max(dom_me,1)
                s_opp = v * board_totals.get(t, 0) * max(dom_opp,1)
                if s_me > s_opp: wins += cnt
                elif s_opp > s_me: losses += cnt
                total += cnt
            if total == 0:
                return 0.0
            p = max(1e-6, min(1-1e-6, wins/total))
            return -(p*math.log(p) + (1-p)*math.log(1-p))

        e_type   = entropy_if_show(True)
        e_number = entropy_if_show(False)

        if abs(e_type - e_number) < 1e-6:
            return self.rng.random() > (self.traits.bluffiness/100.0)
        return e_type >= e_number

    def choose_cards_for_battles(
        self,
        view: "PlayerView",
        battles: list["Battle"],
        memory: "DeckMemory"
    ) -> list[tuple["Battle", "Card", bool]]:
        """
        Return list of (battle_obj, chosen_card, show_type_bool) for all battles involving me.
        """
        intent, intent_strength = self._decide_td_intent(view, battles, memory)
        board_totals = view.board.total_bets_by_type

        hand = list(view.my_hand.cards)
        picks: list[tuple["Battle", "Card", bool]] = []

        # prioritize battles where opponent has already revealed something to me
        def opp_partial_info(b: "Battle") -> int:
            if b.opp_card is None:
                return 0
            else:
                return 1

        ordered = sorted(battles, key=opp_partial_info, reverse=True)

        for b in ordered:
            opp_name = b.opponent_name
            opp_type_known  = (b.opp_card is not None and b.opp_faceup == 'type')
            opp_num_known   = (b.opp_card is not None and not b.opp_faceup == 'number')
            opp_shown_type  = b.opp_card.type  if opp_type_known else None
            opp_shown_value = b.opp_card.number if opp_num_known  else None

            # opponent candidate set from memory + shown info
            opp_cands = memory.possible_given_shown(
                shown_type=opp_shown_type,
                shown_number=opp_shown_value
            )

            # score my current hand for this battle
            if FOCUS_ON_CARD_PLAY:
                print(Back.WHITE + Fore.LIGHTBLACK_EX + f"Choosing for {view.me} in battle vs {opp_name}" + Style.RESET_ALL)
            card_scores: list[tuple[float, "Card"]] = []
            for c in hand:
                sc = self._score_card_for_battle(
                    view=view,
                    my_card=c,
                    opp_name=opp_name,
                    opp_cands=opp_cands,
                    board_totals=board_totals,
                    td_intent=intent,
                    td_strength=intent_strength,
                )
                if FOCUS_ON_CARD_PLAY:
                    print(Back.WHITE + Fore.LIGHTBLACK_EX + f"final score for {c} is {sc:.2f}" + Style.RESET_ALL)
                card_scores.append((sc, c))

            if not card_scores:
                break

            # softmax select
            mx = max(s for s,_ in card_scores)
            ww_temp_min, ww_temp_max = 0.15, 2.0
            temperature = 1.2 - 0.9 * (self.traits.ev_adherence/100.0)
            temperature = max(ww_temp_min, min(ww_temp_max, temperature))
            exps = [math.exp((s - mx)/temperature) for s,_ in card_scores]
            Z = sum(exps) or 1.0
            r = self.rng.random()
            acc = 0.0
            chosen = card_scores[0][1]
            for w, (_s, c) in zip(exps, card_scores):
                acc += w / Z
                if r <= acc:
                    chosen = c
                    break

            # decide show side
            show_type = self._choose_show_side(chosen, opp_cands, board_totals)

            # record and consume from temporary hand
            picks.append((b, chosen, show_type))
            hand.remove(chosen)

        return picks
    
    def _choose_preliminary_bet_for_battle(
        self,
        view: "PlayerView",
        battle_view,                    # has .opponent_name, .opp_card, .opp_faceup, .battle_id
        my_card: Card,
        board_totals: Dict[GladiatorType, int],
        memory: "DeckMemory",
        td_intent: TDIntent,
        td_strength: float,
    ) -> int:
        """
        Score a in [0..behind] with:
          Score(a) = TD-pressure(a)  - Liquidity(a)  + EV(a)  + AggBoost(a)

        EV here is (p_win - p_lose) * a (opponent concede ⇒ no coins from prelims).
        Liquidity reserve uses *future prelims*:
          future_prelims = 3 * (NUMBER_OF_BATTLES * rounds_left_after_current
                                + repetitions_left_in_current_round)
        """
        rng = self.rng
        traits = self.traits

        behind = view.my_bankroll.behind
        if behind <= 0:
            return 0

        # ---------- Opp identity & candidates ----------
        opp_name = battle_view.opponent_name
        opp_type_known  = (battle_view.opp_card is not None and battle_view.opp_faceup == 'type')
        opp_num_known   = (battle_view.opp_card is not None and battle_view.opp_faceup == 'number')
        opp_shown_type  = battle_view.opp_card.type   if opp_type_known else None
        opp_shown_value = battle_view.opp_card.number if opp_num_known  else None

        opp_cands = memory.possible_given_shown(
            shown_type=opp_shown_type,
            shown_number=opp_shown_value
        )

        # ---------- p_win / p_lose for EV ----------
        wins, losses, total, _ = self._win_lose_stats(my_card, opp_cands, board_totals)
        p_win  = (wins   / total) if total > 0 else 0.5
        p_lose = (losses / total) if total > 0 else 0.5

        # ---------- rounds/repetitions left for reserve ----------
        try:
            from ..globals import NUM_PLAYERS, NUMBER_OF_BATTLES
            total_rounds = NUM_PLAYERS
            reps_per_round = NUMBER_OF_BATTLES
        except Exception:
            total_rounds = 4
            reps_per_round = 3

        # round_number is 1-based per your view
        cur_round = max(1, int(view.context.round_number))
        rounds_left_after = max(0, total_rounds - cur_round)

        rep_idx = view.context.battle_phase_index
        if rep_idx is None:
            reps_left_current = reps_per_round  # if not yet started, all repetitions are left
        else:
            # repetitions left *after this repetition*
            reps_left_current = max(0, reps_per_round - (rep_idx + 1))

        future_prelims = 3 * (reps_per_round * rounds_left_after + reps_left_current)

        # ---------- Weights ----------
        # TD pressure (penalize a==0 if going for TD)
        ww_td_zero_now    = WW_VARS.get("ww_td_zero_now")  if not view.training_target else WW_VARS_TRAINING.get("ww_td_zero_now")  # 1.0
        ww_td_zero_later  = WW_VARS.get("ww_td_zero_later")  if not view.training_target else WW_VARS_TRAINING.get("ww_td_zero_later")  # 0.5
        # Liquidity / reserve
        ww_liq_linear       = WW_VARS.get("ww_liq_linear") if not view.training_target else WW_VARS_TRAINING.get("ww_liq_linear")  # 0.3
        ww_liq_later_mult   = WW_VARS.get("ww_liq_later_mult")  if not view.training_target else WW_VARS_TRAINING.get("ww_liq_later_mult")  # 1.5
        ww_reserve_per_pre  = WW_VARS.get("ww_reserve_per_pre") if not view.training_target else WW_VARS_TRAINING.get("ww_reserve_per_pre")  # 1.0
        ww_short_mult_base  = WW_VARS.get("ww_short_mult_base")  if not view.training_target else WW_VARS_TRAINING.get("ww_short_mult_base")  # 0.6   
        # EV
        ww_ev_prelim               = WW_VARS.get("ww_ev_prelim")  if not view.training_target else WW_VARS_TRAINING.get("ww_ev_prelim")  # 0.5
        # Aggressiveness (concave)
        ww_aggr_prelim             = WW_VARS.get("ww_aggr_prelim")  if not view.training_target else WW_VARS_TRAINING.get("ww_aggr_prelim")  # 0.3

        liq_mult = ww_liq_later_mult if td_intent == TDIntent.LATER else 1.0
        reserve_target = ww_reserve_per_pre * future_prelims

        scores: Dict[int, float] = {}
        for a in range(0, behind + 1):
            s = 0.0

            # --- 1) TD pressure against a==0 ---
            if a == 0:
                if td_intent == TDIntent.NOW:
                    s -= ww_td_zero_now * td_strength
                elif td_intent == TDIntent.LATER:
                    s -= ww_td_zero_later * max(0.25, td_strength * 0.8)
                if FOCUS_ON_BATTLE_INITIAL_BET:
                    print(Back.CYAN + Fore.LIGHTYELLOW_EX + f"{view.me} a={a}: score after TD preasure {s}" + Style.RESET_ALL)
            

            # --- 2) Liquidity: linear spend + reserve shortfall ---
            liq_pen = ww_liq_linear * a * liq_mult
            remaining_after = max(0, behind - a)
            short = max(0.0, reserve_target - remaining_after)
            ww_short_mult = WW_VARS.get("ww_short_mult")  if not view.training_target else WW_VARS_TRAINING.get("ww_short_mult")  # 0.4
            liq_pen += (ww_short_mult_base + (ww_short_mult * (traits.ev_adherence / 100.0))) * short
            s -= liq_pen
            if FOCUS_ON_BATTLE_INITIAL_BET:
                    print(Back.CYAN + Fore.LIGHTYELLOW_EX + f"{view.me} a={a}: liquidity {liq_pen}" + Style.RESET_ALL)
            # --- 3) EV: (p_win - p_lose) * a ---
            s += ww_ev_prelim * ((p_win - p_lose) * a)
            if FOCUS_ON_BATTLE_INITIAL_BET:
                    print(Back.CYAN + Fore.LIGHTYELLOW_EX + f"{view.me} a={a}: EV {((p_win - p_lose) * a):.2f}" + Style.RESET_ALL)

            # --- 4) Aggressiveness (concave in a) ---
            s += ww_aggr_prelim * (traits.aggressiveness / 100.0) * math.sqrt(a)

            scores[a] = s
            if FOCUS_ON_BATTLE_INITIAL_BET:
                print(Back.CYAN + Fore.LIGHTYELLOW_EX + f"{view.me} a={a}: final score {s:.2f}" + Style.RESET_ALL)

        # Exploration: small chance
        if rng.random() < (traits.exploration / 100.0):
            return rng.randint(0, behind)

        # Softmax
        mx = max(scores.values()) if scores else 0.0
        temperature = 1.5 - 1.3 * (traits.ev_adherence / 100.0)
        temperature = max(0.15, min(2.5, temperature))
        exps = {a: math.exp((v - mx) / temperature) for a, v in scores.items()}
        Z = sum(exps.values()) or 1.0
        r = rng.random()
        acc = 0.0
        for a, w in exps.items():
            acc += w / Z
            if r <= acc:
                return int(a)
        return 0

   
    # Phase 4 – subphase 1 (playing cards + preliminary bets)
    def assign_cards_and_bets(
        self, view: "PlayerView"
    ) -> List[Tuple[int, "Card", bool, int]]:
        """
        For each of my battles, return (battle_id, card, show_type_bool, bet_amount).
        Uses real Battle objects and self.brain_memory.
        """
        # 1) my battles from the view (expected: List[Battle])
        battles = list(getattr(view, "battles", []))

        # 2) ensure deck memory exists and knows my hand
        if not hasattr(self, "brain_memory") or self.brain_memory is None:
            if GAME_ENGINE_PIRINTS:
                print(Back.RED + Fore.LIGHTMAGENTA_EX + "Brain memory NOT found! Initializing brain memory" + Style.RESET_ALL)
            self.brain_memory = DeckMemory()
            self.brain_memory.remove_cards(view.my_hand.cards)

        # 3) choose (battle, card, show_type) via the brain
        picks = self.choose_cards_for_battles(view, battles, self.brain_memory)
        td_intent, td_strength = self._decide_td_intent(view, battles, self.brain_memory)
        board_totals = view.board.total_bets_by_type


        # 4) map to (battle_id, card, show_type, prelim_bet)
        results: List[Tuple[int, "Card", bool, int]] = []
        budget = view.my_bankroll.behind

        for (battle, card, show_type) in picks:
            battle_id = getattr(battle, "battle_id", None)  
            prelim_bet = self._choose_preliminary_bet_for_battle(
                view=view,
                battle_view=battle,
                my_card=card,
                board_totals=board_totals,
                memory=self.brain_memory,
                td_intent=td_intent,
                td_strength=td_strength,
            )
            prelim_bet = max(0, min(budget, int(prelim_bet)))
            budget -= prelim_bet
            results.append((battle_id, card, show_type, prelim_bet))

        return results

    # Phase 4 – subphase 2 (additional betting, before equalization)
    def additional_battle_bets(self, view: "PlayerView") -> Dict[int, int]:
        """
        Choose up to 3 incremental additional bets across my battles in this subphase.
        Joint choice: (battle, a) via softmax over scores(battle, a).
        Returns {battle_id: delta_amount_to_add}.
        """
        rng = self.rng
        traits = self.traits

        behind = view.my_bankroll.behind
        if behind <= 0:
            return {getattr(b, "battle_id", i): 0 for i, b in enumerate(view.battles)}

        # Weights
        ww_ev_card_bet            = WW_VARS.get("ww_ev_card_bet")  if not view.training_target else WW_VARS_TRAINING.get("ww_ev_card_bet")  # 0.4  # EV per card bet
        ww_concede_mask  = WW_VARS.get("ww_concede_mask")   if not view.training_target else WW_VARS_TRAINING.get("ww_concede_mask")  # 0.3  # boost to flip dominance gap
        ww_bluff_hint    = WW_VARS.get("ww_bluff_hint")  if not view.training_target else WW_VARS_TRAINING.get("ww_bluff_hint")  # 0.2  # bluff bonus if not showing type
        ww_aggr          = WW_VARS.get("ww_aggr")  if not view.training_target else WW_VARS_TRAINING.get("ww_aggr")  # 0.2  # aggressiveness bonus (concave)
        ww_liq_linear    = WW_VARS.get("ww_liq_linear")  if not view.training_target else WW_VARS_TRAINING.get("ww_liq_linear")  # 0.3  # linear liquidity cost
        ww_liq_short     = WW_VARS.get("ww_liq_short")  if not view.training_target else WW_VARS_TRAINING.get("ww_liq_short")  # 0.5  # liquidity shortfall cost

        # Reserve logic: prefer keeping some coins for future prelims (same formula you set before)
        try:
            from ..globals import NUM_PLAYERS, NUMBER_OF_BATTLES
            total_rounds = NUM_PLAYERS
            reps_per_round = NUMBER_OF_BATTLES
        except Exception:
            total_rounds = 4
            reps_per_round = 3

        cur_round = max(1, int(view.context.round_number))
        rounds_left_after = max(0, total_rounds - cur_round)
        rep_idx = view.context.battle_phase_index
        if rep_idx is None:
            reps_left_current = reps_per_round
        else:
            reps_left_current = max(0, reps_per_round - (rep_idx + 1))
        future_prelims = 3 * (reps_per_round * rounds_left_after + reps_left_current)
        reserve_target = 0.25 * future_prelims  # gentler than prelim stage, we’re already in subphase 2

        # Board totals (multipliers) for EV calc
        board_totals = view.board.total_bets_by_type

        # Track cumulative additions per battle within this call
        added: Dict[int, int] = {getattr(b, "battle_id", i): 0 for i, b in enumerate(view.battles)}

                # ---------- precompute per-battle constants (independent of 'a') ----------
        battle_info = []  # list of dicts aligned with view.battles indices
        for i, bview in enumerate(view.battles):
            info = {
                "i": i,
                "bid": getattr(bview, "battle_id", None),
                "valid": True,
                "my_bet0": getattr(bview, "my_bet", 0),
                "opp_bet0": getattr(bview, "opp_bet", 0),
                "bluff_bonus": 0.0,
                "m_ev": 0.0,     # slope for EV wrt 'a' (ww_ev * (p_win - p_lose))
                "ev_const": 0.0, # constant part of EV wrt 'a' (ww_ev * (p_win - p_lose) * my_bet0)
                "base_flip": 0,  # how much 'a' is needed to flip/extend lead for concede calc
                "dom_gap": 0,    # dominance gap vs opp's representation
            }

            my_card  = getattr(bview, "my_card", None)
            opp_card = getattr(bview, "opp_card", None)
            if my_card is None or info["bid"] is None:
                info["valid"] = False
                battle_info.append(info)
                continue
            
            # partial info
            opp_type_known  = (opp_card is not None and getattr(bview, "opp_faceup", None) == 'type')
            opp_num_known   = (opp_card is not None and getattr(bview, "opp_faceup", None) == 'number')
            opp_shown_type  = opp_card.type   if opp_type_known else None
            opp_shown_value = opp_card.number if opp_num_known  else None

            # opponent candidate set from memory
            opp_cands = self.brain_memory.possible_given_shown(
                shown_type=opp_shown_type,
                shown_number=opp_shown_value
            )

            # p_win / p_lose for *this card*
            wins, losses, total, _ = self._win_lose_stats(my_card, opp_cands, board_totals)
            p_win  = (wins   / total) if total > 0 else 0.5
            p_lose = (losses / total) if total > 0 else 0.5

            # EV parts that don't depend on 'a'
            m_ev = ww_ev_card_bet * (p_win - p_lose)                  # slope wrt 'a'
            ev_const = m_ev * info["my_bet0"]                # constant part
            info["m_ev"] = m_ev
            info["ev_const"] = ev_const

            # concede proxy: dominance vs opponent's represented type
            opp_name = getattr(bview, "opponent_name", "")
            opp_rep  = self._opponent_rep_type(opp_name, view.board.raw_bets)
            dom_gap  = 0
            if opp_rep is not None:
                dom_gap = self._multiplier(my_card.type, opp_rep) - self._multiplier(opp_rep, my_card.type)
            info["dom_gap"] = dom_gap

            # bluff bonus (constant wrt 'a')
            show_face = getattr(bview, "my_faceup", None)
            am_showing_type = (show_face == 'type')
            bluff_bonus = 0.0
            if opp_rep is not None and self._multiplier(my_card.type, opp_rep) > 1 and not am_showing_type:
                bluff_bonus = ww_bluff_hint * (self.traits.bluffiness / 100.0)
            info["bluff_bonus"] = bluff_bonus

            # how much add is needed to flip/extend lead (for concede pressure)
            my_bet0  = info["my_bet0"]
            opp_bet0 = info["opp_bet0"]
            info["base_flip"] = max(0, opp_bet0 - my_bet0)

            battle_info.append(info)

        # only consider valid battles
        battles_remaining = [bi for bi, inf in enumerate(battle_info) if inf["valid"]]
        results = {inf["bid"]: 0 for inf in battle_info if inf["valid"]}
        current_budget = behind

        # we allow up to 3 picks, but not more than we have battles or budget
        max_steps = min(3, current_budget, len(battles_remaining))

        def score_a_for_battle(inf: dict, a: int, cur_budget: int) -> float:
            """
            Score for adding 'a' coins to this battle, using only a-dependent terms:
              ev(a) = (1 - ww_concede_mask * p_concede(a)) * (ev_const + m_ev * a)
              liq(a) = ww_liq_linear * a + (ww_liq_short + 0.4*ev_adherence) * short(cur_budget, a)
              aggr(a) = ww_aggr * sqrt(a) * aggressiveness
              + bluff_bonus (a-constant)
            """
            if a < 0 or a > cur_budget:
                return -1e9

            # concede prob from 'a'
            k_c = 0.35
            base_flip = inf["base_flip"]
            dom_gap   = inf["dom_gap"]
            # larger 'a' beyond base_flip => higher concede prob
            p_concede = 1.0 / (1.0 + math.exp(-k_c * ( (a - base_flip) + 0.75 * dom_gap )))

            # EV with masking by concede
            ev = (1.0 - ww_concede_mask * p_concede) * (inf["ev_const"] + inf["m_ev"] * a)

            # liquidity & reserve (depends on current budget at this pick)
            liq = ww_liq_linear * a
            remaining_after = max(0, cur_budget - a)
            short = max(0.0, reserve_target - remaining_after)
            liq += (ww_liq_short + 0.4 * (self.traits.ev_adherence / 100.0)) * short

            # aggressiveness (concave)
            aggr = ww_aggr * (self.traits.aggressiveness / 100.0) * math.sqrt(a)
            if FOCUS_ON_ADDITIONAL_BETS:
                print(Back.MAGENTA + Fore.LIGHTYELLOW_EX + f"Score for adding a={a} to battle {inf['bid']}: ev={ev:.2f} liq={liq:.2f} aggr={aggr:.2f} bluff={inf['bluff_bonus']:.2f}" + Style.RESET_ALL)

            return ev + inf["bluff_bonus"] + aggr - liq

        # ---------- greedy 3 picks with softmax; each pick removes that battle ----------
        for _step in range(max_steps):
            if current_budget <= 0 or not battles_remaining:
                break
            
            # all a in 0..current_budget
            a_candidates = range(0, current_budget + 1)

            # build (score, battle_idx, a) table
            table: List[Tuple[float, int, int]] = []
            for bi in battles_remaining:
                inf = battle_info[bi]
                for a in a_candidates:
                    sc = score_a_for_battle(inf, a, current_budget)
                    table.append((sc, bi, a))

            if not table:
                break
            
            # softmax over (battle, a) pairs
            mx = max(s for s, _, _ in table)
            temperature = 1.2 - 0.9 * (self.traits.ev_adherence / 100.0)
            temperature = max(0.15, min(2.0, temperature))
            weights = [math.exp((s - mx) / temperature) for (s, _, _) in table]
            Z = sum(weights) or 1.0
            r = rng.random()
            acc = 0.0
            chosen_idx = 0
            for k, w in enumerate(weights):
                acc += w / Z
                if r <= acc:
                    chosen_idx = k
                    break
                
            sc, bi, a = table[chosen_idx]
            inf = battle_info[bi]
            bid = inf["bid"]

            # apply pick
            results[bid] += a
            added[bid] += a
            current_budget -= a

            # remove this battle so we never pick it again
            battles_remaining.remove(bi)


        # Any remaining battles not touched already default to +0
        return results

    def decide_equalize_or_concede(
        self,
        view: "PlayerView",
        battle_view,
        deficit_to_match: int,
    ) -> Tuple[bool, int]:
        """
        Decide whether to FIGHT (match/push what we can) or CONCEDE in equalization.
    
        Returns:
            (fight: bool, amount_to_commit: int)
    
        Rules handled:
          - If fight=True and behind < deficit, we commit what we have (engine will let battle proceed unmatched).
          - If behind == 0 and fight=True, engine should apply the “bank stakes 1” rule (we signal 0 here).
          - If fight=False, commit 0 (engine concedes and opponent wins pot + diff from bank).
        """
        rng = self.rng
        traits = self.traits
    
        behind = view.my_bankroll.behind
        if deficit_to_match <= 0:
            return (True, 0)  # nothing to do
    
        # Pull my/opp cards, partial info and compute EV for continuing
        my_card  = getattr(battle_view, "my_card", None)
        opp_card = getattr(battle_view, "opp_card", None)
    
        if my_card is None:
            # If we haven't even placed a card (shouldn't happen here), be conservative: concede if huge deficit
            return (deficit_to_match <= 1, min(behind, deficit_to_match))
    
        board_totals = view.board.total_bets_by_type
    
        opp_type_known  = (opp_card is not None and getattr(battle_view, "opp_faceup", None) == 'type')
        opp_num_known   = (opp_card is not None and getattr(battle_view, "opp_faceup", None) == 'number')
        opp_shown_type  = opp_card.type   if opp_type_known else None
        opp_shown_value = opp_card.number if opp_num_known  else None
    
        opp_cands = self.brain_memory.possible_given_shown(
            shown_type=opp_shown_type,
            shown_number=opp_shown_value
        )
    
        wins, losses, total, _ = self._win_lose_stats(my_card, opp_cands, board_totals)
        p_win  = (wins   / total) if total > 0 else 0.5
        p_lose = (losses / total) if total > 0 else 0.5
    
        # Stubbornness: resist folding when the opponent is likely “bullying”
        # Detect bluff pressure: if our TYPE > opp_rep by dominance and we are NOT showing TYPE,
        # we’re slightly more willing to fight through a deficit.
        opp_name = getattr(battle_view, "opponent_name", "")
        opp_rep  = self._opponent_rep_type(opp_name, view.board.raw_bets)
        show_face = getattr(battle_view, "my_faceup", None)
        am_showing_type = (show_face == 'type')
        dom_gap = 0
        if opp_rep is not None:
            dom_gap = self._multiplier(my_card.type, opp_rep) - self._multiplier(opp_rep, my_card.type)
    
        ww_ev              = WW_VARS.get("ww_ev")  if not view.training_target else WW_VARS_TRAINING.get("ww_ev")  # 0.5  # EV weight
        ww_deficit_pen     = WW_VARS.get("ww_deficit_pen")   if not view.training_target else WW_VARS_TRAINING.get("ww_deficit_pen")  # 0.6  # deficit penalty weight
        ww_stubborn_boost  = WW_VARS.get("ww_stubborn_boost")   if not view.training_target else WW_VARS_TRAINING.get("ww_stubborn_boost")  # 0.2  # stubbornness boost weight
        ww_liq_call_cost   = WW_VARS.get("ww_liq_call_cost")   if not view.training_target else WW_VARS_TRAINING.get("ww_liq_call_cost")  # 0.3  # liquidity cost per coin called
        ww_bank_bias       = WW_VARS.get("ww_bank_bias")   if not view.training_target else WW_VARS_TRAINING.get("ww_bank_bias")  # 0.1  # small bias to keep fighting if broke
    
        # Base call EV approx: (p_win - p_lose) * (my_total_at_risk_after_call)
        my_bet  = getattr(battle_view, "my_bet", 0)
        opp_bet = getattr(battle_view, "opp_bet", 0)
        call_amount = min(behind, deficit_to_match)
        risk_after_call = my_bet + call_amount
    
        ev_call = ww_ev * ((p_win - p_lose) * risk_after_call)
    
        # Deficit pain + liquidity pain
        call_cost = ww_deficit_pen * deficit_to_match + ww_liq_call_cost * call_amount
    
        # Stubbornness: reduce effective cost when (a) we’re type-superior to opp_rep and (b) we’re obscuring our type
        stubborn = (traits.stubbornness / 100.0)
        stubborn_offset = 0.0
        if dom_gap > 0 and not am_showing_type:
            stubborn_offset = ww_stubborn_boost * stubborn * dom_gap
    
        # Net score for fighting
        fight_score = ev_call - call_cost + stubborn_offset
        if FOCUS_ON_EQUALIZING_BETS:
            print(Back.GREEN + Fore.LIGHTYELLOW_EX + f"{view.me} Equalize decision vs {opp_name}: ev_call={ev_call:.2f} call_cost={call_cost:.2f} stubborn_offset={stubborn_offset:.2f} final score: {ev_call - call_cost + stubborn_offset:.2f}" + Style.RESET_ALL)
    
        # A tiny bias to keep fighting if we’re broke and bank can stake 1
        if behind == 0:
            fight_score += ww_bank_bias
    
        # Decide
        if fight_score > 0:
            return (True, call_amount)   # fight (match if can; else push what we have)
        else:
            return (False, 0)            # concede









# -------- Minimal random baseline brain --------
class RandomBrain(PlayerBrain):
    """Keeps your current behavior but routed through the interface for future upgrades."""

    def pick_favored_faction(self, view: PlayerView) -> str:
        others = [n for n in view.players if n != view.me]
        return self.rng.choice(others)

    def pick_representation_bet(self, view: PlayerView) -> Tuple[GladiatorType, int]:
        gtype = self.rng.choice(list(GladiatorType))
        amount = 1  # mimic current logic
        return gtype, amount

    def assign_cards_and_bets(
        self, view: PlayerView
    ) -> List[Tuple[int, Card, bool, int]]:
        actions: List[Tuple[int, Card, bool, int]] = []
        hand = list(view.my_hand.cards)
        # assign one card per battle I’m involved in, if I have enough
        for b in view.battles:
            if not hand:
                break
            card = hand.pop()  # same as your stack-pop style, arbitrary
            show_type = bool(self.rng.getrandbits(1))
            preliminary_bet = 1 if view.my_bankroll.behind > 0 else 0
            actions.append((b.battle_id, card, show_type, preliminary_bet))
        return actions

    def additional_battle_bets(self, view: PlayerView) -> Dict[int, int]:
        out: Dict[int, int] = {}
        # add +1 to each active battle if we have coins (like current logic)
        budget = view.my_bankroll.behind
        for b in view.battles:
            add = 1 if budget > 0 else 0
            out[b.battle_id] = add
            budget -= add
        return out
