from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, random
from ..models import Card, GladiatorType, Battle
from ..observations import PlayerView
from colorama import Fore, Back, Style
from .utils import estimate_future_representation_open_lane
from ..globals import ADDITIONAL_INFO, TARGET_PLAYER, NUMBER_OF_BATTLES, NUM_PLAYERS, FOCUS_ON_BET_SIZING, FOCUS_ON_CARD_PLAY
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
        print(Back.YELLOW + Fore.BLACK + f"Removing from memory: {cards}" + Style.RESET_ALL)
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
        #print(Back.YELLOW + Fore.BLACK + f"possible cards given shown {shown_type} {shown_number}: {out}" + Style.RESET_ALL)
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
    def pick_favored_faction(self, view: PlayerView) -> str:
        favored = ""
        max_coins = -1
        for name in view.players:
            if name != view.me:
                coins = view.others_bankrolls[name].behind + view.others_bankrolls[name].front
                if coins > max_coins:
                    max_coins = coins
                    favored = name
        return favored
                

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
        hand_support: Dict[GladiatorType, float] = {}
        for t in GladiatorType:
            hand_support[t] = 1.0 * hand_counts[t] + 0.3 * hand_strength[t] # hard coded weights

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
        ww_hand   = 1.0
        ww_open = 0.8 * (1.0 - traits.herding / 100.0)
        ww_countp = 0.9                                        # base for counter_power
        # Tilt countering more for higher risk_tolerance (seek edges vs others)
        ww_counter = ww_countp * (0.5 + 0.5 * (traits.risk_tolerance / 100.0))
        ww_bluff = 1.0 * (traits.bluffiness / 100.0)

        scores: Dict[GladiatorType, float] = {}
        for t in GladiatorType:
            s = 0.0
            s += ww_hand   * hand_support[t]
            s += ww_counter * counter_power[t]
            s += ww_open * open_lane_bonus[t]
            s -= ww_bluff  * deception_penalty[t]
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
            ww_round = 0.96                              # small weight; tune if needed
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
            ww_battle = 0.8  # reduce impact so it doesn’t swamp costs
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
                ww_dom = 0.50  # capped impact
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
            ww_liquidty = 0.8
            s -= (lam * a + reserve_penalty) * ww_liquidty
            if FOCUS_ON_BET_SIZING:
                print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: liq={lam*a:.2f} reserve_penalty={reserve_penalty:.2f}" + Style.RESET_ALL)

            # ========= 5) Aggressiveness: concave boost (optional, small) =========
            theta = 0.10  # smaller; concave in 'a'
            s += theta * (self.traits.aggressiveness / 100.0) * math.sqrt(a)
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
        ww_td_norm_now   = 2.0
        ww_td_drive      = 0.8
        ww_td_tempo      = 0.6
        ww_td_norm_later = 1.2
        ww_td_drive_lat  = 0.5
        ww_td_tempo_lat  = 0.8
        ww_td_none_k     = 0.6  # how much (1-norm) favors NONE

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
        ww_td_now_boost     = 0.02   # boost ~ v^2 when NOW
        ww_td_later_penalty = 0.50   # penalty ~ v when LATER
        ww_none_center_bias = 0.05   # light center-ish bias for NONE

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
        ww_belief = 1.0  # weight for belief vector contribution
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
            print(Back.RED + Fore.LIGHTMAGENTA_EX + "Brain memory NOT found! Initializing brain memory" + Style.RESET_ALL)
            self.brain_memory = DeckMemory()
            self.brain_memory.remove_cards(view.my_hand.cards)

        # 3) choose (battle, card, show_type) via the brain
        picks = self.choose_cards_for_battles(view, battles, self.brain_memory)

        # 4) map to (battle_id, card, show_type, prelim_bet)
        results: List[Tuple[int, "Card", bool, int]] = []
        for (battle, card, show_type) in picks:
            battle_id = id(battle)   # stable within run; engine can map id->Battle
            prelim_bet = 1           # simple preliminary bet; tune later
            results.append((battle_id, card, show_type, prelim_bet))

        return results

    # Phase 4 – subphase 2 (additional betting, before equalization)
    def additional_battle_bets(
        self, view: PlayerView
    ) -> Dict[int, int]:
        """
        Return a mapping battle_id -> delta_bet you want to add (can be 0).
        Negative not allowed (no un-betting). Concede is handled by equalization phase
        in your engine; a brain can “signal” concede by returning 0 then failing equalization.
        """
        raise NotImplementedError









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
