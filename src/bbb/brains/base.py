from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, random
from ..models import Card, GladiatorType
from ..observations import PlayerView
from colorama import Fore, Back, Style
from .utils import estimate_future_representation_open_lane
from ..globals import ADDITIONAL_INFO, TARGET_PLAYER, NUMBER_OF_BATTLES, NUM_PLAYERS

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
    exploration: int = 10          # random exploration

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


# -------- Brain interface --------
class PlayerBrain:
    """
    Pluggable decision-maker. All methods are *pure* in the sense they return
    intents; the engine applies side effects.
    """

    def __init__(self, rng: Optional[random.Random] = None, traits: Optional[Traits] = None):
        self.rng = rng or random.Random()
        self.traits = traits or Traits()

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

            print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: liq={lam*a:.2f} reserve_penalty={reserve_penalty:.2f}" + Style.RESET_ALL)

            # ========= 5) Aggressiveness: concave boost (optional, small) =========
            theta = 0.10  # smaller; concave in 'a'
            s += theta * (self.traits.aggressiveness / 100.0) * math.sqrt(a)

            print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"{view.me} a={a}: final score={s:.2f}" + Style.RESET_ALL)

            scores[a] = s

        # Exploration?
        #print(Back.MAGENTA + Fore.LIGHTGREEN_EX + f"Scores for {view.me} with type {chosen_type}: {scores}" + Style.RESET_ALL)
        if self.rng.random() < (self.traits.exploration / 100.0):
            return self.rng.randint(1, behind)

        temperature = 1.5 - 1.3 * (self.traits.ev_adherence / 100.0)
        temperature = max(0.15, min(2.5, temperature))
        a_choice = _softmax_pick(scores, self.rng, temperature)
        return max(1, min(behind, int(a_choice)))


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # Phase 4 – subphase 1 (playing cards + preliminary bets)
    def assign_cards_and_bets(
        self, view: PlayerView
    ) -> List[Tuple[int, Card, bool, int]]:
        """
        For each of my battles, return a tuple:
        (battle_id, card, show_type_bool, bet_amount)
        Order doesn’t matter; engine will apply respecting turn order.
        """
        raise NotImplementedError

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
