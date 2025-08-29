from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from .models import Player, Battle, Card, GladiatorType
from .board import BettingBoard

@dataclass(frozen=True)
class BoardView:
    total_bets_by_type: Dict[GladiatorType, int]
    raw_bets: Dict[GladiatorType, List[Tuple[str, int]]]  # (player_name, amount)

@dataclass(frozen=True)
class BattleView:
    battle_id: int
    opponent_name: str
    my_card: Optional[Card]        # None if I haven't played yet
    opp_card: Optional[Card]       # None if they haven't played yet
    my_faceup: Optional[str]      # "type" or "number" or None (what I'm showing)
    opp_faceup: Optional[str]     # "type" or "number" or None (what they are showing)
    my_bet: int
    opp_bet: int
    winner: Optional[str]         # name or None (for already-resolved battles in logs/history)

@dataclass(frozen=True)
class PlayerBankrollView:
    behind: int   # coins behind screen
    front: int    # coins in front this round

@dataclass(frozen=True)
class PlayerHandView:
    # Full hand is private to the player; exposing it here to the player's brain is OK
    cards: List[Card]

@dataclass(frozen=True)
class RoundContext:
    round_number: int
    battle_phase_index: Optional[int]  # which repetition of phase 4 we're in (0..num_battles-1) or None outside

@dataclass(frozen=True)
class PlayerView:
    """What the player 'knows' at decision time (public info + their private hand)."""
    me: str
    players: List[str]                       # seating in order
    board: BoardView
    my_hand: PlayerHandView
    my_bankroll: PlayerBankrollView
    others_bankrolls: Dict[str, PlayerBankrollView]  # public-ish view: we expose both for now since your engine prints it
    battles: List[BattleView]               # current round’s battles involving me
    favored_faction_choice: Optional[str]   # who I picked (None before phase 1)
    context: RoundContext

def _faceup_of(player_is_p1: bool, battle: Battle) -> Optional[str]:
    if player_is_p1:
        if battle.card1 is None:
            return None
        return "type" if battle.card1_shows_type else "number"
    else:
        if battle.card2 is None:
            return None
        return "type" if battle.card2_shows_type else "number"

def _battle_view_for(me: Player, battle_id: int, battle: Battle) -> BattleView:
    if battle.player1 == me:
        opp = battle.player2
        my_bet, opp_bet = battle.bet1, battle.bet2
        my_faceup = _faceup_of(True, battle)
        my_card = battle.card1
        opp_card = battle.card2
        opp_faceup = _faceup_of(False, battle)
    elif battle.player2 == me:
        opp = battle.player1
        my_bet, opp_bet = battle.bet2, battle.bet1
        my_card = battle.card2
        opp_card = battle.card1
        my_faceup = _faceup_of(False, battle)
        opp_faceup = _faceup_of(True, battle)
    else:
        # Not my battle; still return a public view
        opp = battle.player2
        my_bet = opp_bet = 0
        my_faceup = opp_faceup = None

    winner_name = battle.winner.name if battle.winner else None
    return BattleView(
        battle_id=battle_id,
        opponent_name=opp.name,
        my_faceup=my_faceup,
        my_card=my_card,
        opp_card=opp_card,
        opp_faceup=opp_faceup,
        my_bet=my_bet,
        opp_bet=opp_bet,
        winner=winner_name,
    )

def build_player_view(
    me: Player,
    players: List[Player],
    board: BettingBoard,
    battles: List[Battle],
    round_number: int,
    battle_phase_index: Optional[int],
) -> PlayerView:
    # board view
    totals = board.get_total_bets()
    raw_bets: Dict[GladiatorType, List[tuple[str, int]]] = {}
    for g, entries in board.bets.items():
        raw_bets[g] = [(p.name, amt) for (p, amt) in entries]

    board_view = BoardView(
        total_bets_by_type=totals,
        raw_bets=raw_bets,
    )

    # bankrolls
    my_bankroll = PlayerBankrollView(behind=me.coins, front=me.front_coins)
    others_bankrolls = {
        p.name: PlayerBankrollView(behind=p.coins, front=p.front_coins) for p in players if p is not me
    }

    # battle views (my battles + public list)
    my_battles = []
    for battle_id, b in enumerate(battles):
        v = _battle_view_for(me, battle_id, b)
        my_battles.append(v)

    # hand
    my_hand = PlayerHandView(cards=list(me.cards))

    # favored faction
    favored = me.favored_faction  # string or None

    return PlayerView(
        me=me.name,
        players=[p.name for p in players],
        board=board_view,
        my_hand=my_hand,
        my_bankroll=my_bankroll,
        others_bankrolls=others_bankrolls,
        battles=my_battles,
        favored_faction_choice=favored,
        context=RoundContext(round_number=round_number, battle_phase_index=battle_phase_index),
    )
