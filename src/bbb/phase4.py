from .decisions import player_assign_cards_and_bets, player_additional_battle_bets
from .combat import resolve_battles, correct_bets, give_total_domination
from .brains.base import PlayerBrain
from colorama import Fore, Back, Style

def equalize_all_battles(battles, players, board, round_number, battle_phase_index):
    """
    For each battle needing equalization, ask the short bettor's brain whether to
    fight (match if possible; otherwise add what they can / bank=1 if 0) or concede.
    """
    # Precompute a map: battle object -> its global index in this round's battles

    for battle in battles:
        if battle.winner is not None:
            continue  # already resolved by concession earlier

        bet1, bet2 = battle.bet1, battle.bet2
        if bet1 == bet2:
            continue  # already equal

        # Determine who must act
        if bet1 > bet2:
            acting = battle.player2
            other  = battle.player1
            deficit = bet1 - bet2
        else:
            acting = battle.player1
            other  = battle.player2
            deficit = bet2 - bet1

        # Build the acting player's view
        from .observations import build_player_view  # adjust import path to your project layout
        view = build_player_view(
            acting,
            players,
            board,
            acting.battles,           # their own battles list
            round_number,
            battle_phase_index,
        )
        # Find the correct battle_view inside this player's view by global id
        my_bview = None
        for bv in view.battles:
            if (battle.player1 == acting and bv.opponent_name == battle.player2.name) or \
               (battle.player2 == acting and bv.opponent_name == battle.player1.name):
                my_bview = bv
                break

        if my_bview is None or not hasattr(acting, "brain") or acting.brain is None:
            # Fallback: if no brain or we failed to locate the view, default to old behavior:
            print(Back.RED + Fore.WHITE +  f"defaulting to common behaviour" + Style.RESET_ALL)
            if acting.coins >= deficit:
                # match
                spend = deficit
                acting.coins -= spend
                acting.front_coins += spend
                if acting is battle.player2:
                    battle.bet2 += spend
                else:
                    battle.bet1 += spend
            else:
                # concede
                print(f"{acting.name} concedes!")
                if acting is battle.player2:
                    battle.player1.cards_won.append(battle.card2)
                    battle.winner = battle.player1
                else:
                    battle.player2.cards_won.append(battle.card1)
                    battle.winner = battle.player2
            continue

        # Ask the brain
        try:
            fight, suggested_add = acting.brain.decide_equalize_or_concede(
                view=view,
                battle_view=my_bview,
                deficit_to_match=deficit,
            )
        except Exception as e:
            print(f"[Brain Error] {acting.name} equalize: {e}. Falling back to default.")
            fight, suggested_add = (acting.coins >= deficit), deficit

        if not fight:
            # Concede immediately
            print(f"{acting.name} concedes!")
            if acting is battle.player2:
                battle.player1.cards_won.append(battle.card2)
                battle.winner = battle.player1
            else:
                battle.player2.cards_won.append(battle.card1)
                battle.winner = battle.player2
            continue

        # Fight path: enforce the rules you stated
        if acting.coins >= deficit:
            # Must match fully (brains cannot choose to half-match if they can cover)
            spend = deficit
            acting.coins -= spend
            if acting is battle.player2:
                battle.bet2 += spend
            else:
                battle.bet1 += spend
        else:
            # Cannot fully match
            if acting.coins > 0:
                # Add what they can; keep battle alive with unequal bets
                spend = acting.coins
                acting.coins = 0
                if acting is battle.player2:
                    battle.bet2 += spend
                else:
                    battle.bet1 += spend
            else:
                # coins == 0 -> bank adds 1 to their side (no bankroll change)
                print(Back.RED + Fore.WHITE +  f"{acting.name} fights with bank support (1)." + Style.RESET_ALL)
                if acting is battle.player2:
                    battle.bet2 += 1
                else:
                    battle.bet1 += 1


def play_battle_phase(players, battles, board, round_num, battle_index):
    print(f"\n>>> Battle Phase {battle_index + 1} in Round {round_num}")
    for player in players:
        print(f"player {player.name} has {player.front_coins}")
        player_assign_cards_and_bets(player, players, board, round_num, battle_index)
    for player in reversed(players):
        player_additional_battle_bets(player, players, board, round_num, battle_index)
    equalize_all_battles(battles, players, board, round_num, battle_index)
    resolve_battles(battles, board, players)
    for player in players:
        if player.battles[0].winner == player and ((player.battles[0].player1 == player and player.battles[0].bet1 != 0) or player.battles[0].player2 == player and player.battles[0].bet2 != 0):
            if player.battles[1].winner == player and ((player.battles[1].player1 == player and player.battles[1].bet1 != 0) or player.battles[1].player2 == player and player.battles[1].bet2 != 0):
                if player.battles[2].winner == player and ((player.battles[2].player1 == player and player.battles[2].bet1 != 0) or player.battles[2].player2 == player and player.battles[2].bet2 != 0):
                    give_total_domination(player, board)

    for battle in battles:
        correct_bets(battle)
        battle.reset_battle()
