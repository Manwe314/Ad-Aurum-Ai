from .models import GladiatorType
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER, GAME_ENGINE_PIRINTS, LOGGER, LOGGING, PARALEL_LOGGING
from Analytics.analytics_logger import get_card_logger, COIN_SOURCES

def evaluate_favored_factions(players, round_num):
    for player in players:
        for p in players:
            if p.name == player.favored_faction:
                gained = p.front_coins // 2
                player.coins += gained
                if player.name == TARGET_PLAYER:
                    print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
                if GAME_ENGINE_PIRINTS:
                    print(f"{player.name} gains {gained} coins for favoring {player.favored_faction}" + Style.RESET_ALL)
                if LOGGING:
                    LOGGER.log_cat("success", f"Gains {gained} coins for favoring {player.favored_faction}", player=player.name)
                if PARALEL_LOGGING:
                    if p == players[-1]:
                        get_card_logger().log_new_coins(round_index=round_num, source=COIN_SOURCES['favor_payout_last'], amount=gained)
                    else:
                        get_card_logger().log_new_coins(round_index=round_num, source=COIN_SOURCES['favor_payout_richest'], amount=gained)
                break

def determine_round_winner(players, board):
    if GAME_ENGINE_PIRINTS:
        print("\n--- Determining Round Winner ---")
    max_score = -1
    winners = []

    for player in players:
        total_value = sum(card.number for card in player.cards_won)
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
        if GAME_ENGINE_PIRINTS:
            print(f"{player.name} total card value: {total_value}" + Style.RESET_ALL)
        if LOGGING:
            LOGGER.log_cat("info", f"Total cards won's value: {total_value}", player=player.name)
        if total_value > max_score:
            max_score = total_value
            winners = [player]
        elif total_value == max_score:
            winners.append(player)

    if len(winners) == 1:
        winner = winners[0]
        winner.rounds_won += 1
        # Collect all bets not on winner’s type
        winner_type = board.get_player_bet_type(winner)
        total_gain = 0
        for g_type, entries in board.bets.items():
            if g_type != winner_type:
                for _, amount in entries:
                    total_gain += amount
        winner.front_coins += total_gain
        if GAME_ENGINE_PIRINTS:
            print(f"{winner.name} wins the round and collects {total_gain} coins!")
        if LOGGING:
            LOGGER.log_cat("success", f"Wins the round and collects {total_gain} coins!", player=winner.name, stats=True, player_obj=winner)
        return

    def rel_sum(g_type, others):
        three_x_chain = [GladiatorType.A, GladiatorType.B, GladiatorType.C, GladiatorType.D, GladiatorType.E]
        two_x_chain   = [GladiatorType.A, GladiatorType.C, GladiatorType.E, GladiatorType.B, GladiatorType.D]

        def bonus(type1, type2, chain, mult):
            idx1 = chain.index(type1)
            return mult if chain[(idx1 + 1) % len(chain)] == type2 else 1

        score = 0
        for other in others:
            score = max(score, bonus(g_type, other, three_x_chain, 3))
            score = max(score, bonus(g_type, other, two_x_chain, 2))
        return score
    
    types = {p: board.get_player_bet_type(p) for p in winners}
    best_score = -1
    final_winners = []

    for p, g_type in types.items():
        others = [t for q, t in types.items() if q != p]
        score = rel_sum(g_type, others)
        if score > best_score:
            best_score = score
            final_winners = [p]
        elif score == best_score:
            final_winners.append(p)

    # Coin collection
    excluded_types = [types[p] for p in final_winners]
    total_gain = sum(
        amount for g_type, entries in board.bets.items()
        if g_type not in excluded_types
        for _, amount in entries
    )
    split_gain = total_gain // len(final_winners)

    for p in final_winners:
        p.front_coins += split_gain
        p.rounds_won += 1

    if len(final_winners) == 1:
        if GAME_ENGINE_PIRINTS:
            print(f"{final_winners[0].name} wins tie-break and gets {split_gain} coins!")
        if LOGGING:
            LOGGER.log_cat("success", f"Wins tie-break and gets {split_gain} coins!", player=final_winners[0].name)
    else:
        if GAME_ENGINE_PIRINTS:
            print(f"Tie remains among {[p.name for p in final_winners]}. Each receives {split_gain} coins.")
        if LOGGING:
            LOGGER.log_cat("success", f"Tie remains among {[p.name for p in final_winners]}. Each receives {split_gain} coins.")
