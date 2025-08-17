from .models import GladiatorType
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER

def evaluate_favored_factions(players):
    for player in players:
        for p in players:
            if p.name == player.favored_faction:
                gained = p.front_coins // 2
                player.coins += gained
                if player.name == TARGET_PLAYER:
                    print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
                print(f"{player.name} gains {gained} coins for favoring {player.favored_faction}" + Style.RESET_ALL)
                break

def determine_round_winner(players, board):
    print("\n--- Determining Round Winner ---")
    max_score = -1
    winners = []

    for player in players:
        total_value = sum(card.number for card in player.cards_won)
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
        print(f"{player.name} total card value: {total_value}" + Style.RESET_ALL)
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
        # if winner.name == 'P1':
        #     print(Back.CYAN + f"{total_gain} given to {winner.name} atm: {winner.front_coins}" + Style.RESET_ALL)
        print(f"{winner.name} wins the round and collects {total_gain} coins!")
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
        # if p.name == 'P1':
        #     print(Back.CYAN + f"{split_gain} given to {p.name} atm: {p.front_coins}" + Style.RESET_ALL)
        p.rounds_won += 1

    if len(final_winners) == 1:
        print(f"{final_winners[0].name} wins tie-break and gets {split_gain} coins!")
    else:
        print(f"Tie remains among {[p.name for p in final_winners]}. Each receives {split_gain} coins.")
