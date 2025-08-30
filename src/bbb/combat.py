from .models import GladiatorType
from colorama import Fore, Back, Style
from .globals import GAME_ENGINE_PIRINTS

def rel_bonus(type1, type2, chain, multiplier):
    idx1 = chain.index(type1)
    return multiplier if chain[(idx1 + 1) % len(chain)] == type2 else 1

# Strength calculation logic
def calculate_strength(card, board, opponent_card):
    base_strength = card.number * board.get_total_bets().get(card.type, 0)
    three_x_chain = [GladiatorType.A, GladiatorType.B, GladiatorType.C, GladiatorType.D, GladiatorType.E]
    two_x_chain = [GladiatorType.A, GladiatorType.C, GladiatorType.E, GladiatorType.B, GladiatorType.D]

    bonus_3x = rel_bonus(card.type, opponent_card.type, three_x_chain, 3)
    bonus_2x = rel_bonus(card.type, opponent_card.type, two_x_chain, 2)
    return base_strength * max(bonus_3x, bonus_2x)

def resolve_battles(battles, board, players):
    three_x_chain = [GladiatorType.A, GladiatorType.B, GladiatorType.C, GladiatorType.D, GladiatorType.E]
    two_x_chain = [GladiatorType.A, GladiatorType.C, GladiatorType.E, GladiatorType.B, GladiatorType.D]
    for battle in battles:
        if battle.card1 is None or battle.card2 is None or battle.winner is not None:
            continue
        if GAME_ENGINE_PIRINTS:
            print(Back.LIGHTRED_EX + Fore.LIGHTBLACK_EX + f"\nResolving {battle}:" + Style.RESET_ALL)
        str1 = calculate_strength(battle.card1, board, battle.card2)
        str2 = calculate_strength(battle.card2, board, battle.card1)
        if GAME_ENGINE_PIRINTS:
            print(f"{battle.player1.name} strength: {str1}, {battle.player2.name} strength: {str2}")
        for p in players:
            p.brain.brain_memory.remove_cards([battle.card1, battle.card2])
        if str1 > str2:
            battle.player1.cards_won.append(battle.card2)
            battle.winner = battle.player1
            if GAME_ENGINE_PIRINTS:
                print(f"{battle.player1.name} wins and takes {battle.card2}!" + Style.RESET_ALL)
        elif str2 > str1:
            battle.player2.cards_won.append(battle.card1)
            battle.winner = battle.player2
            if GAME_ENGINE_PIRINTS:
                print(f"{battle.player2.name} wins and takes {battle.card1}!" + Style.RESET_ALL)
        elif str1 == str2:
            if rel_bonus(battle.card1.type, battle.card2.type, three_x_chain, 3) > 1 or rel_bonus(battle.card1.type, battle.card2.type, two_x_chain, 2) > 1:
                battle.player1.cards_won.append(battle.card2)
                battle.winner = battle.player1
                if GAME_ENGINE_PIRINTS:
                    print(f"{battle.player1.name} wins by type bonus and takes {battle.card2}!" + Style.RESET_ALL)
            if rel_bonus(battle.card2.type, battle.card1.type, three_x_chain, 3) > 1 or rel_bonus(battle.card2.type, battle.card1.type, two_x_chain, 2) > 1:
                battle.player2.cards_won.append(battle.card1)
                battle.winner = battle.player2
                if GAME_ENGINE_PIRINTS:
                    print(f"{battle.player2.name} wins by type bonus and takes {battle.card1}!" + Style.RESET_ALL)
        else:
            if GAME_ENGINE_PIRINTS:
                print("Stalemate. No one wins this battle.")
            battle.winner = None

def correct_bets(battle):
    if battle.winner is None:
        battle.player1.front_coins += battle.bet1
        battle.player2.front_coins += battle.bet2
    if battle.winner == battle.player1:
        battle.player1.front_coins += battle.bet1 + battle.bet2
        if battle.bet1 > battle.bet2:
            battle.player1.front_coins += (battle.bet1 - battle.bet2)
    if battle.winner == battle.player2:
        battle.player2.front_coins += battle.bet1 + battle.bet2
        if battle.bet2 > battle.bet1:
            battle.player2.front_coins += (battle.bet2 - battle.bet1)

def give_total_domination(player, board):
    max_bet = 0
    second_max_bet = 0
    for g_type, entries in board.bets.items():
        for _, amount in entries:
            if amount >= max_bet:
                second_max_bet = max_bet
                max_bet = amount
            elif amount > second_max_bet:
                second_max_bet = amount
    if GAME_ENGINE_PIRINTS:
        print(Back.BLUE +  f"{player.name} achieves total domination! Gains {second_max_bet} coins." + Style.RESET_ALL)
    player.front_coins += second_max_bet
