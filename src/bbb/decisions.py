import random
from .models import GladiatorType
from .board import BettingBoard
from bbb.brains.base import RandomBrain, Traits
from .observations import PlayerView, build_player_view
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER

def choose_favored_faction(player, other_players):
    if getattr(player, 'brain', None) is not None:
        view = build_player_view(player, other_players, board=BettingBoard(), battles=[], round_number=0, battle_phase_index=None)
        player.favored_faction = player.brain.pick_favored_faction(view)
    else:
        player.favored_faction = random.choice([p.name for p in other_players if p.name != player.name])
    if player.name == TARGET_PLAYER:
        print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
    print(f"{player.name} favors {player.favored_faction}" + Style.RESET_ALL)

def choose_betting_type(player, players ,board):
    if getattr(player, 'brain', None) is not None:
        view = build_player_view(player, players, board, battles=[], round_number=0, battle_phase_index=None)
        chosen_type, bet_amount = player.brain.pick_representation_bet(view)
        player.coins -= bet_amount
    else:
        # Randomly choose a type for simplicity
        chosen_type = random.choice(list(GladiatorType))
        bet_amount = 1
        player.coins -= bet_amount
    board.place_bet(player, chosen_type, bet_amount)
    if player.name == TARGET_PLAYER:
        print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
    print(f"{player.name} bets {bet_amount} on {chosen_type.value}" + Style.RESET_ALL)

def player_assign_cards_and_bets(player):
    for battle in player.battles:
        if not player.cards:
            continue
        card = player.cards.pop()
        show_type = random.choice([True, False])
        # if player.coins <= 0:
        #     print(Fore.RED + Back.GREEN + f"{player.name} has no coins left to bet." + Style.RESET_ALL)
        bet = min(1, player.coins)
        player.coins -= bet
        if player == battle.player1:
            battle.set_cards(card, show_type, battle.card2, battle.card2_shows_type)
            battle.bet1 = bet
        else:
            battle.set_cards(battle.card1, battle.card1_shows_type, card, show_type)
            battle.bet2 = bet
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW +ADDITIONAL_INFO)
        print(f"{player.name} assigns card {card} showing {'type' if show_type else 'number'} with bet {bet} in battle against {battle.player2.name if player == battle.player1 else battle.player1.name}" + Style.RESET_ALL)

def player_additional_battle_bets(player):
    for battle in player.battles:
        # if player.coins <= 0:
        #     print(Fore.RED + Back.GREEN + f"{player.name} has no coins left to bet." + Style.RESET_ALL)
        additional_bet = min(1, player.coins)
        player.coins -= additional_bet
        if player == battle.player1:
            battle.bet1 += additional_bet
        else:
            battle.bet2 += additional_bet
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
        print(f"{player.name} adds additional bet {additional_bet} in battle against {battle.player2.name if player == battle.player1 else battle.player1.name}" + Style.RESET_ALL)
