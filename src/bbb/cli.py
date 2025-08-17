from .models import Player, Deck, Battle
from .board import BettingBoard
from .decisions import choose_favored_faction, choose_betting_type
from .phase4 import play_battle_phase
from .rounds import determine_round_winner, evaluate_favored_factions
from .brains.base import PlayerBrain, Traits
import random
from typing import List, Dict, Optional
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER



def simulate_game(num_players=4, num_battles=3, starting_coins=10):
    print("=== Starting Simulation ===")
    players = [Player(f"P{i+1}", starting_coins) for i in range(num_players)]
    board = BettingBoard()
    deck = Deck()

    for i, p in enumerate(players):
        p.brain = PlayerBrain(rng=random.Random(1000 + i), traits=Traits(
            aggressiveness = 50, 
            risk_tolerance = 50, 
            bluffiness = 20,
            herding = 50,
            ev_adherence = 60,
            exploration = 3,
        ))
    for round_num in range(1, num_players + 1):
        print(f"\n--- Round {round_num} ---")
        board.reset()
        deck = Deck()
        deck.shuffle()
        for player in players:
            player.reset_round()

        #phase one
        for player in players:
            choose_favored_faction(player, players)

        #phase two
        cards_per_player = num_battles * 3 + 1
        for player in players:
            player.cards = deck.draw(cards_per_player)
            if player.name == TARGET_PLAYER:
                print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
            print(f"{player.name} draws: {player.cards}" + Style.RESET_ALL)

        #phase three
        for player in players:
            choose_betting_type(player, players, board)

        #phase four
        battles = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                battle = Battle(players[i], players[j])
                players[i].battles.append(battle)
                players[j].battles.append(battle)
                battles.append(battle)
        #print(Back.LIGHTRED_EX + Fore.LIGHTBLACK_EX + f"\nBattles created: {battles}"  + Style.RESET_ALL)

        for battle_index in range(num_battles):
            play_battle_phase(players, battles, board, round_num, battle_index)

        #phase 5
        determine_round_winner(players, board)

        #phase6
        evaluate_favored_factions(players)
    
    for player in players:
            player.reset_round()

    for player in players:
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
        print(f"{player.name} ends with {player.coins} coins and {player.rounds_won} rounds won." + Style.RESET_ALL)

if __name__ == "__main__":
    simulate_game(num_players=4, num_battles=3, starting_coins=10)
