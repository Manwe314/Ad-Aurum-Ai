from .models import Player, Deck, Battle
from .board import BettingBoard
from .decisions import choose_favored_faction, choose_betting_type
from .phase4 import play_battle_phase
from .rounds import determine_round_winner, evaluate_favored_factions
from .brains.base import PlayerBrain, Traits
import random
from typing import List, Dict, Optional
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER, GAME_ENGINE_PIRINTS, LOGGER, LOGGING
from .brains.base import DeckMemory
from .utils.logger_html import HtmlLogger



def rotate_players(players: list) -> list:
    if not players:
        return players
    return players[1:] + players[:1]


def simulate_game(num_players=4, num_battles=3, starting_coins=10):
    if GAME_ENGINE_PIRINTS:
        print("=== Starting Simulation ===")
    players = [Player(f"P{i+1}", starting_coins) for i in range(num_players)]
    board = BettingBoard()
    names = []

    for i, p in enumerate(players):
        p.brain = PlayerBrain(rng=random.Random(1000 + i), traits=Traits(
            aggressiveness = 50, 
            risk_tolerance = 50, 
            bluffiness = 50,
            herding = 50,
            ev_adherence = 50,
            exploration = 5,
        ))
        names.append(p.name)
    
    if LOGGING:
        LOGGER.set_players(names)
        
    for round_num in range(1, num_players + 1):
        if GAME_ENGINE_PIRINTS:
            print(f"\n--- Round {round_num} ---")
        if LOGGING:
            LOGGER.section(f"Round {round_num}")
        board.reset()
        deck = Deck()
        deck.shuffle()
        for player in players:
            player.reset_round()
        total_back = 0
        total_front = 0

        #phase one
        if LOGGING:
            LOGGER.section("Recap before phase 1")
            for p in players:
                LOGGER.log_cat("debug", text=f"having in total {p.coins + p.front_coins}", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")
            LOGGER.section("Phase 1: Choosing Favored Faction")
        for player in players:
            choose_favored_faction(player, players)

        #phase two
        cards_per_player = num_battles * 3 + 1
        if LOGGING:
            LOGGER.section("Recap before phase 2")
            total_back = 0
            total_front = 0
            for p in players:
                LOGGER.log_cat("debug", text="", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")
            LOGGER.section("Phase 2: Drawing Cards")
        for player in players:
            player.cards = deck.draw(cards_per_player)
            player.brain.brain_memory = DeckMemory()
            player.brain.brain_memory.remove_cards(player.cards)
            if player.name == TARGET_PLAYER:
                print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
            if GAME_ENGINE_PIRINTS:
                print(f"{player.name} draws: {player.cards}" + Style.RESET_ALL)
            LOGGER.log_player(category="info", text=f"draws: {player.cards}", player=player.name)

        #phase three
        if LOGGING:
            LOGGER.section("Recap before phase 3")
            total_back = 0
            total_front = 0
            for p in players:
                LOGGER.log_cat("debug", text="", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")    
            LOGGER.section("Phase 3: Choosing Betting Types")
        for player in players:
            choose_betting_type(player, players, board)

        #phase four
        if LOGGING:
            LOGGER.section("Recap before phase 4")
            total_back = 0
            total_front = 0
            for p in players:
                LOGGER.log_cat("debug", text="", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")
            LOGGER.section("Phase 4: BATTLE!")
        battles = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                battle = Battle(players[i], players[j])
                players[i].battles.append(battle)
                players[j].battles.append(battle)
                battles.append(battle)

        for battle_index in range(num_battles):
            play_battle_phase(players, battles, board, round_num, battle_index)

        #phase 5
        if LOGGING:
            LOGGER.section("Recap before phase 5")
            total_back = 0
            total_front = 0
            for p in players:
                LOGGER.log_cat("debug", text="", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")
            LOGGER.section("Phase 5: Determining Round Winner")
        determine_round_winner(players, board)

        #phase6
        if LOGGING:
            LOGGER.section("Recap before phase 6")
            total_back = 0
            total_front = 0
            for p in players:
                LOGGER.log_cat("debug", text="", player=p.name , stats=True, player_obj=p)
                total_back += p.coins
                total_front += p.front_coins
            LOGGER.log_cat("debug", f"Total coins in game: {total_back} back + {total_front} front = {total_back + total_front}")
            LOGGER.section("Phase 6: End of Round Cleanup")
        evaluate_favored_factions(players)
        players = rotate_players(players)
    
    for player in players:
            player.reset_round()

    if LOGGING:
        LOGGER.section("Final Results")
    for player in players:
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW + ADDITIONAL_INFO)
        if GAME_ENGINE_PIRINTS:
            print(f"{player.name} ends with {player.coins} coins and {player.rounds_won} rounds won." + Style.RESET_ALL)
        if LOGGING:
            LOGGER.log_player(player.name, f"ends with {player.coins} coins and {player.rounds_won} rounds won.", category="info", stats=True, player_obj=player)
    if LOGGING:
        LOGGER.save()

if __name__ == "__main__":
    simulate_game(num_players=4, num_battles=3, starting_coins=10)
