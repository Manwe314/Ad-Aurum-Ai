import random
from collections import defaultdict
from enum import Enum

# Define Gladiator Types
class GladiatorType(Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'

# Define Card class
class Card:
    def __init__(self, gladiator_type: GladiatorType, number: int):
        self.type = gladiator_type
        self.number = number

    def __repr__(self):
        return f"Card({self.type.value}, {self.number})"

# Battle class
class Battle:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.card1 = None
        self.card2 = None
        self.bet1 = 0
        self.bet2 = 0
        self.card1_shows_type = True
        self.card2_shows_type = True
        self.winner = None  # Track who won this battle

    def set_cards(self, card1, show1, card2, show2):
        self.card1 = card1
        self.card2 = card2
        self.card1_shows_type = show1
        self.card2_shows_type = show2

    def set_bets(self, bet1, bet2):
        self.bet1 = bet1
        self.bet2 = bet2
        
    
    def reset_battle(self):
        self.winner = None
        self.card1 = None
        self.card2 = None
        self.bet1 = 0
        self.bet2 = 0
        self.card1_shows_type = True
        self.card2_shows_type = True

    def __repr__(self):
        return f"Battle({self.player1.name} vs {self.player2.name}, Bet1: {self.bet1}, Bet2: {self.bet2})"

# Player class
class Player:
    def __init__(self, name, starting_coins):
        self.name = name
        self.coins = starting_coins
        self.front_coins = 0
        self.cards = []
        self.battles = []
        self.favored_faction = None
        self.cards_won = []
        self.rounds_won = 0

    def reset_round(self):
        self.cards = []
        self.battles.clear()
        self.favored_faction = None
        self.cards_won.clear()
        self.coins += self.front_coins
        self.front_coins = 0

    def __repr__(self):
        return f"Player({self.name}, Coins: {self.coins}, Front: {self.front_coins})"

# Deck logic
class Deck:
    def __init__(self):
        self.cards = self.generate_deck()

    def generate_deck(self):
        cards = []
        values = [1, 3, 5, 5, 7, 7, 9, 10, 11]
        for g_type in GladiatorType:
            for number in values:
                cards.append(Card(g_type, number))
        return cards

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, count):
        drawn = self.cards[:count]
        self.cards = self.cards[count:]
        return drawn

# Betting board
class BettingBoard:
    def __init__(self):
        self.bets = defaultdict(list)

    def place_bet(self, player, gladiator_type, amount):
        self.bets[gladiator_type].append((player, amount))

    def get_total_bets(self):
        return {g_type: sum(amount for _, amount in entries) for g_type, entries in self.bets.items()}

    def reset(self):
        self.bets.clear()

    def get_player_bet_type(self, player):
        for g_type, entries in self.bets.items():
            for p, _ in entries:
                if p == player:
                    return g_type
        return None 

# Strength calculation logic

def calculate_strength(card: Card, board: BettingBoard, opponent_card: Card):
    base_strength = card.number * board.get_total_bets().get(card.type, 0)
    three_x_chain = [GladiatorType.A, GladiatorType.B, GladiatorType.C, GladiatorType.D, GladiatorType.E]
    two_x_chain = [GladiatorType.A, GladiatorType.C, GladiatorType.E, GladiatorType.B, GladiatorType.D]

    def rel_bonus(type1, type2, chain, multiplier):
        idx1 = chain.index(type1)
        return multiplier if chain[(idx1 + 1) % len(chain)] == type2 else 1

    bonus_3x = rel_bonus(card.type, opponent_card.type, three_x_chain, 3)
    bonus_2x = rel_bonus(card.type, opponent_card.type, two_x_chain, 2)
    return base_strength * max(bonus_3x, bonus_2x)

# Favored faction evaluation

def evaluate_favored_factions(players):
    for player in players:
        for p in players:
            if p.name == player.favored_faction:
                gained = p.front_coins // 2
                player.coins += gained
                print(f"{player.name} gains {gained} coins for favoring {player.favored_faction}")
                break

# Player decisions

def choose_favored_faction(player, other_players):
    player.favored_faction = random.choice([p.name for p in other_players if p.name != player.name])
    print(f"{player.name} favors {player.favored_faction}")

def choose_betting_type(player, board):
    chosen_type = random.choice(list(GladiatorType))
    bet_amount = 1
    player.coins -= bet_amount
    board.place_bet(player, chosen_type, bet_amount)
    print(f"{player.name} bets {bet_amount} on {chosen_type.value}")

def player_assign_cards_and_bets(player):
    for battle in player.battles:
        if not player.cards:
            continue
        card = player.cards.pop()
        show_type = random.choice([True, False])
        bet = min(1, player.coins)
        player.coins -= bet
        player.front_coins += bet
        if player == battle.player1:
            battle.set_cards(card, show_type, battle.card2, battle.card2_shows_type)
            battle.bet1 = bet
        else:
            battle.set_cards(battle.card1, battle.card1_shows_type, card, show_type)
            battle.bet2 = bet
        print(f"{player.name} assigns card {card} showing {'type' if show_type else 'number'} with bet {bet} in battle against {battle.player2.name if player == battle.player1 else battle.player1.name}")

def player_additional_battle_bets(player):
    for battle in player.battles:
        additional_bet = min(1, player.coins)
        if player == battle.player1:
            battle.bet1 += additional_bet
        else:
            battle.bet2 += additional_bet
        player.coins -= additional_bet
        player.front_coins += additional_bet
        print(f"{player.name} adds additional bet {additional_bet} in battle against {battle.player2.name if player == battle.player1 else battle.player1.name}")

def equalize_all_battles(battles):
    for battle in battles:
        if battle.bet1 > battle.bet2:
            diff = battle.bet1 - battle.bet2
            if battle.player2.coins >= diff:
                battle.player2.coins -= diff
                battle.player2.front_coins += diff
                battle.bet2 += diff
            else:
                print(f"{battle.player2.name} concedes!")
                battle.player1.cards_won.append(battle.card2)
                battle.winner = battle.player1
        elif battle.bet2 > battle.bet1:
            diff = battle.bet2 - battle.bet1
            if battle.player1.coins >= diff:
                battle.player1.coins -= diff
                battle.player1.front_coins += diff
                battle.bet1 += diff
            else:
                print(f"{battle.player1.name} concedes!")
                battle.player2.cards_won.append(battle.card1)
                battle.winner = battle.player2

def resolve_battles(battles, board):
    for battle in battles:
        if battle.card1 is None or battle.card2 is None or battle.winner is not None:
            continue
        print(f"\nResolving {battle}:")
        str1 = calculate_strength(battle.card1, board, battle.card2)
        str2 = calculate_strength(battle.card2, board, battle.card1)
        print(f"{battle.player1.name} strength: {str1}, {battle.player2.name} strength: {str2}")

        if str1 > str2:
            battle.player1.cards_won.append(battle.card2)
            battle.winner = battle.player1
            battle.player1.front_coins += battle.bet2
            print(f"{battle.player1.name} wins and takes {battle.card2}!")
        elif str2 > str1:
            battle.player2.cards_won.append(battle.card1)
            battle.winner = battle.player2
            battle.player2.front_coins += battle.bet1
            print(f"{battle.player2.name} wins and takes {battle.card1}!")
        else:
            print("Stalemate. No one wins this battle.")
            battle.winner = None


def correct_bets(battle):
    if battle.winner is None:
        return
    if battle.winner == battle.player1:
        battle.player2.front_coins -= battle.bet2
    if battle.winner == battle.player2:
        battle.player1.front_coins -= battle.bet1

def give_total_domination(player, board):
    max_bet = 0
    second_max_bet = 0
    for g_type, entries in board.bets.items():
        for _, amount in entries:
            if amount >= max_bet:
                max_bet = amount
                second_max_bet = max_bet
            elif amount > second_max_bet:
                second_max_bet = amount
    print(f"{player.name} achieves total domination! Gains {second_max_bet} coins.")
    player.front_coins += second_max_bet
# Battle Phase handler

def play_battle_phase(players, battles, board, round_num, battle_index):
    print(f"\n>>> Battle Phase {battle_index + 1} in Round {round_num}")
    for player in players:
        print(f"player {player.name} has {player.front_coins}")
        player_assign_cards_and_bets(player)
    for player in reversed(players):
        player_additional_battle_bets(player)
    equalize_all_battles(battles)
    resolve_battles(battles, board)
    for player in players:
        if player.battles[0].winner == player and ((player.battles[0].player1 == player and player.battles[0].bet1 != 0) or player.battles[0].player2 == player and player.battles[0].bet2 != 0):
            if player.battles[1].winner == player and ((player.battles[1].player1 == player and player.battles[1].bet1 != 0) or player.battles[1].player2 == player and player.battles[1].bet2 != 0):
                if player.battles[2].winner == player and ((player.battles[2].player1 == player and player.battles[2].bet1 != 0) or player.battles[2].player2 == player and player.battles[2].bet2 != 0):
                    give_total_domination(player, board)

    for battle in battles:
        correct_bets(battle)
        battle.reset_battle()

def determine_round_winner(players, board):
    print("\n--- Determining Round Winner ---")
    max_score = -1
    winners = []

    for player in players:
        total_value = sum(card.number for card in player.cards_won)
        print(f"{player.name} total card value: {total_value}")
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
        p.rounds_won += 1

    if len(final_winners) == 1:
        print(f"{final_winners[0].name} wins tie-break and gets {split_gain} coins!")
    else:
        print(f"Tie remains among {[p.name for p in final_winners]}. Each receives {split_gain} coins.")

# Game simulation

def simulate_game(num_players=4, num_battles=3, starting_coins=10):
    print("=== Starting Simulation ===")
    players = [Player(f"P{i+1}", starting_coins) for i in range(num_players)]
    board = BettingBoard()
    deck = Deck()

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
            print(f"{player.name} draws: {player.cards}")

        #phase three
        for player in players:
            choose_betting_type(player, board)

        #phase four
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
        determine_round_winner(players, board)

        #phase6
        evaluate_favored_factions(players)
    
    for player in players:
        print(f"{player.name} ends with {player.coins} coins and {player.rounds_won} rounds won.")

if __name__ == "__main__":
    simulate_game(num_players=4, num_battles=3, starting_coins=10)


#invastigate bugs / add final winner calculation / add running out of money mechanics. 