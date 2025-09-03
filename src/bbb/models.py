import random
from enum import Enum
from colorama import Fore, Back, Style

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
        self.fight_regardless = False  # New attribute to indicate forced fight

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
        self.fight_regardless = False

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
        self.brain = None
        self.training_target = False

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
