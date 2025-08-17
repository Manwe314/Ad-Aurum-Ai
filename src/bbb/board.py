from collections import defaultdict

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
