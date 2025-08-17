from .decisions import player_assign_cards_and_bets, player_additional_battle_bets
from .combat import resolve_battles, correct_bets, give_total_domination

def equalize_all_battles(battles):
    for battle in battles:
        if battle.bet1 > battle.bet2:
            diff = battle.bet1 - battle.bet2
            if battle.player2.coins >= diff:
                battle.player2.coins -= diff
                battle.bet2 += diff
            else:
                print(f"{battle.player2.name} concedes!")
                battle.player1.cards_won.append(battle.card2)
                battle.winner = battle.player1
        elif battle.bet2 > battle.bet1:
            diff = battle.bet2 - battle.bet1
            if battle.player1.coins >= diff:
                battle.player1.coins -= diff
                battle.bet1 += diff
            else:
                print(f"{battle.player1.name} concedes!")
                battle.player2.cards_won.append(battle.card1)
                battle.winner = battle.player2

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
