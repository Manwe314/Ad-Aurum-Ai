import random
from .models import GladiatorType
from .board import BettingBoard
from bbb.brains.base import RandomBrain, Traits, PlayerBrain, DeckMemory
from .observations import PlayerView, build_player_view
from colorama import Fore, Back, Style
from .globals import ADDITIONAL_INFO, TARGET_PLAYER, FOCUS_ON_CARD_PLAY

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

def player_assign_cards_and_bets(player, all_players, board):
    """
    Engine hook: if the player has a brain with assign_cards_and_bets(view),
    use it; otherwise fall back to the old random assignment.

    Args:
        player: the Player instance taking the action
        all_players: the ordered list of all Player objects (needed to build the view)
        board: the BettingBoard (needed to build the view)
    """
    # If a brain is present, delegate to it
    if not hasattr(player, "brain") or player.brain is None:
        print(f"[Brain Missing] {player.name} has no brain; using random assignment.")
    if hasattr(player, "brain") and player.brain is not None:
        # Build the PlayerView for this player
        view = build_player_view(player, all_players, board, player.battles, round_number=0, battle_phase_index=None)

        # Ask the brain for (battle_id, card, show_type, bet)
        try:
            results = player.brain.assign_cards_and_bets(view)
            if FOCUS_ON_CARD_PLAY:
                print(Back.WHITE + Fore.LIGHTBLACK_EX + f"RESULTS {results}" + Style.RESET_ALL)
        except Exception as e:
            print(f"[Brain Error] {player.name}: {e}. Falling back to random.")
            results = []

        if results:
            for (bid, card, show_type, bet) in results:
                idx = view.battle_view_to_idx.get(bid)
                battle = player.battles[idx] if idx is not None else None
                if battle is None:
                    continue  # skip unknown battle id

                # Ensure the chosen card is actually in hand
                if card not in player.cards:
                    # If desync happens, pick a fallback from hand
                    if player.cards:
                        card = player.cards.pop()
                    else:
                        continue
                else:
                    player.cards.remove(card)

                # Clamp bet to available coins; must be >= 1 by rule
                bet = max(1, min(player.coins, int(bet)))
                player.coins -= bet
                # If you want to mirror older behavior, uncomment the next line:
                # player.front_coins += bet

                # Apply card + visibility to the shared Battle object
                if player == battle.player1:
                    battle.set_cards(card, show_type, battle.card2, battle.card2_shows_type)
                    battle.bet1 = bet
                    opponent_name = battle.player2.name
                else:
                    battle.set_cards(battle.card1, battle.card1_shows_type, card, show_type)
                    battle.bet2 = bet
                    opponent_name = battle.player1.name

                if player.name == TARGET_PLAYER:
                    print(Fore.BLACK + Back.YELLOW +ADDITIONAL_INFO)
                print(f"{player.name} (brain) assigns card {card} showing {'type' if show_type else 'number'} "
                      f"with bet {bet} in battle against {opponent_name}" + Style.RESET_ALL)
            return  # brain path done

    # --- Fallback: your original random logic (unchanged) ---
    for battle in player.battles:
        if not player.cards:
            continue
        card = player.cards.pop()
        show_type = random.choice([True, False])
        bet = min(1, player.coins)
        player.coins -= bet
        # player.front_coins += bet  # keep commented to match your current behavior
        if player == battle.player1:
            battle.set_cards(card, show_type, battle.card2, battle.card2_shows_type)
            battle.bet1 = bet
            opponent_name = battle.player2.name
        else:
            battle.set_cards(battle.card1, battle.card1_shows_type, card, show_type)
            battle.bet2 = bet
            opponent_name = battle.player1.name
        if player.name == TARGET_PLAYER:
            print(Fore.BLACK + Back.YELLOW +ADDITIONAL_INFO)
        print(f"{player.name} assigns card {card} showing {'type' if show_type else 'number'} "
              f"with bet {bet} in battle against {opponent_name}" + Style.RESET_ALL)

        

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
