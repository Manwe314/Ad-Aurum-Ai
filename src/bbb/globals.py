from .utils.logger_html import HtmlLogger
ADDITIONAL_INFO = "" # extra text to add on highlighted player lines
NUMBER_OF_BATTLES = 3 # number of battles per round
NUM_PLAYERS = 4 # number of players in the game
TARGET_PLAYER = None # the player whose lines to highlight
FOCUS_ON_BET_SIZING = False # set to True to enable debug prints
FOCUS_ON_CARD_PLAY = False # set to True to enable debug prints
FOCUS_ON_BATTLE_INITIAL_BET = False # set to True to enable debug prints
FOCUS_ON_ADDITIONAL_BETS = False # set to True to enable debug prints
FOCUS_ON_EQUALIZING_BETS = False # set to True to enable debug prints
GAME_ENGINE_PIRINTS = False # set to True to enable debug prints
LOGGING = True # set to False to disable logging
LOGGER = HtmlLogger(path="./game_report.html", title="Game Report")

WW_VARS = {
    "ww_aggr": 0.08,
    "ww_aggr_prelim": 0.12,
    "ww_bank_bias": 0.3,
    "ww_battle": 0.8,
    "ww_belief": 1.0,
    "ww_bluff": 1.0,
    "ww_bluff_hint": 0.25,
    "ww_concede_mask": 0.7,
    "ww_countp": 0.9,
    "ww_deficit_pen": 0.33,
    "ww_dom": 0.5,
    "ww_ev": 1.0,
    "ww_ev_prelim": 0.95,
    "ww_ev_card_bet": 1.0,
    "ww_hand": 1.0,
    "ww_hand_count": 1.0,
    "ww_hand_strength": 0.3,
    "ww_last_bias": 1.2,
    "ww_liq_call_cost": 0.35,
    "ww_liq_later_mult": 1.4,
    "ww_liq_linear": 0.35,
    "ww_liq_linear_prelim": 0.85,
    "ww_liq_short": 0.4,
    "ww_liquidty": 0.8,
    "ww_none_center_bias": 0.05,
    "ww_open": 0.8,
    "ww_reserve_per_pre": 0.3,
    "ww_round": 0.96,
    "ww_short_mult": 0.4,
    "ww_short_mult_base": 0.6,
    "ww_stubborn_boost": 0.6,
    "ww_td_drive": 0.8,
    "ww_td_drive_lat": 0.5,
    "ww_td_later_penalty": 0.5,
    "ww_td_none_k": 0.6,
    "ww_td_norm_later": 1.2,
    "ww_td_norm_now": 2.0,
    "ww_td_now_boost": 0.02,
    "ww_td_tempo": 0.6,
    "ww_td_tempo_lat": 0.8,
    "ww_td_zero_later": 0.65,
    "ww_td_zero_now": 1.2,
    "ww_theta": 0.1
}
