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
LOGGING = False # set to False to disable logging
PARALEL_LOGGING = False # set to True to enable logging in parallel games
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

WW_VARS_TRAINING = {
    "ww_aggr": 1.4172646643898716,
    "ww_aggr_prelim": 0.0680357783154364,
    "ww_bank_bias": 0.5422489005540064,
    "ww_battle": 1.6288349368949209,
    "ww_belief": 0.9332221916645265,
    "ww_bluff": 0.8045876344896115,
    "ww_bluff_hint": 0.6167740698242766,
    "ww_concede_mask": 1.9641044204273586,
    "ww_countp": 1.9145374108674935,
    "ww_deficit_pen": 0.9157089242098937,
    "ww_dom": 0.4803635578145319,
    "ww_ev": 1.121698829069025,
    "ww_ev_prelim": 1.0781342889045147,
    "ww_ev_card_bet": 0.257091138459634,
    "ww_hand": 1.3601611195801935,
    "ww_hand_count": 0.677526883303471,
    "ww_hand_strength": 0.7242490352588119,
    "ww_last_bias": 1.3136714248892924,
    "ww_liq_call_cost": 1.2575117827561664,
    "ww_liq_later_mult": 1.0252288940010326,
    "ww_liq_linear": 1.6475501541010573,
    "ww_liq_linear_prelim": 1.5892671470484465,
    "ww_liq_short": 1.27828087461873,
    "ww_liquidty": 1.8912471015246335,
    "ww_none_center_bias": 0.4771313140869674,
    "ww_open": 1.980327585730281,
    "ww_reserve_per_pre": 1.3036191960865298,
    "ww_round": 1.9089226410850437,
    "ww_short_mult": 1.2432129885592704,
    "ww_short_mult_base": 1.8196332068453676,
    "ww_stubborn_boost": 0.9729666543605129,
    "ww_td_drive": 0.9197677333681333,
    "ww_td_drive_lat": 0.11546563291484649,
    "ww_td_later_penalty": 0.6857907518460801,
    "ww_td_none_k": 1.901988308989757,
    "ww_td_norm_later": 0.8153351558536617,
    "ww_td_norm_now": 1.084781572357671,
    "ww_td_now_boost": 0.49868662442386824,
    "ww_td_tempo": 0.667044436349194,
    "ww_td_tempo_lat": 0.2065857410174497,
    "ww_td_zero_later": 0.8932502423348407,
    "ww_td_zero_now": 0.5682476845202028,
    "ww_theta": 1.1607162584018391
}
