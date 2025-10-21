"""
4-Player Chess JAX Environment

A JAX-native implementation of 4-player chess supporting parallel rollouts
for reinforcement learning training pipelines.
"""

from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    RED, BLUE, YELLOW, GREEN,
    BOARD_SIZE, NUM_VALID_SQUARES, NUM_PLAYERS
)

__version__ = "0.1.0"

__all__ = [
    "FourPlayerChessEnv",
    "EnvState",
    "EMPTY", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING",
    "RED", "BLUE", "YELLOW", "GREEN",
    "BOARD_SIZE", "NUM_VALID_SQUARES", "NUM_PLAYERS"
]