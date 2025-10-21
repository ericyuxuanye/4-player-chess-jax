"""
Constants for 4-player chess environment.
"""

import jax.numpy as jnp

# Piece types
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PIECE_TYPES = [EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
PIECE_NAMES = {
    EMPTY: "Empty",
    PAWN: "Pawn",
    KNIGHT: "Knight",
    BISHOP: "Bishop",
    ROOK: "Rook",
    QUEEN: "Queen",
    KING: "King"
}

# Players (clockwise order)
RED = 0
BLUE = 1
YELLOW = 2
GREEN = 3

NUM_PLAYERS = 4
PLAYER_NAMES = {RED: "Red", BLUE: "Blue", YELLOW: "Yellow", GREEN: "Green"}
PLAYER_COLORS = {RED: "🔴", BLUE: "🔵", YELLOW: "🟡", GREEN: "🟢"}

# Board dimensions
BOARD_SIZE = 14  # 14x14 grid
CENTRAL_SIZE = 8  # Central 8x8 area
EXTENSION_WIDTH = 3  # Width of each player's extension

# Number of valid squares on the board (cross-shaped)
# Central 8x8 = 64
# 4 extensions of 3x8 = 96
# Total = 160
NUM_VALID_SQUARES = 160

# Action space dimensions
# source_square (160) × dest_square (160) × promotion_type (4)
NUM_PROMOTION_TYPES = 4  # Queen, Rook, Bishop, Knight
ACTION_SPACE_SIZE = NUM_VALID_SQUARES * NUM_VALID_SQUARES * NUM_PROMOTION_TYPES

# Promotion piece types (indices for promotion encoding)
PROMOTE_QUEEN = 0
PROMOTE_ROOK = 1
PROMOTE_BISHOP = 2
PROMOTE_KNIGHT = 3

# Board channel indices
CHANNEL_PIECE_TYPE = 0
CHANNEL_OWNER = 1
CHANNEL_HAS_MOVED = 2
CHANNEL_VALID_SQUARE = 3
NUM_BOARD_CHANNELS = 4

# Observation channels (ego-centric view)
# 7 piece types × 4 players = 28
# + 1 valid squares mask
# + 1 en passant square
# + 4 player active flags
# + 1 current player indicator
NUM_OBSERVATION_CHANNELS = 35

# Piece movement offsets
# Knight moves: L-shape
KNIGHT_OFFSETS = jnp.array([
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [1, -2], [1, 2], [2, -1], [2, 1]
], dtype=jnp.int32)

# King moves: 8 adjacent squares
KING_OFFSETS = jnp.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1]
], dtype=jnp.int32)

# Sliding piece directions
# Bishop: 4 diagonals
BISHOP_DIRECTIONS = jnp.array([
    [-1, -1], [-1, 1], [1, -1], [1, 1]
], dtype=jnp.int32)

# Rook: 4 orthogonal
ROOK_DIRECTIONS = jnp.array([
    [-1, 0], [1, 0], [0, -1], [0, 1]
], dtype=jnp.int32)

# Queen: combination of bishop and rook
QUEEN_DIRECTIONS = jnp.concatenate([BISHOP_DIRECTIONS, ROOK_DIRECTIONS], axis=0)

# Pawn move directions for each player (forward direction)
# Red moves up (negative row), Blue moves left (negative col),
# Yellow moves down (positive row), Green moves right (positive col)
PAWN_FORWARD = {
    RED: jnp.array([-1, 0], dtype=jnp.int32),
    BLUE: jnp.array([0, -1], dtype=jnp.int32),
    YELLOW: jnp.array([1, 0], dtype=jnp.int32),
    GREEN: jnp.array([0, 1], dtype=jnp.int32),
}

# Pawn double move (from starting rank)
PAWN_DOUBLE_FORWARD = {
    RED: jnp.array([-2, 0], dtype=jnp.int32),
    BLUE: jnp.array([0, -2], dtype=jnp.int32),
    YELLOW: jnp.array([2, 0], dtype=jnp.int32),
    GREEN: jnp.array([0, 2], dtype=jnp.int32),
}

# Pawn capture directions (diagonal forward)
PAWN_CAPTURES = {
    RED: jnp.array([[-1, -1], [-1, 1]], dtype=jnp.int32),
    BLUE: jnp.array([[-1, -1], [1, -1]], dtype=jnp.int32),
    YELLOW: jnp.array([[1, -1], [1, 1]], dtype=jnp.int32),
    GREEN: jnp.array([[-1, 1], [1, 1]], dtype=jnp.int32),
}

# Starting ranks for pawns (for double move)
PAWN_START_RANKS = {
    RED: 12,     # Row 12
    BLUE: 12,    # Col 12 (treating as "rank" for Blue)
    YELLOW: 1,   # Row 1
    GREEN: 1,    # Col 1 (treating as "rank" for Green)
}

# Promotion ranks (8th rank for each player)
PROMOTION_RANKS = {
    RED: 3,      # Reaches row 3 (Yellow's territory)
    BLUE: 3,     # Reaches col 3 (Green's territory)
    YELLOW: 10,  # Reaches row 10 (Red's territory)
    GREEN: 10,   # Reaches col 10 (Blue's territory)
}

# Scoring values for captured pieces
PIECE_VALUES = {
    EMPTY: 0,
    PAWN: 1,
    KNIGHT: 3,
    BISHOP: 3,
    ROOK: 5,
    QUEEN: 9,
    KING: 0,  # Can't capture king
}

# Scoring bonuses
SCORE_CHECKMATE = 20
SCORE_STALEMATE_SELF = 20
SCORE_STALEMATE_OPPONENT_PER_REMAINING = 10
SCORE_CHECK_TWO_KINGS = 5
SCORE_CHECK_THREE_KINGS = 5
SCORE_CHECK_TWO_KINGS_NON_QUEEN = 5
SCORE_CHECK_THREE_KINGS_NON_QUEEN = 20

# Special piece value for promoted queen
PROMOTED_QUEEN_VALUE = 1

# Game limits
MAX_MOVES = 500  # Maximum moves before draw
NO_PROGRESS_LIMIT = 50  # 50-move rule (no capture or pawn move)

# Invalid player marker
INACTIVE_PLAYER = -1

# ASCII rendering characters
ASCII_PIECES = {
    (RED, PAWN): "♙", (RED, KNIGHT): "♘", (RED, BISHOP): "♗",
    (RED, ROOK): "♖", (RED, QUEEN): "♕", (RED, KING): "♔",
    (BLUE, PAWN): "♟", (BLUE, KNIGHT): "♞", (BLUE, BISHOP): "♝",
    (BLUE, ROOK): "♜", (BLUE, QUEEN): "♛", (BLUE, KING): "♚",
    (YELLOW, PAWN): "♙̲", (YELLOW, KNIGHT): "♘̲", (YELLOW, BISHOP): "♗̲",
    (YELLOW, ROOK): "♖̲", (YELLOW, QUEEN): "♕̲", (YELLOW, KING): "♔̲",
    (GREEN, PAWN): "♟̲", (GREEN, KNIGHT): "♞̲", (GREEN, BISHOP): "♝̲",
    (GREEN, ROOK): "♜̲", (GREEN, QUEEN): "♛̲", (GREEN, KING): "♚̲",
}

# Simple ASCII for better compatibility
ASCII_PIECES_SIMPLE = {
    (RED, PAWN): "rP", (RED, KNIGHT): "rN", (RED, BISHOP): "rB",
    (RED, ROOK): "rR", (RED, QUEEN): "rQ", (RED, KING): "rK",
    (BLUE, PAWN): "bP", (BLUE, KNIGHT): "bN", (BLUE, BISHOP): "bB",
    (BLUE, ROOK): "bR", (BLUE, QUEEN): "bQ", (BLUE, KING): "bK",
    (YELLOW, PAWN): "yP", (YELLOW, KNIGHT): "yN", (YELLOW, BISHOP): "yB",
    (YELLOW, ROOK): "yR", (YELLOW, QUEEN): "yQ", (YELLOW, KING): "yK",
    (GREEN, PAWN): "gP", (GREEN, KNIGHT): "gN", (GREEN, BISHOP): "gB",
    (GREEN, ROOK): "gR", (GREEN, QUEEN): "gQ", (GREEN, KING): "gK",
}