"""
Board representation and utilities for 4-player chess.

The board is a 14x14 cross-shaped grid with corners masked out.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
import chex

from four_player_chess.constants import (
    BOARD_SIZE, CENTRAL_SIZE, EXTENSION_WIDTH,
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    RED, BLUE, YELLOW, GREEN, NUM_PLAYERS,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED, CHANNEL_VALID_SQUARE,
    NUM_BOARD_CHANNELS
)


def create_valid_square_mask() -> chex.Array:
    """
    Create a boolean mask indicating valid squares on the 4-player chess board.
    
    The board is cross-shaped:
    - Central 8x8 area is valid (rows 3-10, cols 3-10)
    - 4 extensions of 3x8 for each player
    
    Returns:
        Array of shape (14, 14) with 1 for valid squares, 0 for invalid corners
    """
    mask = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int32)
    
    # Central 8x8 area (rows 3-10, cols 3-10)
    mask = mask.at[3:11, 3:11].set(1)
    
    # Red extension (bottom, rows 11-13, cols 3-10)
    mask = mask.at[11:14, 3:11].set(1)
    
    # Blue extension (right, rows 3-10, cols 11-13)
    mask = mask.at[3:11, 11:14].set(1)
    
    # Yellow extension (top, rows 0-2, cols 3-10)
    mask = mask.at[0:3, 3:11].set(1)
    
    # Green extension (left, rows 3-10, cols 0-2)
    mask = mask.at[3:11, 0:3].set(1)
    
    return mask


def is_valid_square(row: chex.Numeric, col: chex.Numeric, valid_mask: chex.Array) -> chex.Array:
    """
    Check if a square is valid (within bounds and not in a corner).
    
    Args:
        row: Row index
        col: Column index
        valid_mask: Valid square mask from create_valid_square_mask()
    
    Returns:
        Boolean indicating if square is valid
    """
    in_bounds = (row >= 0) & (row < BOARD_SIZE) & (col >= 0) & (col < BOARD_SIZE)
    is_valid = valid_mask[row, col] > 0
    return in_bounds & is_valid


def create_initial_board() -> chex.Array:
    """
    Create the initial board state for 4-player chess.
    
    Returns:
        Array of shape (14, 14, 4) with initial piece positions.
        Channels: [piece_type, owner, has_moved, valid_square]
    """
    board = jnp.zeros((BOARD_SIZE, BOARD_SIZE, NUM_BOARD_CHANNELS), dtype=jnp.int32)
    
    # Set valid square mask
    valid_mask = create_valid_square_mask()
    board = board.at[:, :, CHANNEL_VALID_SQUARE].set(valid_mask)
    
    # Red pieces (bottom, rows 11-13, cols 3-10)
    # Back rank (row 13)
    board = board.at[13, 3, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[13, 4, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[13, 5, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[13, 6, CHANNEL_PIECE_TYPE].set(QUEEN)
    board = board.at[13, 7, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[13, 8, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[13, 9, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[13, 10, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[13, 3:11, CHANNEL_OWNER].set(RED)
    
    # Pawns (row 12)
    board = board.at[12, 3:11, CHANNEL_PIECE_TYPE].set(PAWN)
    board = board.at[12, 3:11, CHANNEL_OWNER].set(RED)
    
    # Blue pieces (right, rows 3-10, cols 11-13)
    # Back rank (col 13)
    board = board.at[3, 13, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[4, 13, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[5, 13, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[6, 13, CHANNEL_PIECE_TYPE].set(QUEEN)
    board = board.at[7, 13, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[8, 13, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[9, 13, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[10, 13, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[3:11, 13, CHANNEL_OWNER].set(BLUE)
    
    # Pawns (col 12)
    board = board.at[3:11, 12, CHANNEL_PIECE_TYPE].set(PAWN)
    board = board.at[3:11, 12, CHANNEL_OWNER].set(BLUE)
    
    # Yellow pieces (top, rows 0-2, cols 3-10)
    # Back rank (row 0)
    board = board.at[0, 3, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[0, 4, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[0, 5, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[0, 6, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[0, 7, CHANNEL_PIECE_TYPE].set(QUEEN)
    board = board.at[0, 8, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[0, 9, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[0, 10, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[0, 3:11, CHANNEL_OWNER].set(YELLOW)
    
    # Pawns (row 1)
    board = board.at[1, 3:11, CHANNEL_PIECE_TYPE].set(PAWN)
    board = board.at[1, 3:11, CHANNEL_OWNER].set(YELLOW)
    
    # Green pieces (left, rows 3-10, cols 0-2)
    # Back rank (col 0)
    board = board.at[3, 0, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[4, 0, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[5, 0, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[6, 0, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[7, 0, CHANNEL_PIECE_TYPE].set(QUEEN)
    board = board.at[8, 0, CHANNEL_PIECE_TYPE].set(BISHOP)
    board = board.at[9, 0, CHANNEL_PIECE_TYPE].set(KNIGHT)
    board = board.at[10, 0, CHANNEL_PIECE_TYPE].set(ROOK)
    board = board.at[3:11, 0, CHANNEL_OWNER].set(GREEN)
    
    # Pawns (col 1)
    board = board.at[3:11, 1, CHANNEL_PIECE_TYPE].set(PAWN)
    board = board.at[3:11, 1, CHANNEL_OWNER].set(GREEN)
    
    return board


def get_initial_king_positions() -> chex.Array:
    """
    Get initial positions of all four kings.
    
    Returns:
        Array of shape (4, 2) with [row, col] for each king
    """
    return jnp.array([
        [13, 7],  # Red king
        [7, 13],  # Blue king
        [0, 6],   # Yellow king
        [6, 0],   # Green king
    ], dtype=jnp.int32)


def square_to_index(row: int, col: int, valid_mask: chex.Array) -> int:
    """
    Convert board coordinates to a flat index (0-159).
    
    Only valid for squares that are on the board.
    
    Args:
        row: Row coordinate (0-13)
        col: Column coordinate (0-13)
        valid_mask: Valid square mask
    
    Returns:
        Flat index (0-159) for the square
    """
    # Count valid squares before this position
    mask_flat = valid_mask.flatten()
    board_flat_idx = row * BOARD_SIZE + col
    # Count how many valid squares come before this index
    return int(jnp.sum(mask_flat[:board_flat_idx] * mask_flat[board_flat_idx]))


def index_to_square(idx: int, valid_mask: chex.Array) -> Tuple[int, int]:
    """
    Convert flat index (0-159) to board coordinates.
    
    Args:
        idx: Flat index (0-159)
        valid_mask: Valid square mask
    
    Returns:
        Tuple of (row, col) coordinates
    """
    # Find the idx-th valid square
    mask_flat = valid_mask.flatten()
    valid_indices = jnp.where(mask_flat)[0]
    flat_idx = valid_indices[idx]
    row = flat_idx // BOARD_SIZE
    col = flat_idx % BOARD_SIZE
    return int(row), int(col)


def get_piece_at(board: chex.Array, row: chex.Numeric, col: chex.Numeric) -> Tuple[chex.Numeric, chex.Numeric]:
    """
    Get the piece type and owner at a given square.
    
    Args:
        board: Board state array (14, 14, 4)
        row: Row coordinate
        col: Column coordinate
    
    Returns:
        Tuple of (piece_type, owner)
    """
    piece_type = board[row, col, CHANNEL_PIECE_TYPE]
    owner = board[row, col, CHANNEL_OWNER]
    return piece_type, owner


def set_piece_at(
    board: chex.Array, 
    row: chex.Numeric, 
    col: chex.Numeric,
    piece_type: chex.Numeric,
    owner: chex.Numeric,
    has_moved: bool = False
) -> chex.Array:
    """
    Set a piece at a given square.
    
    Args:
        board: Board state array (14, 14, 4)
        row: Row coordinate
        col: Column coordinate
        piece_type: Type of piece to place
        owner: Owner of the piece
        has_moved: Whether the piece has moved
    
    Returns:
        Updated board array
    """
    board = board.at[row, col, CHANNEL_PIECE_TYPE].set(piece_type)
    board = board.at[row, col, CHANNEL_OWNER].set(owner)
    board = board.at[row, col, CHANNEL_HAS_MOVED].set(int(has_moved))
    return board


def clear_square(board: chex.Array, row: chex.Numeric, col: chex.Numeric) -> chex.Array:
    """
    Clear a square (set to empty).
    
    Args:
        board: Board state array (14, 14, 4)
        row: Row coordinate
        col: Column coordinate
    
    Returns:
        Updated board array
    """
    board = board.at[row, col, CHANNEL_PIECE_TYPE].set(EMPTY)
    board = board.at[row, col, CHANNEL_OWNER].set(0)
    board = board.at[row, col, CHANNEL_HAS_MOVED].set(0)
    return board


@jax.jit
def count_pieces(board: chex.Array, player: chex.Numeric) -> chex.Numeric:
    """
    Count the number of pieces a player has on the board.
    
    Args:
        board: Board state array (14, 14, 4)
        player: Player ID
    
    Returns:
        Number of pieces
    """
    owner_channel = board[:, :, CHANNEL_OWNER]
    piece_channel = board[:, :, CHANNEL_PIECE_TYPE]
    # Count non-empty squares owned by player
    return jnp.sum((owner_channel == player) & (piece_channel != EMPTY))


@jax.jit
def find_king_position(board: chex.Array, player: chex.Numeric) -> chex.Array:
    """
    Find the position of a player's king.
    
    Args:
        board: Board state array (14, 14, 4)
        player: Player ID
    
    Returns:
        Array [row, col] of king position, or [-1, -1] if not found
    """
    is_king = (board[:, :, CHANNEL_PIECE_TYPE] == KING) & (board[:, :, CHANNEL_OWNER] == player)
    positions = jnp.argwhere(is_king, size=1, fill_value=-1)
    return jnp.where(
        positions[0, 0] == -1,
        jnp.array([-1, -1], dtype=jnp.int32),
        positions[0]
    )