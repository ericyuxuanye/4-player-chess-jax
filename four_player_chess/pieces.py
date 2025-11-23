"""
Piece movement logic for 4-player chess.

All functions are pure and JIT-compatible.
"""

import jax
import jax.numpy as jnp
import chex

from four_player_chess.constants import (
    BOARD_SIZE, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    RED, BLUE, YELLOW, GREEN,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    KNIGHT_OFFSETS, KING_OFFSETS, BISHOP_DIRECTIONS, ROOK_DIRECTIONS, QUEEN_DIRECTIONS,
    PAWN_FORWARD, PAWN_DOUBLE_FORWARD, PAWN_CAPTURES,
    PAWN_START_RANKS, PROMOTION_RANKS,
    QUEENSIDE_CASTLING, KINGSIDE_CASTLING,
    INITIAL_KING_POSITIONS, INITIAL_ROOK_POSITIONS,
    CASTLING_KING_DESTINATIONS, CASTLING_ROOK_DESTINATIONS
)
from four_player_chess.utils import dict_to_jax_array


@jax.jit
def get_pawn_moves(
    board: chex.Array,
    row: chex.Numeric,
    col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    Get all legal pawn moves from a given position.
    
    Returns:
        Boolean array of shape (14, 14) indicating legal destination squares
    """
    moves = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    
    # Get pawn direction based on player
    forward = jnp.array([
        PAWN_FORWARD[RED],
        PAWN_FORWARD[BLUE],
        PAWN_FORWARD[YELLOW],
        PAWN_FORWARD[GREEN]
    ])[player]
    
    # Single forward move
    new_row = row + forward[..., 0]
    new_col = col + forward[..., 1]
    in_bounds = (new_row >= 0) & (new_row < BOARD_SIZE) & (new_col >= 0) & (new_col < BOARD_SIZE)
    is_valid = jnp.where(in_bounds, valid_mask[new_row, new_col] > 0, False)
    is_empty = jnp.where(in_bounds, board[new_row, new_col, CHANNEL_PIECE_TYPE] == EMPTY, False)
    moves = jnp.where(is_valid & is_empty, moves.at[new_row, new_col].set(True), moves)
    
    # Double forward move (from starting position)
    has_moved = board[row, col, CHANNEL_HAS_MOVED] > 0
    
    # Check if on starting rank (depends on player orientation)
    on_start_rank = jnp.where(
        (player == RED) | (player == YELLOW),
        row == dict_to_jax_array(PAWN_START_RANKS)[player],
        col == dict_to_jax_array(PAWN_START_RANKS)[player],
    )
    
    can_double = (~has_moved) & on_start_rank & is_empty  # Can only double if single move is empty
    
    double_forward = jnp.array([
        PAWN_DOUBLE_FORWARD[RED],
        PAWN_DOUBLE_FORWARD[BLUE],
        PAWN_DOUBLE_FORWARD[YELLOW],
        PAWN_DOUBLE_FORWARD[GREEN]
    ])[player]
    
    double_row = row + double_forward[..., 0]
    double_col = col + double_forward[..., 1]
    double_in_bounds = (double_row >= 0) & (double_row < BOARD_SIZE) & (double_col >= 0) & (double_col < BOARD_SIZE)
    double_is_valid = jnp.where(double_in_bounds, valid_mask[double_row, double_col] > 0, False)
    double_is_empty = jnp.where(double_in_bounds, board[double_row, double_col, CHANNEL_PIECE_TYPE] == EMPTY, False)
    
    moves = jnp.where(
        can_double & double_is_valid & double_is_empty,
        moves.at[double_row, double_col].set(True),
        moves
    )
    
    # Capture moves (diagonal)
    captures = jnp.array([
        PAWN_CAPTURES[RED],
        PAWN_CAPTURES[BLUE],
        PAWN_CAPTURES[YELLOW],
        PAWN_CAPTURES[GREEN]
    ])[player]
    
    for i in range(2):  # Two diagonal captures
        cap_row = row + captures[..., i, 0]
        cap_col = col + captures[..., i, 1]
        cap_in_bounds = (cap_row >= 0) & (cap_row < BOARD_SIZE) & (cap_col >= 0) & (cap_col < BOARD_SIZE)
        cap_is_valid = jnp.where(cap_in_bounds, valid_mask[cap_row, cap_col] > 0, False)
        
        # Can capture opponent piece or en passant
        has_opponent = jnp.where(
            cap_in_bounds,
            (board[cap_row, cap_col, CHANNEL_PIECE_TYPE] != EMPTY) &
            (board[cap_row, cap_col, CHANNEL_OWNER] != player),
            False
        )
        
        is_en_passant = (cap_row == en_passant_square[0]) & (cap_col == en_passant_square[1])
        
        can_capture = cap_is_valid & (has_opponent | is_en_passant)
        moves = jnp.where(can_capture, moves.at[cap_row, cap_col].set(True), moves)
    
    return moves


@jax.jit
def get_knight_moves(
    board: chex.Array,
    row: chex.Numeric,
    col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array
) -> chex.Array:
    """
    Get all legal knight moves from a given position.
    
    Returns:
        Boolean array of shape (14, 14) indicating legal destination squares
    """
    moves = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    
    for i in range(8):  # 8 possible L-shaped moves
        new_row = row + KNIGHT_OFFSETS[i, 0]
        new_col = col + KNIGHT_OFFSETS[i, 1]
        
        in_bounds = (new_row >= 0) & (new_row < BOARD_SIZE) & (new_col >= 0) & (new_col < BOARD_SIZE)
        is_valid = jnp.where(in_bounds, valid_mask[new_row, new_col] > 0, False)
        
        # Can move to empty square or capture opponent
        # Note: Empty squares have owner=0, which conflicts with Red player (also 0)
        # So we need to check piece type as well
        target_piece = jnp.where(in_bounds, board[new_row, new_col, CHANNEL_PIECE_TYPE], EMPTY + 1)
        target_owner = jnp.where(in_bounds, board[new_row, new_col, CHANNEL_OWNER], player)
        is_empty_or_opponent = (target_piece == EMPTY) | (target_owner != player)
        can_move = is_valid & is_empty_or_opponent
        
        moves = jnp.where(can_move, moves.at[new_row, new_col].set(True), moves)
    
    return moves


@jax.jit
def get_sliding_moves(
    board: chex.Array,
    row: chex.Numeric,
    col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array,
    directions: chex.Array
) -> chex.Array:
    moves = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)

    # Prepare arrays
    dirs = jnp.asarray(directions, dtype=jnp.int32)  # (n_dirs, 2)
    n_dirs = dirs.shape[0]
    max_dist = BOARD_SIZE - 1
    dists = jnp.arange(1, BOARD_SIZE, dtype=jnp.int32)  # (max_dist,)

    r = jnp.int32(row)
    c = jnp.int32(col)
    p = jnp.int32(player)
    
    # Add dimensions to support both scalar and batched (vmapped) inputs
    # If r/c/p are batched, shape will be (..., 1, 1) to broadcast with (n_dirs, max_dist)
    r = r[..., None, None] if r.ndim > 0 else r
    c = c[..., None, None] if c.ndim > 0 else c
    p = p[..., None, None] if p.ndim > 0 else p
    
    drs = dirs[:, 0][:, None]  # (n_dirs, 1)
    dcs = dirs[:, 1][:, None]  # (n_dirs, 1)

    # Compute all target squares for each direction and distance: shape (..., n_dirs, max_dist)
    new_rows = r + drs * dists[None, :]
    new_cols = c + dcs * dists[None, :]

    # Clamp indices for safe advanced indexing, mask out-of-bounds later
    clamped_rows = jnp.clip(new_rows, 0, BOARD_SIZE - 1)
    clamped_cols = jnp.clip(new_cols, 0, BOARD_SIZE - 1)

    in_bounds = (new_rows >= 0) & (new_rows < BOARD_SIZE) & (new_cols >= 0) & (new_cols < BOARD_SIZE)

    # Gather board info at all candidate squares
    target_piece = board[clamped_rows, clamped_cols, CHANNEL_PIECE_TYPE]  # (n_dirs, max_dist)
    target_owner = board[clamped_rows, clamped_cols, CHANNEL_OWNER]      # (n_dirs, max_dist)

    is_valid = jnp.where(in_bounds, valid_mask[clamped_rows, clamped_cols] > 0, False)
    is_empty = (target_piece == EMPTY) & in_bounds & is_valid
    is_opponent = (target_piece != EMPTY) & (target_owner != p) & in_bounds & is_valid

    # Determine if path to each distance is clear (all previous squares empty)
    empties_int = is_empty.astype(jnp.int32)
    cum_empties = jnp.cumprod(empties_int, axis=-1)  # includes current square, last axis is distances
    # Exclusive prefix: for distance k, need emptiness of distances < k
    # Need to match the shape of cum_empties for concatenation
    ones_shape = list(cum_empties.shape)
    ones_shape[-1] = 1  # Last dimension becomes 1
    exclusive_prefix = jnp.concatenate(
        [jnp.ones(ones_shape, dtype=jnp.int32), cum_empties[..., :-1]], axis=-1
    ).astype(jnp.bool_)
    path_clear = exclusive_prefix

    # Can move to a square if path is clear and square is valid and either empty or first opponent piece
    can_move_here = path_clear & is_valid & (is_empty | is_opponent)

    # Scatter results into moves board
    # Use .max() instead of .set() to handle duplicate indices correctly.
    # When out-of-bounds moves are clamped to board edges, they create duplicates.
    # .max() will OR the boolean values together, while .set() would use only the last value.
    flat_rows = clamped_rows.flatten()
    flat_cols = clamped_cols.flatten()
    flat_mask = can_move_here.flatten()

    moves = moves.at[(flat_rows, flat_cols)].max(flat_mask)

    return moves


@jax.jit
def get_king_moves(
    board: chex.Array,
    row: chex.Numeric,
    col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array,
    castling_rights: chex.Array = None
) -> chex.Array:
    """
    Get all legal king moves from a given position, including castling.

    Note: This does not check if moves leave king in check - that's done separately.

    Returns:
        Boolean array of shape (14, 14) indicating legal destination squares
    """
    moves = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)

    # Regular king moves: 8 adjacent squares
    for i in range(8):
        new_row = row + KING_OFFSETS[i, 0]
        new_col = col + KING_OFFSETS[i, 1]

        in_bounds = (new_row >= 0) & (new_row < BOARD_SIZE) & (new_col >= 0) & (new_col < BOARD_SIZE)
        is_valid = jnp.where(in_bounds, valid_mask[new_row, new_col] > 0, False)

        # Can move to empty square or capture opponent
        # Note: Empty squares have owner=0, which conflicts with Red player (also 0)
        # So we need to check piece type as well
        target_piece = jnp.where(in_bounds, board[new_row, new_col, CHANNEL_PIECE_TYPE], EMPTY + 1)
        target_owner = jnp.where(in_bounds, board[new_row, new_col, CHANNEL_OWNER], player)
        is_empty_or_opponent = (target_piece == EMPTY) | (target_owner != player)
        can_move = is_valid & is_empty_or_opponent

        moves = jnp.where(can_move, moves.at[new_row, new_col].set(True), moves)

    # Castling moves
    if castling_rights is not None:
        # Check if king is on its initial square
        initial_king_pos = dict_to_jax_array(INITIAL_KING_POSITIONS)[player]
        on_initial_square = (row == initial_king_pos[0]) & (col == initial_king_pos[1])

        # Check if king has moved
        king_has_moved = board[row, col, CHANNEL_HAS_MOVED] > 0
        can_castle = on_initial_square & (~king_has_moved)

        # Get castling info for this player
        rook_positions = dict_to_jax_array(INITIAL_ROOK_POSITIONS)[player]  # (2, 2) array
        king_destinations = dict_to_jax_array(CASTLING_KING_DESTINATIONS)[player]  # (2, 2) array

        # Check both queenside and kingside castling
        for side in [QUEENSIDE_CASTLING, KINGSIDE_CASTLING]:
            # Check if we have castling rights for this side
            has_castling_right = castling_rights[player, side]

            # Get rook position for this side
            rook_row = rook_positions[side, 0]
            rook_col = rook_positions[side, 1]

            # Check if rook is still there and hasn't moved
            rook_present = board[rook_row, rook_col, CHANNEL_PIECE_TYPE] == ROOK
            rook_is_ours = board[rook_row, rook_col, CHANNEL_OWNER] == player
            rook_not_moved = board[rook_row, rook_col, CHANNEL_HAS_MOVED] == 0
            rook_ok = rook_present & rook_is_ours & rook_not_moved

            # Check if squares between king and rook are empty
            # Need to check different squares based on player orientation and side
            path_clear = check_castling_path(board, row, col, rook_row, rook_col, player, side)

            # If all conditions met, add castling destination
            can_castle_this_side = can_castle & has_castling_right & rook_ok & path_clear
            king_dest_row = king_destinations[side, 0]
            king_dest_col = king_destinations[side, 1]

            moves = jnp.where(
                can_castle_this_side,
                moves.at[king_dest_row, king_dest_col].set(True),
                moves
            )

    return moves


def check_castling_path(
    board: chex.Array,
    king_row: chex.Numeric,
    king_col: chex.Numeric,
    rook_row: chex.Numeric,
    rook_col: chex.Numeric,
    player: chex.Numeric,
    side: chex.Numeric
) -> chex.Array:
    """
    Check if the path between king and rook is clear for castling.

    Returns:
        Boolean indicating if path is clear
    """
    # For Red and Yellow (horizontal castling), check columns between king and rook
    # For Blue and Green (vertical castling), check rows between king and rook
    is_horizontal = (player == RED) | (player == YELLOW)

    # Get the range of squares to check based on side and player
    # For queenside: check squares between rook and king (excluding both)
    # For kingside: check squares between king and rook (excluding both)

    def check_horizontal():
        # For horizontal players, king and rook are on same row
        # Queenside: rook_col < king_col, check cols between
        # Kingside: king_col < rook_col, check cols between
        is_queenside = side == QUEENSIDE_CASTLING

        # Queenside checks columns between rook and king (3 squares)
        # Kingside checks columns between king and rook (2 squares)
        # Pad kingside to 3 squares with -1 (invalid)
        queenside_cols = jnp.array([rook_col + 1, rook_col + 2, rook_col + 3], dtype=jnp.int32)
        kingside_cols = jnp.array([king_col + 1, king_col + 2, -1], dtype=jnp.int32)

        cols_to_check = jnp.where(is_queenside, queenside_cols, kingside_cols)

        # Check if all squares are empty
        def check_col(col):
            # Invalid col (padding) is always True (doesn't block)
            is_valid_col = col >= 0
            is_empty = board[king_row, jnp.maximum(col, 0), CHANNEL_PIECE_TYPE] == EMPTY
            return (~is_valid_col) | is_empty

        checks = jax.vmap(check_col)(cols_to_check)
        return jnp.all(checks)

    def check_vertical():
        # For vertical players, king and rook are on same column
        # Queenside: rook_row < king_row, check rows between
        # Kingside: king_row < rook_row, check rows between
        is_queenside = side == QUEENSIDE_CASTLING

        # Queenside checks rows between rook and king (3 squares)
        # Kingside checks rows between king and rook (2 squares)
        # Pad kingside to 3 squares with -1 (invalid)
        queenside_rows = jnp.array([rook_row + 1, rook_row + 2, rook_row + 3], dtype=jnp.int32)
        kingside_rows = jnp.array([king_row + 1, king_row + 2, -1], dtype=jnp.int32)

        rows_to_check = jnp.where(is_queenside, queenside_rows, kingside_rows)

        # Check if all squares are empty
        def check_row(row):
            # Invalid row (padding) is always True (doesn't block)
            is_valid_row = row >= 0
            is_empty = board[jnp.maximum(row, 0), king_col, CHANNEL_PIECE_TYPE] == EMPTY
            return (~is_valid_row) | is_empty

        checks = jax.vmap(check_row)(rows_to_check)
        return jnp.all(checks)

    return jax.lax.cond(is_horizontal, check_horizontal, check_vertical)


def get_pseudo_legal_moves(
    board: chex.Array,
    row: chex.Numeric,
    col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array,
    en_passant_square: chex.Array,
    castling_rights: chex.Array = None
) -> chex.Array:
    """
    Get all pseudo-legal moves for a piece at a given position.

    "Pseudo-legal" means the moves follow piece movement rules but may leave
    the king in check. Full legality checking is done separately.

    Args:
        board: Board state array (14, 14, 4)
        row: Row of the piece
        col: Column of the piece
        player: Player who owns the piece
        valid_mask: Valid square mask
        en_passant_square: En passant target square
        castling_rights: Castling rights array (4, 2)

    Returns:
        Boolean array of shape (14, 14) indicating pseudo-legal destination squares
    """
    piece_type = board[row, col, CHANNEL_PIECE_TYPE]
    piece_owner = board[row, col, CHANNEL_OWNER]

    # Get moves based on piece type
    moves = jnp.where(
        piece_type == PAWN,
        get_pawn_moves(board, row, col, player, valid_mask, en_passant_square),
        jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    )

    moves = jnp.where(
        piece_type == KNIGHT,
        get_knight_moves(board, row, col, player, valid_mask),
        moves
    )

    moves = jnp.where(
        piece_type == BISHOP,
        get_sliding_moves(board, row, col, player, valid_mask, BISHOP_DIRECTIONS),
        moves
    )

    moves = jnp.where(
        piece_type == ROOK,
        get_sliding_moves(board, row, col, player, valid_mask, ROOK_DIRECTIONS),
        moves
    )

    moves = jnp.where(
        piece_type == QUEEN,
        get_sliding_moves(board, row, col, player, valid_mask, QUEEN_DIRECTIONS),
        moves
    )

    moves = jnp.where(
        piece_type == KING,
        get_king_moves(board, row, col, player, valid_mask, castling_rights),
        moves
    )

    # Return empty moves if not the player's piece
    not_my_piece = piece_owner != player
    return jnp.where(not_my_piece, jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_), moves)


def can_piece_attack_square(
    board: chex.Array,
    piece_row: chex.Numeric,
    piece_col: chex.Numeric,
    target_row: chex.Numeric,
    target_col: chex.Numeric,
    valid_mask: chex.Array
) -> chex.Array:
    """
    Check if a piece at (piece_row, piece_col) can attack (target_row, target_col).
    
    This is used for check detection.
    
    Returns:
        Boolean indicating if the piece can attack the target square
    """
    piece_type = board[piece_row, piece_col, CHANNEL_PIECE_TYPE]
    piece_owner = board[piece_row, piece_col, CHANNEL_OWNER]
    
    # Get pseudo-legal moves (using dummy en passant square)
    dummy_ep = jnp.array([-1, -1], dtype=jnp.int32)
    moves = get_pseudo_legal_moves(board, piece_row, piece_col, piece_owner, valid_mask, dummy_ep)
    
    # Empty square can't attack - use jnp.where instead of if
    result = moves[target_row, target_col]
    return jnp.where(piece_type == EMPTY, jnp.bool_(False), result)