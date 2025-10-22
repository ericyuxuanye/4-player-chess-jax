"""
Game rules implementation: check detection, legal moves, game end conditions.

This is a simplified implementation focused on core mechanics.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from four_player_chess.constants import (
    BOARD_SIZE, EMPTY, KING, PAWN, BISHOP, ROOK, QUEEN,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER,
    NUM_PLAYERS, INACTIVE_PLAYER
)
from four_player_chess.pieces import get_pseudo_legal_moves, can_piece_attack_square
from four_player_chess.precompute import (
    LEGAL_DEST_NEAR_4P, LEGAL_DEST_FAR_4P, BETWEEN_4P, CAN_MOVE_4P, COORD_TO_IDX
)


def is_square_attacked(
    board: chex.Array,
    target_row: chex.Numeric,
    target_col: chex.Numeric,
    by_player: chex.Numeric,
    valid_mask: chex.Array,
) -> chex.Array:
    """
    Check if a square is attacked by any piece of a given player.
    
    OPTIMIZED: Uses sparse iteration (only occupied squares) and precomputed tables.
    
    Args:
        board: Board state
        target_row: Row of square to check
        target_col: Column of square to check
        by_player: Player ID whose pieces we check
        valid_mask: Valid square mask
    
    Returns:
        Boolean indicating if square is under attack
    """
    # Convert target position to valid square index (not flat board index!)
    # The precompute tables use valid square indices (0-159), not flat board indices
    target_idx = COORD_TO_IDX[target_row, target_col]
    
    # Get all pieces belonging to the attacking player
    piece_owners = board[:, :, CHANNEL_OWNER]
    piece_types = board[:, :, CHANNEL_PIECE_TYPE]
    
    # Find occupied squares belonging to this player (sparse iteration)
    belongs_to_player = (piece_owners == by_player) & (piece_types != EMPTY)
    
    # Use jnp.nonzero with size parameter for JIT compilation
    # Estimate: max 16 pieces per player (typical game has 12-16)
    flat_board = belongs_to_player.flatten()
    attacker_flat_positions = jnp.nonzero(flat_board, size=16, fill_value=-1)[0]
    
    def check_attacker(attacker_flat_pos):
        """Check if a single attacker can reach the target."""
        # Skip invalid positions (from padding)
        is_valid_attacker = attacker_flat_pos >= 0
        
        # Get piece type at attacker position (using flat board index)
        attacker_row = attacker_flat_pos // BOARD_SIZE
        attacker_col = attacker_flat_pos % BOARD_SIZE
        piece_type = board[attacker_row, attacker_col, CHANNEL_PIECE_TYPE]
        
        # Convert attacker position to valid square index for precompute table lookup
        attacker_idx = COORD_TO_IDX[attacker_row, attacker_col]
        
        # Check if geometrically possible to attack target
        # Use valid square indices (0-159) for precompute table access
        can_reach = CAN_MOVE_4P[piece_type, attacker_idx, target_idx]
        
        # For far moves (sliding pieces), check if path is clear
        between_squares = BETWEEN_4P[attacker_idx, target_idx]
        
        def check_path(_):
            """Check if path is clear for sliding pieces."""
            # Extract piece types along path
            between_rows = between_squares // BOARD_SIZE
            between_cols = between_squares % BOARD_SIZE
            
            # Check each square in path
            def is_square_empty(sq_idx):
                sq = between_squares[sq_idx]
                # -1 indicates no more squares in path
                is_path_square = sq >= 0
                br = between_rows[sq_idx]
                bc = between_cols[sq_idx]
                is_empty = board[br, bc, CHANNEL_PIECE_TYPE] == EMPTY
                return (~is_path_square) | is_empty
            
            # All squares in path must be empty
            path_checks = jax.vmap(is_square_empty)(jnp.arange(12))
            return jnp.all(path_checks)
        
        def skip_path(_):
            return jnp.bool_(True)
        
        # Only check path for sliding pieces (not knights/kings)
        is_sliding = jnp.isin(piece_type, jnp.array([BISHOP, ROOK, QUEEN]))
        path_clear = jax.lax.cond(is_sliding, check_path, skip_path, None)
        
        # Can attack if: valid attacker, can reach, and path is clear
        return is_valid_attacker & can_reach & path_clear
    
    # Check all potential attackers
    attacks = jax.vmap(check_attacker)(attacker_flat_positions)
    
    return jnp.any(attacks)


def is_in_check(
    board: chex.Array,
    king_position: chex.Array,
    player: chex.Numeric,
    player_active: chex.Array,
    valid_mask: chex.Array
) -> chex.Array:
    """
    Check if a player's king is in check.
    
    Args:
        board: Board state
        king_position: Position of the king [row, col]
        player: Player ID
        player_active: Array of active players
        valid_mask: Valid square mask
    
    Returns:
        Boolean indicating if king is in check
    """
    king_row, king_col = king_position[0], king_position[1]
    
    # If king position is invalid, player is eliminated
    king_invalid = (king_row < 0) | (king_col < 0)
    
    # Check if any opponent's piece attacks the king
    attacks = jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_)
    for opponent in range(NUM_PLAYERS):
        is_opponent = opponent != player
        is_active = player_active[opponent]
        should_check = is_opponent & is_active
        
        attacked = is_square_attacked(board, king_row, king_col, opponent, valid_mask)
        attacks = attacks.at[opponent].set(should_check & attacked)
    
    in_check = jnp.any(attacks)
    return jnp.where(king_invalid, jnp.bool_(False), in_check)


@jax.jit
def get_all_legal_moves_for_player(
    board: chex.Array,
    player: chex.Numeric,
    king_position: chex.Array,
    player_active: chex.Array,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    JIT-friendly, vectorized version to get all legal moves for a player.

    Returns boolean array shape (BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE).
    """
    dest_idxs = jnp.arange(BOARD_SIZE, dtype=jnp.int32)
    src_idxs = jnp.arange(BOARD_SIZE, dtype=jnp.int32)

    def legal_moves_for_source(sr: jnp.int32, sc: jnp.int32):
        # Scalars captured from vmapped args
        piece_owner = board[sr, sc, CHANNEL_OWNER]
        piece_type = board[sr, sc, CHANNEL_PIECE_TYPE]
        is_valid_square = valid_mask[sr, sc] > 0
        is_my_piece = (piece_owner == player) & (piece_type != EMPTY) & is_valid_square

        pseudo_legal = get_pseudo_legal_moves(
            board, sr, sc, player, valid_mask, en_passant_square
        )

        def check_dest(dr: jnp.int32, dc: jnp.int32):
            is_pseudo = pseudo_legal[dr, dc]

            def true_fn(_):
                # build test board with functional updates
                tb = board
                tb = tb.at[dr, dc, CHANNEL_PIECE_TYPE].set(piece_type)
                tb = tb.at[dr, dc, CHANNEL_OWNER].set(player)
                tb = tb.at[sr, sc, CHANNEL_PIECE_TYPE].set(EMPTY)
                tb = tb.at[sr, sc, CHANNEL_OWNER].set(0)

                test_king_pos = jnp.where(
                    piece_type == KING,
                    jnp.array([dr, dc], dtype=jnp.int32),
                    king_position
                )

                in_check_after = is_in_check(tb, test_king_pos, player, player_active, valid_mask)
                return is_my_piece & is_pseudo & (~in_check_after)

            false_fn = lambda _: jnp.bool_(False)

            return jax.lax.cond(is_my_piece & is_pseudo, true_fn, false_fn, operand=None)

        # vectorize over dest rows and cols to produce a (BOARD_SIZE, BOARD_SIZE) matrix
        row_fn = lambda dr: jax.vmap(lambda dc: check_dest(dr, dc))(dest_idxs)
        return jax.vmap(row_fn)(dest_idxs)

    # vectorize over source rows and cols to produce full (sr,sc,dr,dc) array
    src_row_fn = lambda sr: jax.vmap(lambda sc: legal_moves_for_source(sr, sc))(src_idxs)
    legal_moves = jax.vmap(src_row_fn)(src_idxs)

    return legal_moves


def has_legal_moves(
    board: chex.Array,
    player: chex.Numeric,
    king_position: chex.Array,
    player_active: chex.Array,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    Check if a player has any legal moves.
    
    Returns:
        Boolean indicating if player has at least one legal move
    """
    legal_moves = get_all_legal_moves_for_player(
        board, player, king_position, player_active, valid_mask, en_passant_square
    )
    return jnp.any(legal_moves)


def is_checkmate(
    board: chex.Array,
    player: chex.Numeric,
    king_position: chex.Array,
    player_active: chex.Array,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    Check if a player is checkmated.
    
    Checkmate occurs when:
    1. King is in check
    2. Player has no legal moves
    
    Returns:
        Boolean indicating checkmate
    """
    in_check = is_in_check(board, king_position, player, player_active, valid_mask)
    has_moves = has_legal_moves(board, player, king_position, player_active, valid_mask, en_passant_square)
    
    # Checkmate = in check AND no legal moves
    return in_check & (~has_moves)


def is_stalemate(
    board: chex.Array,
    player: chex.Numeric,
    king_position: chex.Array,
    player_active: chex.Array,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    Check if a player is stalemated.
    
    Stalemate occurs when:
    1. King is NOT in check
    2. Player has no legal moves
    
    Returns:
        Boolean indicating stalemate
    """
    in_check = is_in_check(board, king_position, player, player_active, valid_mask)
    has_moves = has_legal_moves(board, player, king_position, player_active, valid_mask, en_passant_square)
    
    # Stalemate = NOT in check AND no legal moves
    return (~in_check) & (~has_moves)


def is_move_legal(
    board: chex.Array,
    source_row: chex.Numeric,
    source_col: chex.Numeric,
    dest_row: chex.Numeric,
    dest_col: chex.Numeric,
    player: chex.Numeric,
    king_position: chex.Array,
    player_active: chex.Array,
    valid_mask: chex.Array,
    en_passant_square: chex.Array
) -> chex.Array:
    """
    Check if a specific move is legal.
    
    Returns:
        Boolean indicating if the move is legal
    """
    # Check piece ownership
    piece_owner = board[source_row, source_col, CHANNEL_OWNER]
    piece_type = board[source_row, source_col, CHANNEL_PIECE_TYPE]
    
    # Use jnp.where instead of if statements for JAX compatibility
    wrong_owner = piece_owner != player
    is_empty = piece_type == EMPTY
    invalid_piece = wrong_owner | is_empty
    
    # Check if destination is valid
    # Note: valid_mask contains int32 values (0 or 1), so convert to bool before negating
    invalid_dest = valid_mask[dest_row, dest_col] == 0
    
    # Get pseudo-legal moves
    pseudo_legal = get_pseudo_legal_moves(
        board, source_row, source_col, player, valid_mask, en_passant_square
    )
    
    not_pseudo_legal = ~pseudo_legal[dest_row, dest_col]
    
    # Early return using jnp.where
    # If any basic checks fail, return False
    basic_checks_fail = invalid_piece | invalid_dest | not_pseudo_legal
    
    # Simulate move and check for check
    test_board = board.copy()
    test_board = test_board.at[dest_row, dest_col, CHANNEL_PIECE_TYPE].set(piece_type)
    test_board = test_board.at[dest_row, dest_col, CHANNEL_OWNER].set(player)
    test_board = test_board.at[source_row, source_col, CHANNEL_PIECE_TYPE].set(EMPTY)
    test_board = test_board.at[source_row, source_col, CHANNEL_OWNER].set(0)
    
    # Update king position if moved
    test_king_pos = jnp.where(
        piece_type == KING,
        jnp.array([dest_row, dest_col], dtype=jnp.int32),
        king_position
    )
    
    # Check if king is in check after move
    in_check_after = is_in_check(test_board, test_king_pos, player, player_active, valid_mask)
    
    # Return False if basic checks fail, otherwise return whether king is not in check
    return jnp.where(basic_checks_fail, False, ~in_check_after)