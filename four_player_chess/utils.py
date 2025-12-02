"""
Utility functions for 4-player chess environment.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from four_player_chess.constants import NUM_PLAYERS, INACTIVE_PLAYER
import four_player_chess.coordinate_transforms as coordinate_transforms


@jax.jit
def get_next_active_player(current_player: chex.Numeric, player_active: chex.Array) -> chex.Numeric:
    """
    Find the next active player in clockwise order.
    
    Args:
        current_player: Current player ID (0-3)
        player_active: Boolean array (4,) indicating which players are active
    
    Returns:
        Next active player ID, or current_player if no active players remain
    """
    # JAX-compatible: vectorize the search and pick the first active candidate
    current_player = jnp.asarray(current_player, dtype=jnp.int32)
    offsets = jnp.arange(1, NUM_PLAYERS + 1, dtype=jnp.int32)
    candidates = (current_player + offsets) % jnp.asarray(NUM_PLAYERS, dtype=jnp.int32)

    # boolean mask for which candidates are active
    active_mask = player_active[candidates].astype(jnp.bool_)
    has_active = jnp.any(active_mask)

    # index of the first True in active_mask (returns 0 if none True)
    first_rel = jnp.argmax(active_mask)

    # choose the first active candidate if any, otherwise return current_player
    next_player = jnp.where(has_active, candidates[first_rel], current_player)
    return next_player.astype(jnp.int32)


@jax.jit
def count_active_players(player_active: chex.Array) -> chex.Numeric:
    """Count how many players are still active."""
    return jnp.sum(player_active)


def _encode_action_no_jit(
    source_row: chex.Numeric,
    source_col: chex.Numeric,
    dest_row: chex.Numeric,
    dest_col: chex.Numeric,
    promotion_type: chex.Numeric,
    valid_mask: chex.Array
) -> chex.Numeric:
    """Internal implementation of encode_action for use without JIT (calls JIT version)."""
    # Just call the JIT-compatible version
    return encode_action(
        source_row, source_col, dest_row, dest_col, promotion_type, valid_mask
    )


@jax.jit
def encode_action(
    source_row: chex.Numeric,
    source_col: chex.Numeric,
    dest_row: chex.Numeric,
    dest_col: chex.Numeric,
    promotion_type: chex.Numeric,
    valid_mask: chex.Array
) -> chex.Numeric:
    """
    Encode a move as a single integer action.

    Action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type

    Args:
        source_row: Source square row
        source_col: Source square column
        dest_row: Destination square row
        dest_col: Destination square column
        promotion_type: Promotion piece type (0-3)
        valid_mask: Valid square mask

    Returns:
        Integer action ID
    """
    # Convert board coordinates to flat indices (0-159)
    mask_flat = valid_mask.flatten().astype(jnp.int32)

    # Create index array for masking
    indices = jnp.arange(len(mask_flat), dtype=jnp.int32)

    # Source index: count valid squares before source position
    source_flat = source_row * 14 + source_col
    source_idx = jnp.sum(mask_flat, where=indices < source_flat) * mask_flat[source_flat]

    # Dest index: count valid squares before dest position
    dest_flat = dest_row * 14 + dest_col
    dest_idx = jnp.sum(mask_flat, where=indices < dest_flat) * mask_flat[dest_flat]

    # Encode action
    action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
    return action.astype(jnp.int32)


@jax.jit
def decode_action(action: chex.Numeric, valid_mask: chex.Array) -> Tuple[chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric]:
    """
    Decode an integer action into move components.
    
    Returns:
        Tuple of (source_row, source_col, dest_row, dest_col, promotion_type)
    """
    # Extract components
    promotion_type = action % 4
    temp = action // 4
    dest_idx = temp % 160
    source_idx = temp // 160
    
    # Convert flat indices to board coordinates
    # Use size=160 since we know there are exactly 160 valid squares
    mask_flat = valid_mask.flatten()
    valid_indices = jnp.where(mask_flat, size=160, fill_value=-1)[0]
    
    source_flat_idx = valid_indices[source_idx]
    source_row = source_flat_idx // 14
    source_col = source_flat_idx % 14
    
    dest_flat_idx = valid_indices[dest_idx]
    dest_row = dest_flat_idx // 14
    dest_col = dest_flat_idx % 14
    
    return source_row, source_col, dest_row, dest_col, promotion_type


def encode_action_relative(
    rel_src_row: chex.Numeric,
    rel_src_col: chex.Numeric,
    rel_dst_row: chex.Numeric,
    rel_dst_col: chex.Numeric,
    promotion_type: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array
) -> chex.Numeric:
    """
    Encode action using player-relative coordinates.

    Transforms relative coordinates to absolute coordinates based on the player's
    perspective, then encodes the action using the standard encoding.

    Args:
        rel_src_row: Source row in player's perspective
        rel_src_col: Source column in player's perspective
        rel_dst_row: Destination row in player's perspective
        rel_dst_col: Destination column in player's perspective
        promotion_type: Promotion piece type (0-3)
        player: Player ID (0=Red, 1=Blue, 2=Yellow, 3=Green)
        valid_mask: Valid square mask

    Returns:
        Integer action ID
    """
    # Transform source coordinates
    abs_src_row, abs_src_col = coordinate_transforms.relative_to_absolute_coords(
        rel_src_row, rel_src_col, player, valid_mask
    )

    # Transform destination coordinates
    abs_dst_row, abs_dst_col = coordinate_transforms.relative_to_absolute_coords(
        rel_dst_row, rel_dst_col, player, valid_mask
    )

    # Use non-JIT implementation to avoid tracing issues
    return _encode_action_no_jit(
        abs_src_row, abs_src_col, abs_dst_row, abs_dst_col,
        promotion_type, valid_mask
    )


def decode_action_relative(
    action: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array
) -> Tuple[chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric]:
    """
    Decode action to player-relative coordinates.

    Decodes the action to absolute coordinates first, then transforms to
    coordinates from the player's perspective.

    Args:
        action: Integer action ID
        player: Player ID (0=Red, 1=Blue, 2=Yellow, 3=Green)
        valid_mask: Valid square mask

    Returns:
        Tuple of (rel_src_row, rel_src_col, rel_dst_row, rel_dst_col, promotion_type)
    """
    # Decode to absolute coordinates
    abs_src_row, abs_src_col, abs_dst_row, abs_dst_col, promo = decode_action(
        action, valid_mask
    )

    # Transform to relative coordinates
    rel_src_row, rel_src_col = coordinate_transforms.absolute_to_relative_coords(
        abs_src_row, abs_src_col, player, valid_mask
    )
    rel_dst_row, rel_dst_col = coordinate_transforms.absolute_to_relative_coords(
        abs_dst_row, abs_dst_col, player, valid_mask
    )

    return rel_src_row, rel_src_col, rel_dst_row, rel_dst_col, promo


@jax.jit
def is_game_over(player_active: chex.Array, move_count: chex.Numeric, max_moves: int) -> chex.Array:
    """
    Check if the game is over.

    Game ends when:
    - Only 0 or 1 players remain active
    - Maximum move limit reached

    Returns:
        Boolean indicating if game is over
    """
    num_active = count_active_players(player_active)
    too_many_moves = move_count >= max_moves
    not_enough_players = num_active <= 1

    return too_many_moves | not_enough_players

def dict_to_jax_array(d: dict[int, int]) -> chex.Array:
    """
    Convert a dictionary of int mappings (0-n) to a JAX array.
    """
    max_key = max(d.keys())
    temp_list = [d.get(i, -1) for i in range(max_key + 1)]
    return jnp.array(temp_list, dtype=jnp.int32)