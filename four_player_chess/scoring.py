"""
Scoring system for 4-player chess (Free-for-All variant).

Simplified version focusing on key scoring events.
"""

import jax
import jax.numpy as jnp
import chex

from four_player_chess.constants import (
    PIECE_VALUES, EMPTY,
    SCORE_CHECKMATE, SCORE_STALEMATE_SELF,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER
)
from four_player_chess.utils import dict_to_jax_array


@jax.jit
def calculate_capture_points(captured_piece_type: chex.Numeric, is_promoted_queen: bool = False) -> chex.Numeric:
    """
    Calculate points for capturing a piece.
    
    Args:
        captured_piece_type: Type of captured piece
        is_promoted_queen: If True and piece is queen, only worth 1 point
    
    Returns:
        Points awarded for the capture
    """
    base_value = dict_to_jax_array(PIECE_VALUES).at[captured_piece_type].get(mode="fill", fill_value=0)
    
    return jax.lax.cond(
        is_promoted_queen & (captured_piece_type == 5),  # QUEEN = 5
        lambda _: 1,
        lambda _: base_value,
        operand=None
    )


@jax.jit
def calculate_checkmate_score(num_remaining_players: chex.Numeric) -> chex.Numeric:
    """
    Calculate points for checkmating an opponent.
    
    Args:
        num_remaining_players: Number of players still active after checkmate
    
    Returns:
        Points for checkmate (base 20 points)
    """
    return SCORE_CHECKMATE


@jax.jit
def calculate_stalemate_score(is_self: bool, num_remaining_players: chex.Numeric) -> chex.Numeric:
    """
    Calculate points for stalemate.
    
    Args:
        is_self: If True, player stalemated themselves (gets 20 points)
        num_remaining_players: Number of players still active
    
    Returns:
        Points for stalemate
    """
    if is_self:
        return SCORE_STALEMATE_SELF
    else:
        # Stalemating opponent: 10 points per remaining player
        return 10 * num_remaining_players


@jax.jit
def update_scores_for_elimination(
    player_scores: chex.Array,
    eliminating_player: chex.Numeric,
    eliminated_player: chex.Numeric,
    is_checkmate: chex.Numeric,
    num_remaining: chex.Numeric
) -> chex.Array:
    """
    JIT-compatible update of scores when a player is eliminated.

    Uses jax.lax.cond for branch control so the function can be jitted.
    """
    def checkmate_branch(ps):
        bonus = calculate_checkmate_score(num_remaining)
        return ps.at[eliminating_player].add(bonus)

    def stalemate_branch(ps):
        # Determine if stalemate was self-inflicted in a JAX-friendly way
        is_self_stalemate = (eliminating_player == eliminated_player)

        def self_branch(inner_ps):
            bonus = SCORE_STALEMATE_SELF
            return inner_ps.at[eliminated_player].add(bonus)

        def opponent_branch(inner_ps):
            bonus = 10 * num_remaining
            return inner_ps.at[eliminating_player].add(bonus)

        return jax.lax.cond(is_self_stalemate, self_branch, opponent_branch, operand=ps)

    return jax.lax.cond(is_checkmate, checkmate_branch, stalemate_branch, operand=player_scores)
