"""
Coordinate transformation functions for player-relative (ego-centric) input.

This module provides transformations between absolute board coordinates and
player-relative coordinates, allowing each player to use the same coordinate
notation from their perspective.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from four_player_chess.constants import RED, BLUE, YELLOW, GREEN


def relative_to_absolute_coords(
    rel_row: chex.Numeric,
    rel_col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array
) -> Tuple[chex.Numeric, chex.Numeric]:
    """
    Transform player-relative coordinates to absolute board coordinates.

    Each player sees the board as if their pieces are at the "bottom".
    This function rotates coordinates based on the player's perspective.

    Args:
        rel_row: Row in player's perspective (0-13)
        rel_col: Column in player's perspective (0-13)
        player: Player ID (0=Red, 1=Blue, 2=Yellow, 3=Green)
        valid_mask: Valid square mask (14, 14) - unused but kept for API consistency

    Returns:
        (abs_row, abs_col): Absolute board coordinates

    Examples:
        Red (player 0): Identity - no rotation
        Blue (player 1): 90° clockwise rotation
        Yellow (player 2): 180° rotation
        Green (player 3): 270° clockwise rotation
    """
    def red_transform(coords):
        """Red: Identity transformation."""
        rel_r, rel_c = coords
        return rel_r, rel_c

    def blue_transform(coords):
        """Blue: 90° clockwise rotation."""
        rel_r, rel_c = coords
        return rel_c, 13 - rel_r

    def yellow_transform(coords):
        """Yellow: 180° rotation."""
        rel_r, rel_c = coords
        return 13 - rel_r, 13 - rel_c

    def green_transform(coords):
        """Green: 270° clockwise rotation (90° counter-clockwise)."""
        rel_r, rel_c = coords
        return 13 - rel_c, rel_r

    coords = (rel_row, rel_col)
    abs_row, abs_col = jax.lax.switch(
        player,
        [red_transform, blue_transform, yellow_transform, green_transform],
        coords
    )

    return abs_row, abs_col


def absolute_to_relative_coords(
    abs_row: chex.Numeric,
    abs_col: chex.Numeric,
    player: chex.Numeric,
    valid_mask: chex.Array
) -> Tuple[chex.Numeric, chex.Numeric]:
    """
    Transform absolute board coordinates to player-relative coordinates.

    Inverse of relative_to_absolute_coords. Converts absolute board positions
    to coordinates from the player's perspective.

    Args:
        abs_row: Absolute row on board (0-13)
        abs_col: Absolute column on board (0-13)
        player: Player ID (0=Red, 1=Blue, 2=Yellow, 3=Green)
        valid_mask: Valid square mask (14, 14) - unused but kept for API consistency

    Returns:
        (rel_row, rel_col): Coordinates in player's perspective

    Examples:
        Red (player 0): Identity - no rotation
        Blue (player 1): 90° counter-clockwise rotation
        Yellow (player 2): 180° rotation
        Green (player 3): 270° counter-clockwise rotation (90° clockwise)
    """
    def red_inverse(coords):
        """Red: Identity transformation."""
        abs_r, abs_c = coords
        return abs_r, abs_c

    def blue_inverse(coords):
        """Blue: Inverse of 90° clockwise is 90° counter-clockwise."""
        abs_r, abs_c = coords
        return 13 - abs_c, abs_r

    def yellow_inverse(coords):
        """Yellow: 180° rotation is its own inverse."""
        abs_r, abs_c = coords
        return 13 - abs_r, 13 - abs_c

    def green_inverse(coords):
        """Green: Inverse of 270° clockwise is 90° clockwise."""
        abs_r, abs_c = coords
        return abs_c, 13 - abs_r

    coords = (abs_row, abs_col)
    rel_row, rel_col = jax.lax.switch(
        player,
        [red_inverse, blue_inverse, yellow_inverse, green_inverse],
        coords
    )

    return rel_row, rel_col
