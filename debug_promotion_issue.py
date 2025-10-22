#!/usr/bin/env python3
"""
Debug script to verify the green pawn promotion issue.
"""

import jax.numpy as jnp
from four_player_chess.constants import (
    RED, BLUE, YELLOW, GREEN, PAWN, PROMOTION_RANKS
)
from four_player_chess.board import create_initial_board
from four_player_chess.utils import dict_to_jax_array

def test_promotion_logic():
    """Test the promotion logic for each player."""
    
    print("=" * 60)
    print("PROMOTION RANK ANALYSIS")
    print("=" * 60)
    
    promotion_ranks_array = dict_to_jax_array(PROMOTION_RANKS)
    
    print("\n1. Promotion Ranks (from constants):")
    for player, name in [(RED, "RED"), (BLUE, "BLUE"), (YELLOW, "YELLOW"), (GREEN, "GREEN")]:
        rank = PROMOTION_RANKS[player]
        print(f"   {name:7} = {rank}")
    
    print("\n2. Player Movement Directions:")
    print("   RED:    Moves UP (negative row)    - Vertical   - row 12 → row 3")
    print("   BLUE:   Moves LEFT (negative col)  - Horizontal - col 12 → col 3")
    print("   YELLOW: Moves DOWN (positive row)  - Vertical   - row 1 → row 10")
    print("   GREEN:  Moves RIGHT (positive col) - Horizontal - col 1 → col 10")
    
    print("\n3. Current Bug:")
    print("   The code checks: dest_row == promotion_rank for ALL players")
    print("   This is CORRECT for RED and YELLOW (vertical movement)")
    print("   This is WRONG for BLUE and GREEN (horizontal movement)")
    
    print("\n4. Green Pawn Issue Example:")
    print("   Green pawn at (row=10, col=3):")
    print("   - Current code checks: dest_row (10) == PROMOTION_RANKS[GREEN] (10) → TRUE ✗")
    print("   - Should check:        dest_col (3) == PROMOTION_RANKS[GREEN] (10) → FALSE ✓")
    print("")
    print("   Green pawn at (row=7, col=10):")
    print("   - Current code checks: dest_row (7) == PROMOTION_RANKS[GREEN] (10) → FALSE ✗")
    print("   - Should check:        dest_col (10) == PROMOTION_RANKS[GREEN] (10) → TRUE ✓")
    
    print("\n5. Correct Logic Should Be:")
    print("   if player in [RED, YELLOW]:")
    print("       promote when dest_row == promotion_rank")
    print("   if player in [BLUE, GREEN]:")
    print("       promote when dest_col == promotion_rank")
    
    print("\n" + "=" * 60)
    print("SOLUTION: Use conditional check based on player orientation")
    print("=" * 60)

if __name__ == "__main__":
    test_promotion_logic()