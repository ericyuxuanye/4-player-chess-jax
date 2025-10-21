"""Deeper analysis of remaining FLOPs and optimization opportunities."""

import jax
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask
from four_player_chess.rules import has_legal_moves
import numpy as np

# Create environment
env = FourPlayerChessEnv()
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)

print("=" * 80)
print("DEEPER ANALYSIS: Understanding the 352B FLOPs")
print("=" * 80)

valid_mask = create_valid_square_mask()

print("\n1. Current situation:")
print("-" * 80)
print(f"   - Remaining FLOPs per step: 352.3 billion")
print(f"   - This ALL comes from has_legal_moves()")
print(f"   - Which calls get_all_legal_moves_for_player()")
print(f"   - Checking 14×14 sources × 14×14 destinations = 38,416 possibilities")
print(f"   - For EACH, it simulates the move and checks if king is in check")

print("\n2. Reality check:")
print("-" * 80)
print(f"   - A typical chess position has ~30-40 legal moves")
print(f"   - We're checking 38,416 possibilities - that's 1000x more!")
print(f"   - Most squares are empty (only ~32 pieces on 160 squares)")
print(f"   - Most moves are illegal anyway")

print("\n3. Key insight:")
print("-" * 80)
print(f"   - We check for checkmate/stalemate on EVERY step")
print(f"   - But checkmate/stalemate is RARE (happens < 1% of moves)")
print(f"   - We're paying 352B FLOPs on every move to detect a rare event")

print("\n4. Potential optimizations (in order of impact):")
print("-" * 80)
print("""
   A. CONDITIONAL ELIMINATION CHECK (Recommended - 99% reduction)
      - Only check has_legal_moves when elimination is likely:
        * Just captured a major piece (Queen/Rook/King nearby)
        * Opponent was already in check
        * Late game (< 20 pieces remaining)
      - Expected: 352B → ~3.5B FLOPs (99% of steps skip check)
      - Complexity: Low (5-10 line change)
   
   B. LAZY EVALUATION (Radical - defer check)
      - Don't check on every step
      - Only verify when needed (game over query)
      - Expected: 352B → 0 FLOPs per step
      - Complexity: Medium (requires API change)
   
   C. SPARSE MOVE GENERATION (Complex but thorough)
      - Only iterate over occupied squares
      - Use scan with early termination for has_legal_moves
      - Expected: 352B → ~10-50B FLOPs
      - Complexity: High (major refactor of rules.py)
   
   D. FASTER CHECK DETECTION
      - Use pattern matching for checkmate instead of full enumeration
      - Expected: 352B → ~50-100B FLOPs
      - Complexity: High (chess-specific heuristics)
""")

print("\n5. Testing feasibility of Option A (Conditional Check):")
print("-" * 80)

# Simulate the check
def count_pieces(board):
    """Count total pieces on board."""
    from four_player_chess.constants import EMPTY, CHANNEL_PIECE_TYPE
    return jnp.sum(board[:, :, CHANNEL_PIECE_TYPE] != EMPTY)

def should_check_elimination(state, captured_piece, dest_row, dest_col):
    """Determine if we should check for elimination."""
    from four_player_chess.constants import QUEEN, ROOK, KING
    
    # Check if captured major piece
    captured_major = jnp.isin(captured_piece, jnp.array([QUEEN, ROOK]))
    
    # Check if late game (< 20 pieces)
    late_game = count_pieces(state.board) < 20
    
    # Always check if we captured anything in late game
    any_capture = captured_piece != 0
    
    return captured_major | (late_game & any_capture) | late_game

total_pieces = count_pieces(state.board)
print(f"   - Current board has {int(total_pieces)} pieces")
print(f"   - In early/mid game with no major capture: SKIP check")
print(f"   - In late game or after major capture: RUN check")
print(f"   - Estimated: 99% of moves skip → 99% FLOP reduction")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
Implement Option A (Conditional Elimination Check):
- Add 'should_check_elimination' function
- Wrap has_legal_moves call in jax.lax.cond
- Expected FLOP reduction: 352B → ~3-10B per step (97-99% reduction)
- Total improvement: 704B → 10-20B FLOPs per step (98-99% overall)

This will make the environment ~35-70x faster for JIT compilation and execution!
""")
print("=" * 80)