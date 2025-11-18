"""Test that the fix works for sliding moves."""
import sys
sys.path.append('/home/user/4-player-chess-jax')

import jax.numpy as jnp
from four_player_chess.pieces import get_sliding_moves
from four_player_chess.constants import ROOK_DIRECTIONS, QUEEN_DIRECTIONS
from four_player_chess.board import create_valid_square_mask

# Create a simple board with a queen at [9, 10]
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)
board = board.at[9, 10, 0].set(5)  # Queen (piece type 5)
board = board.at[9, 10, 1].set(0)  # Red player (player 0)

# Set valid mask
valid_mask = create_valid_square_mask()

# Also mark valid squares in the board
board = board.at[:, :, 3].set(valid_mask)

# Get sliding moves for the queen
moves = get_sliding_moves(board, 9, 10, 0, valid_mask, QUEEN_DIRECTIONS)

print("Testing queen moves from [9, 10]:")
print("="*60)

# Check horizontal moves to the right
print("\nHorizontal moves to the right (row 9):")
for col in range(11, 14):
    result = "✓ VALID" if moves[9, col] else "✗ INVALID"
    print(f"  [9, {col:2d}]: {result}")

# The critical test: [9, 13] should be valid!
if moves[9, 13]:
    print("\n🎉 SUCCESS! The bug is fixed!")
    print("   Queen can now move to [9, 13]")
else:
    print("\n❌ FAILED! The bug still exists.")
    print("   Queen cannot move to [9, 13]")

print("\n" + "="*60)
print("Full move map (showing all valid moves):")
print("="*60)
for row in range(14):
    for col in range(14):
        if row == 9 and col == 10:
            print("Q ", end="")  # Queen position
        elif moves[row, col]:
            print("* ", end="")  # Valid move
        elif valid_mask[row, col]:
            print(". ", end="")  # Empty square
        else:
            print("  ", end="")  # Invalid square
    print()
