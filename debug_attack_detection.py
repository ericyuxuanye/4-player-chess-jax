"""
Debug the is_square_attacked function to find the discrepancy.
"""

import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask
from four_player_chess.rules import is_square_attacked
from four_player_chess.constants import BLUE

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

valid_mask = create_valid_square_mask()
king_pos = state.king_positions[0]  # Red king position

print("="*70)
print("DEBUGGING is_square_attacked FUNCTION")
print("="*70)

print(f"\nRed king position: [{int(king_pos[0])}, {int(king_pos[1])}]")

# Test is_square_attacked for Blue
is_attacked_by_blue = is_square_attacked(
    state.board,
    king_pos[0],
    king_pos[1],
    BLUE,
    valid_mask
)

print(f"\nis_square_attacked(Red king, by Blue): {bool(is_attacked_by_blue)}")

# Let's also check each color
for player_id in range(4):
    from four_player_chess.constants import PLAYER_NAMES
    is_attacked = is_square_attacked(
        state.board,
        king_pos[0],
        king_pos[1],
        player_id,
        valid_mask
    )
    print(f"  Attacked by {PLAYER_NAMES[player_id]}: {bool(is_attacked)}")

# Let's manually check the board near the Red king
print(f"\n\nBOARD NEAR RED KING (rows 11-13, cols 5-9):")
print(f"{'Row':<5} {'Col':<5} {'Piece Type':<12} {'Owner':<10}")
print("-"*50)

from four_player_chess.board import get_piece_at
from four_player_chess.constants import PIECE_NAMES

for r in range(11, 14):
    for c in range(5, 10):
        piece_type, piece_owner = get_piece_at(state.board, r, c)
        piece_name = PIECE_NAMES.get(int(piece_type), 'Empty')
        owner_name = PLAYER_NAMES[int(piece_owner)] if int(piece_type) != 0 else "N/A"
        print(f"{r:<5} {c:<5} {piece_name:<12} {owner_name:<10}")

print("\n" + "="*70)