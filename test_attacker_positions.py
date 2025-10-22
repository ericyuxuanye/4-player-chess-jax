"""
Test to confirm the bug in is_square_attacked with invalid attacker positions.
"""

import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask
from four_player_chess.constants import BLUE, CHANNEL_OWNER, CHANNEL_PIECE_TYPE, EMPTY, BOARD_SIZE

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

print("="*70)
print("TESTING ATTACKER POSITION BUG")
print("="*70)

# Simulate the logic from is_square_attacked
piece_owners = state.board[:, :, CHANNEL_OWNER]
piece_types = state.board[:, :, CHANNEL_PIECE_TYPE]

# Find Blue pieces
belongs_to_player = (piece_owners == BLUE) & (piece_types != EMPTY)
flat_board = belongs_to_player.flatten()

print(f"\nNumber of Blue pieces: {int(jnp.sum(belongs_to_player))}")

# Get attacker positions with size=16 and fill_value=-1
attacker_positions = jnp.nonzero(flat_board, size=16, fill_value=-1)[0]

print(f"\nAttacker positions (with padding):")
for i, pos in enumerate(attacker_positions):
    pos_int = int(pos)
    if pos_int >= 0:
        row = pos_int // BOARD_SIZE
        col = pos_int % BOARD_SIZE
        piece_type = int(state.board[row, col, CHANNEL_PIECE_TYPE])
        print(f"  [{i}] pos={pos_int:3d} -> ({row:2d},{col:2d}) piece_type={piece_type}")
    else:
        # This is the bug - what happens with -1?
        row = pos_int // BOARD_SIZE
        col = pos_int % BOARD_SIZE  # This will be 13, not -1!
        print(f"  [{i}] pos={pos_int:3d} -> ({row:2d},{col:2d}) [PADDING - NOTE col wraps!]")
        
        # What data do we access?
        try:
            piece_at_wrapped = state.board[row, col, CHANNEL_PIECE_TYPE]
            owner_at_wrapped = state.board[row, col, CHANNEL_OWNER]
            print(f"      Accessing board[-1, 13] gives piece_type={int(piece_at_wrapped)}, owner={int(owner_at_wrapped)}")
        except:
            print(f"      Error accessing board[-1, 13]")

print("\n" + "="*70)