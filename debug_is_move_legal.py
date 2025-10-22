"""
Debug the is_move_legal function in detail.
"""

import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask, get_piece_at
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check
from four_player_chess.constants import EMPTY, KING, CHANNEL_PIECE_TYPE, CHANNEL_OWNER

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

# Move details
source_row, source_col = 12, 3
dest_row, dest_col = 10, 3
player = 0

valid_mask = create_valid_square_mask()

print("="*70)
print("DEBUGGING is_move_legal STEP BY STEP")
print("="*70)

# Step 1: Check piece ownership
piece_owner = state.board[source_row, source_col, CHANNEL_OWNER]
piece_type = state.board[source_row, source_col, CHANNEL_PIECE_TYPE]

print(f"\n1. PIECE CHECKS")
print(f"   piece_owner = {int(piece_owner)}, player = {player}")
print(f"   piece_type = {int(piece_type)} (should not be {EMPTY})")

wrong_owner = piece_owner != player
is_empty = piece_type == EMPTY
invalid_piece = wrong_owner | is_empty

print(f"   wrong_owner = {bool(wrong_owner)}")
print(f"   is_empty = {bool(is_empty)}")
print(f"   invalid_piece = {bool(invalid_piece)}")

# Step 2: Check destination validity
invalid_dest = ~valid_mask[dest_row, dest_col]
print(f"\n2. DESTINATION CHECK")
print(f"   valid_mask[{dest_row},{dest_col}] = {bool(valid_mask[dest_row, dest_col])}")
print(f"   invalid_dest = {bool(invalid_dest)}")

# Step 3: Check pseudo-legal
pseudo_legal = get_pseudo_legal_moves(
    state.board, source_row, source_col, player, valid_mask, state.en_passant_square
)
not_pseudo_legal = ~pseudo_legal[dest_row, dest_col]

print(f"\n3. PSEUDO-LEGAL CHECK")
print(f"   pseudo_legal[{dest_row},{dest_col}] = {bool(pseudo_legal[dest_row, dest_col])}")
print(f"   not_pseudo_legal = {bool(not_pseudo_legal)}")

# Step 4: Basic checks
basic_checks_fail = invalid_piece | invalid_dest | not_pseudo_legal
print(f"\n4. BASIC CHECKS")
print(f"   basic_checks_fail = {bool(basic_checks_fail)}")

# Step 5: Simulate move
test_board = state.board.copy()
test_board = test_board.at[dest_row, dest_col, CHANNEL_PIECE_TYPE].set(piece_type)
test_board = test_board.at[dest_row, dest_col, CHANNEL_OWNER].set(player)
test_board = test_board.at[source_row, source_col, CHANNEL_PIECE_TYPE].set(EMPTY)
test_board = test_board.at[source_row, source_col, CHANNEL_OWNER].set(0)

print(f"\n5. MOVE SIMULATION")
print(f"   Moved piece from ({source_row},{source_col}) to ({dest_row},{dest_col})")

# Step 6: King position
test_king_pos = jnp.where(
    piece_type == KING,
    jnp.array([dest_row, dest_col], dtype=jnp.int32),
    state.king_positions[player]
)
print(f"   test_king_pos = [{int(test_king_pos[0])}, {int(test_king_pos[1])}]")

# Step 7: Check detection
in_check_after = is_in_check(test_board, test_king_pos, player, state.player_active, valid_mask)
print(f"\n6. CHECK DETECTION")
print(f"   in_check_after = {bool(in_check_after)}")
print(f"   ~in_check_after = {bool(~in_check_after)}")

# Step 8: Final result
result = jnp.where(basic_checks_fail, False, ~in_check_after)
print(f"\n7. FINAL RESULT")
print(f"   jnp.where(basic_checks_fail={bool(basic_checks_fail)}, False, ~in_check_after={bool(~in_check_after)})")
print(f"   result = {bool(result)}")
print(f"   result type = {type(result)}")
print(f"   result value = {result}")

print("\n" + "="*70)