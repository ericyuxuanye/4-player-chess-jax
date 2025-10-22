"""
Debug script to diagnose the move validation issue for move 12,3,10,3
"""

import jax
import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask, get_piece_at
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_move_legal, is_in_check, is_square_attacked
from four_player_chess.constants import PIECE_NAMES, PLAYER_NAMES

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

print("="*70)
print("DEBUGGING MOVE 12,3 -> 10,3 (Red pawn forward 2 squares)")
print("="*70)

# Move details
source_row, source_col = 12, 3
dest_row, dest_col = 10, 3
current_player = int(state.current_player)

print(f"\n1. INITIAL STATE")
print(f"   Current player: {current_player} ({PLAYER_NAMES[current_player]})")
print(f"   Red king position: {state.king_positions[0]}")

# Check piece at source
piece_type, piece_owner = get_piece_at(state.board, source_row, source_col)
print(f"\n2. PIECE AT SOURCE ({source_row},{source_col})")
print(f"   Piece type: {int(piece_type)} ({PIECE_NAMES.get(int(piece_type), 'Unknown')})")
print(f"   Owner: {int(piece_owner)} ({PLAYER_NAMES[int(piece_owner)]})")

# Check if Red king is in check BEFORE the move
valid_mask = create_valid_square_mask()
king_pos = state.king_positions[0]
in_check_before = is_in_check(
    state.board, 
    king_pos, 
    0,  # Red player
    state.player_active, 
    valid_mask
)
print(f"\n3. RED KING CHECK STATUS BEFORE MOVE")
print(f"   Is Red king in check? {bool(in_check_before)}")

# Check pseudo-legal moves
pseudo_legal = get_pseudo_legal_moves(
    state.board, source_row, source_col,
    current_player, valid_mask, state.en_passant_square
)
print(f"\n4. PSEUDO-LEGAL MOVES")
print(f"   Can pawn move to ({dest_row},{dest_col})? {bool(pseudo_legal[dest_row, dest_col])}")

# Show all pseudo-legal destinations for this pawn
valid_dests = []
for r in range(14):
    for c in range(14):
        if pseudo_legal[r, c]:
            valid_dests.append((r, c))
print(f"   All pseudo-legal destinations: {valid_dests}")

# Simulate the move manually
print(f"\n5. SIMULATING MOVE")
test_board = state.board.copy()
test_board = test_board.at[dest_row, dest_col, 0].set(piece_type)  # piece type
test_board = test_board.at[dest_row, dest_col, 1].set(current_player)  # owner
test_board = test_board.at[source_row, source_col, 0].set(0)  # clear source - piece type
test_board = test_board.at[source_row, source_col, 1].set(0)  # clear source - owner

print(f"   Moved piece from ({source_row},{source_col}) to ({dest_row},{dest_col})")

# Check if king is in check AFTER the move
in_check_after = is_in_check(
    test_board,
    king_pos,  # King hasn't moved
    0,  # Red player
    state.player_active,
    valid_mask
)
print(f"\n6. RED KING CHECK STATUS AFTER MOVE")
print(f"   Is Red king in check? {bool(in_check_after)}")

# Check which opponent pieces might be attacking the king
if in_check_after:
    print(f"\n7. ANALYZING ATTACKS ON RED KING AT {king_pos}")
    for opponent in [1, 2, 3]:  # Blue, Yellow, Green
        if state.player_active[opponent]:
            attacks = is_square_attacked(
                test_board,
                king_pos[0], king_pos[1],
                opponent,
                valid_mask
            )
            print(f"   {PLAYER_NAMES[opponent]} attacking? {bool(attacks)}")
            
            # Find which pieces are attacking
            if attacks:
                print(f"   Checking {PLAYER_NAMES[opponent]} pieces:")
                for r in range(14):
                    for c in range(14):
                        p_type, p_owner = get_piece_at(test_board, r, c)
                        if int(p_owner) == opponent and int(p_type) != 0:
                            # Check if this piece can attack the king
                            pseudo = get_pseudo_legal_moves(
                                test_board, r, c, opponent, valid_mask,
                                jnp.array([-1, -1], dtype=jnp.int32)
                            )
                            if pseudo[int(king_pos[0]), int(king_pos[1])]:
                                print(f"      -> {PIECE_NAMES[int(p_type)]} at ({r},{c}) can attack king!")

# Test is_move_legal function
print(f"\n8. TESTING is_move_legal FUNCTION")
move_legal = is_move_legal(
    state.board,
    source_row, source_col,
    dest_row, dest_col,
    current_player,
    state.king_positions[current_player],
    state.player_active,
    valid_mask,
    state.en_passant_square
)
print(f"   is_move_legal result: {bool(move_legal)}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)