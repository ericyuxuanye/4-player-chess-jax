"""
Debug why Red knight has no legal moves on first move.
"""

import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask, get_piece_at
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_move_legal, is_in_check
from four_player_chess.constants import PIECE_NAMES, KNIGHT

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

valid_mask = create_valid_square_mask()

print("="*70)
print("DEBUGGING RED KNIGHT MOVES")
print("="*70)

# Red knights are at (13, 4) and (13, 9)
knight_positions = [(13, 4), (13, 9)]

for knight_row, knight_col in knight_positions:
    piece_type, piece_owner = get_piece_at(state.board, knight_row, knight_col)
    
    print(f"\nKnight at ({knight_row},{knight_col}):")
    print(f"  Piece type: {int(piece_type)} ({PIECE_NAMES.get(int(piece_type), 'Unknown')})")
    print(f"  Owner: {int(piece_owner)}")
    
    if int(piece_type) != KNIGHT:
        print(f"  WARNING: Not a knight!")
        continue
    
    # Get pseudo-legal moves
    pseudo_legal = get_pseudo_legal_moves(
        state.board, knight_row, knight_col,
        0,  # Red player
        valid_mask, 
        state.en_passant_square
    )
    
    # Find all pseudo-legal destinations
    pseudo_dests = []
    for r in range(14):
        for c in range(14):
            if pseudo_legal[r, c]:
                pseudo_dests.append((r, c))
    
    print(f"\n  Pseudo-legal destinations ({len(pseudo_dests)}):")
    for dest_r, dest_c in pseudo_dests:
        print(f"    -> ({dest_r},{dest_c})")
    
    # Check which are actually legal (not leaving king in check)
    legal_dests = []
    for dest_r, dest_c in pseudo_dests:
        is_legal = is_move_legal(
            state.board,
            knight_row, knight_col,
            dest_r, dest_c,
            0,  # Red player
            state.king_positions[0],
            state.player_active,
            valid_mask,
            state.en_passant_square
        )
        if is_legal:
            legal_dests.append((dest_r, dest_c))
        else:
            # Debug why it's not legal
            # Simulate the move
            test_board = state.board.copy()
            test_board = test_board.at[dest_r, dest_c, 0].set(piece_type)
            test_board = test_board.at[dest_r, dest_c, 1].set(0)
            test_board = test_board.at[knight_row, knight_col, 0].set(0)
            test_board = test_board.at[knight_row, knight_col, 1].set(0)
            
            in_check = is_in_check(
                test_board,
                state.king_positions[0],
                0,
                state.player_active,
                valid_mask
            )
            print(f"    -> ({dest_r},{dest_c}): ILLEGAL (leaves king in check: {bool(in_check)})")
    
    print(f"\n  Legal destinations ({len(legal_dests)}):")
    for dest_r, dest_c in legal_dests:
        print(f"    -> ({dest_r},{dest_c})")

# Check if Red king is in check initially
king_pos = state.king_positions[0]
in_check_initially = is_in_check(
    state.board,
    king_pos,
    0,
    state.player_active,
    valid_mask
)
print(f"\n\nRed king at [{int(king_pos[0])}, {int(king_pos[1])}]")
print(f"Is Red king in check initially? {bool(in_check_initially)}")

print("\n" + "="*70)