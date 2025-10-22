"""
Find which Blue piece is attacking the Red king at start.
"""

import jax.numpy as jnp
from jax import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.board import create_valid_square_mask, get_piece_at
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.constants import PIECE_NAMES, PLAYER_NAMES, BLUE

# Initialize environment
env = FourPlayerChessEnv()
key = random.PRNGKey(0)
state, obs = env.reset(key)

valid_mask = create_valid_square_mask()
king_pos = state.king_positions[0]  # Red king position

print("="*70)
print(f"RED KING POSITION: [{int(king_pos[0])}, {int(king_pos[1])}]")
print("="*70)

print("\nBLUE PIECES ON THE BOARD:")
print(f"{'Row':<5} {'Col':<5} {'Piece':<10} {'Can Attack King?':<20}")
print("-"*70)

for r in range(14):
    for c in range(14):
        piece_type, piece_owner = get_piece_at(state.board, r, c)
        
        if int(piece_owner) == BLUE and int(piece_type) != 0:
            piece_name = PIECE_NAMES.get(int(piece_type), 'Unknown')
            
            # Check if this piece can attack the king
            pseudo = get_pseudo_legal_moves(
                state.board, r, c, BLUE, valid_mask, 
                jnp.array([-1, -1], dtype=jnp.int32)
            )
            
            can_attack = pseudo[int(king_pos[0]), int(king_pos[1])]
            
            status = "YES ★★★" if can_attack else "No"
            print(f"{r:<5} {c:<5} {piece_name:<10} {status:<20}")
            
            if can_attack:
                print(f"   ↳ ATTACKING: {piece_name} at ({r},{c}) → Red King at ({int(king_pos[0])},{int(king_pos[1])})")

print("\n" + "="*70)