"""
Debug why is_square_attacked says Blue attacks but we can't find the piece
"""

import requests
import json
import jax.numpy as jnp
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    PAWN, KING, QUEEN, CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    RED, BLUE
)
from four_player_chess.board import create_valid_square_mask
from four_player_chess.rules import is_square_attacked

# Load game state
url = "https://gist.githubusercontent.com/ericyuxuanye/22ed11686879721da3101c466ce3a91d/raw/ceab36d4ec1bac8e8e05f15cf4afce16ddc67628/gistfile1.txt"
response = requests.get(url)
raw_data = json.loads(response.text)
game_data = raw_data['basic_state']
detailed_state = raw_data.get('detailed_state', {})

board_data = game_data['board']
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)
for row in range(14):
    for col in range(14):
        cell = board_data[row][col]
        board = board.at[row, col, CHANNEL_PIECE_TYPE].set(cell['piece'])
        board = board.at[row, col, CHANNEL_OWNER].set(cell['owner'])
        board = board.at[row, col, 2].set(1 if cell.get('has_moved', False) else 0)
        board = board.at[row, col, 3].set(1 if cell['valid'] else 0)

king_positions = jnp.array(detailed_state['king_positions'])
valid_mask = create_valid_square_mask()

print("Testing attack detection with pawn at [11, 9]")
print("=" * 80)

# Simulate move: [12, 9] -> [11, 9]
test_board = board.copy()
test_board = test_board.at[11, 9, CHANNEL_PIECE_TYPE].set(PAWN)
test_board = test_board.at[11, 9, CHANNEL_OWNER].set(RED)
test_board = test_board.at[11, 9, CHANNEL_HAS_MOVED].set(1)
test_board = test_board.at[12, 9, CHANNEL_PIECE_TYPE].set(0)
test_board = test_board.at[12, 9, CHANNEL_OWNER].set(0)

king_row, king_col = int(king_positions[RED][0]), int(king_positions[RED][1])

print(f"Red King at: [{king_row}, {king_col}]")
print(f"Red Pawn moved to: [11, 9]")
print()

# Check if Blue attacks the king
attacked_by_blue = is_square_attacked(
    test_board,
    king_row,
    king_col,
    BLUE,
    valid_mask
)

print(f"is_square_attacked says Blue attacks: {bool(attacked_by_blue)}")
print()

# Manually check each Blue piece
print("Checking each Blue piece manually:")
print("-" * 80)

piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

blue_pieces = []
for row in range(14):
    for col in range(14):
        if int(test_board[row, col, CHANNEL_OWNER]) == BLUE:
            pt = int(test_board[row, col, CHANNEL_PIECE_TYPE])
            if pt > 0:
                blue_pieces.append((row, col, pt))

# Check using pseudo-legal moves
from four_player_chess.pieces import get_pseudo_legal_moves

for row, col, piece_type in blue_pieces:
    # Get moves for this piece
    moves = get_pseudo_legal_moves(
        test_board, row, col, BLUE, valid_mask, jnp.array([-1, -1], dtype=jnp.int32)
    )

    can_attack_king = bool(moves[king_row, king_col])
    if can_attack_king:
        print(f"✓ {piece_names[piece_type]} at [{row}, {col}] CAN attack king")

# Now check with can_piece_attack_square
print("\nUsing can_piece_attack_square:")
from four_player_chess.pieces import can_piece_attack_square

for row, col, piece_type in blue_pieces:
    can_attack = can_piece_attack_square(
        test_board, row, col,
        king_row, king_col,
        valid_mask
    )

    if can_attack:
        print(f"✓ {piece_names[piece_type]} at [{row}, {col}] CAN attack king")

# Check using precomputed tables directly
print("\nUsing precomputed CAN_MOVE_4P table:")
from four_player_chess.precompute import CAN_MOVE_4P, COORD_TO_IDX

king_idx = COORD_TO_IDX[king_row, king_col]

for row, col, piece_type in blue_pieces:
    attacker_idx = COORD_TO_IDX[row, col]
    can_reach_geometric = bool(CAN_MOVE_4P[piece_type, attacker_idx, king_idx])

    if can_reach_geometric:
        print(f"⚠️  {piece_names[piece_type]} at [{row}, {col}] - Precompute table says CAN reach king!")
        print(f"    But get_pseudo_legal_moves says it {'CAN' if piece_type in [p[2] for p in blue_pieces if get_pseudo_legal_moves(test_board, p[0], p[1], BLUE, valid_mask, jnp.array([-1, -1], dtype=jnp.int32))[king_row, king_col]] else 'CANNOT'} attack")

        # For pawns, check the direction
        if piece_type == PAWN:
            print(f"    This is a PAWN - checking direction...")
            print(f"    Pawn at [{row}, {col}], King at [{king_row}, {king_col}]")
            print(f"    Row difference: {king_row - row}, Col difference: {king_col - col}")
            print(f"    Blue pawns move DOWN (positive row direction)")
            if king_row < row:
                print(f"    ❌ King is ABOVE the pawn - pawn cannot attack backward!")
