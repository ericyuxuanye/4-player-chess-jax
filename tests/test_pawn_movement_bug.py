"""
Test to investigate the pawn movement bug where a pawn can move 2 squares
but not 1 square.
"""

import requests
import json
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    PAWN, KING, QUEEN, ROOK, BISHOP, KNIGHT,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER,
    RED, BLUE, YELLOW, GREEN
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_move_legal
from four_player_chess.board import create_valid_square_mask

# Fetch the game state
url = "https://gist.githubusercontent.com/ericyuxuanye/22ed11686879721da3101c466ce3a91d/raw/ceab36d4ec1bac8e8e05f15cf4afce16ddc67628/gistfile1.txt"
response = requests.get(url)
raw_data = json.loads(response.text)

# Handle the nested structure
if 'basic_state' in raw_data:
    game_data = raw_data['basic_state']
else:
    game_data = raw_data

print("Game State Analysis")
print("=" * 80)
print(f"Current player: {game_data.get('current_player', 0)} (Red)")
print(f"Move count: {game_data.get('move_count', 0)}")
print(f"Player scores: {game_data.get('player_scores', [0,0,0,0])}")
print()

# Convert board to numpy array
board_data = game_data['board']
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)

# Parse the board
for row in range(14):
    for col in range(14):
        cell = board_data[row][col]
        board = board.at[row, col, CHANNEL_PIECE_TYPE].set(cell['piece'])
        board = board.at[row, col, CHANNEL_OWNER].set(cell['owner'])
        board = board.at[row, col, 2].set(1 if cell.get('has_moved', False) else 0)  # has_moved
        board = board.at[row, col, 3].set(1 if cell['valid'] else 0)  # valid square

# Create the state
state = EnvState(
    board=board,
    current_player=jnp.int32(game_data.get('current_player', 0)),
    player_scores=jnp.array(game_data.get('player_scores', [0,0,0,0])),
    player_active=jnp.array(game_data.get('active_players', [True, True, True, True])),
    move_count=jnp.int32(game_data.get('move_count', 0)),
    en_passant_square=jnp.array(game_data.get('en_passant_square', [-1, -1])),
    king_positions=jnp.array(game_data.get('king_positions', [[13,6], [6,13], [0,6], [6,0]])),
    castling_rights=jnp.array(game_data.get('castling_rights', [[True, True], [True, True], [True, True], [True, True]])),
    last_capture_move=jnp.int32(game_data.get('last_capture_move', 0)),
    promoted_pieces=jnp.array(game_data.get('promoted_pieces', [[False]*14]*14))
)

# Find Red's pawns in the starting area
print("Red's Pieces (Player 0):")
print("-" * 80)
piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

red_pawns = []
for row in range(14):
    for col in range(14):
        piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
        owner = int(board[row, col, CHANNEL_OWNER])
        has_moved = int(board[row, col, 2])

        if owner == RED and piece_type != 0:
            print(f"  [{row:2d}, {col:2d}] {piece_names[piece_type]:6s} (moved: {bool(has_moved)})")
            if piece_type == PAWN:
                red_pawns.append((row, col))

print()
print("Analyzing Red's Pawns:")
print("-" * 80)

env = FourPlayerChessEnv()
valid_mask = create_valid_square_mask()

for pawn_row, pawn_col in red_pawns:
    print(f"\nPawn at [{pawn_row}, {pawn_col}]:")

    # Check what's blocking it
    print(f"  Square ahead [1 square]: [{pawn_row-1}, {pawn_col}]")
    if pawn_row > 0:
        piece_ahead_1 = int(board[pawn_row-1, pawn_col, CHANNEL_PIECE_TYPE])
        owner_ahead_1 = int(board[pawn_row-1, pawn_col, CHANNEL_OWNER])
        print(f"    Occupied: {piece_ahead_1 != 0} (type={piece_names.get(piece_ahead_1, '?')}, owner={owner_ahead_1})")

    print(f"  Square ahead [2 squares]: [{pawn_row-2}, {pawn_col}]")
    if pawn_row > 1:
        piece_ahead_2 = int(board[pawn_row-2, pawn_col, CHANNEL_PIECE_TYPE])
        owner_ahead_2 = int(board[pawn_row-2, pawn_col, CHANNEL_OWNER])
        print(f"    Occupied: {piece_ahead_2 != 0} (type={piece_names.get(piece_ahead_2, '?')}, owner={owner_ahead_2})")

    # Get pseudo-legal moves
    pseudo_legal = get_pseudo_legal_moves(
        board, pawn_row, pawn_col, RED, valid_mask, state.en_passant_square
    )

    # Check specific moves
    moves_to_check = [
        (pawn_row - 1, pawn_col, "1 square forward"),
        (pawn_row - 2, pawn_col, "2 squares forward"),
    ]

    print(f"\n  Move Analysis:")
    for dest_row, dest_col, desc in moves_to_check:
        if dest_row >= 0 and dest_row < 14:
            pseudo = bool(pseudo_legal[dest_row, dest_col])

            if pseudo:
                legal = is_move_legal(
                    board, pawn_row, pawn_col, dest_row, dest_col, RED,
                    state.king_positions[RED], state.player_active, valid_mask, state.en_passant_square
                )
            else:
                legal = False

            print(f"    {desc:20s} -> [{dest_row:2d}, {dest_col:2d}]: pseudo={pseudo}, legal={legal}")

# Look for potential discovered check scenarios
print("\n" + "=" * 80)
print("Red King Position:", state.king_positions[RED])
print("\nLooking for potential discovered check issues...")

red_king_pos = state.king_positions[RED]
red_king_row, red_king_col = int(red_king_pos[0]), int(red_king_pos[1])

print(f"\nRed King at [{red_king_row}, {red_king_col}]")

# Find enemy pieces that could be attacking the king
for row in range(14):
    for col in range(14):
        piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
        owner = int(board[row, col, CHANNEL_OWNER])

        if owner != RED and owner != 0 and piece_type in [QUEEN, ROOK, BISHOP]:
            print(f"  Potential threat: {piece_names[piece_type]} at [{row}, {col}] (Player {owner})")
