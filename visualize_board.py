"""
Visualize the board to understand the pawn movement issue.
"""

import requests
import json
import jax.numpy as jnp
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    PAWN, KING, QUEEN, ROOK, BISHOP, KNIGHT,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER,
    RED, BLUE, YELLOW, GREEN
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check, is_square_attacked
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

# Convert board to numpy array
board_data = game_data['board']
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)

# Parse the board
for row in range(14):
    for col in range(14):
        cell = board_data[row][col]
        board = board.at[row, col, CHANNEL_PIECE_TYPE].set(cell['piece'])
        board = board.at[row, col, CHANNEL_OWNER].set(cell['owner'])
        board = board.at[row, col, 2].set(1 if cell.get('has_moved', False) else 0)
        board = board.at[row, col, 3].set(1 if cell['valid'] else 0)

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

# Visualize the board
piece_symbols = {
    (0, 0): '  ·  ',  # empty
    (1, 0): ' ♟R  ', (1, 1): ' ♟B  ', (1, 2): ' ♟Y  ', (1, 3): ' ♟G  ',  # pawns
    (2, 0): ' ♞R  ', (2, 1): ' ♞B  ', (2, 2): ' ♞Y  ', (2, 3): ' ♞G  ',  # knights
    (3, 0): ' ♝R  ', (3, 1): ' ♝B  ', (3, 2): ' ♝Y  ', (3, 3): ' ♝G  ',  # bishops
    (4, 0): ' ♜R  ', (4, 1): ' ♜B  ', (4, 2): ' ♜Y  ', (4, 3): ' ♜G  ',  # rooks
    (5, 0): ' ♛R  ', (5, 1): ' ♛B  ', (5, 2): ' ♛Y  ', (5, 3): ' ♛G  ',  # queens
    (6, 0): ' ♚R  ', (6, 1): ' ♚B  ', (6, 2): ' ♚Y  ', (6, 3): ' ♚G  ',  # kings
}

print("\nBoard Visualization:")
print("=" * 80)
print("   ", end="")
for col in range(14):
    print(f"  {col:2d} ", end="")
print()

for row in range(14):
    print(f"{row:2d}  ", end="")
    for col in range(14):
        piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
        owner = int(board[row, col, CHANNEL_OWNER])
        valid = bool(board[row, col, 3])

        if not valid:
            print("  X  ", end="")
        else:
            symbol = piece_symbols.get((piece_type, owner), '  ?  ')
            print(symbol, end="")
    print()

print("\n" + "=" * 80)
print(f"Red King Position: {state.king_positions[RED]}")
print()

# Focus on Red's back rank and pawn rank
print("Red's Back Rank (row 13):")
for col in range(14):
    piece_type = int(board[13, col, CHANNEL_PIECE_TYPE])
    owner = int(board[13, col, CHANNEL_OWNER])
    if piece_type != 0 and owner == RED:
        piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
        print(f"  Column {col}: {piece_names.get(piece_type, '?')}")

print("\nRed's Pawn Rank (row 12):")
for col in range(14):
    piece_type = int(board[12, col, CHANNEL_PIECE_TYPE])
    owner = int(board[12, col, CHANNEL_OWNER])
    if piece_type == PAWN and owner == RED:
        print(f"  Column {col}: Pawn")

print("\n" + "=" * 80)
print("Checking specific pawns for discovered check scenarios...")
print()

valid_mask = create_valid_square_mask()

# Check each pawn at row 12
for pawn_col in range(3, 11):
    if int(board[12, pawn_col, CHANNEL_PIECE_TYPE]) == PAWN and int(board[12, pawn_col, CHANNEL_OWNER]) == RED:
        print(f"\nPawn at [12, {pawn_col}]:")

        # Get pseudo-legal moves
        pseudo_moves = get_pseudo_legal_moves(
            board, 12, pawn_col, RED, valid_mask, state.en_passant_square
        )

        # Check moves: 1 square and 2 squares forward
        can_move_1 = bool(pseudo_moves[11, pawn_col])
        can_move_2 = bool(pseudo_moves[10, pawn_col])

        print(f"  Pseudo-legal: 1 sq={can_move_1}, 2 sq={can_move_2}")

        # Now check if these moves would leave king in check
        from four_player_chess.constants import CHANNEL_HAS_MOVED

        # Test 1 square forward
        if can_move_1:
            test_board_1 = board.copy()
            test_board_1 = test_board_1.at[11, pawn_col, CHANNEL_PIECE_TYPE].set(PAWN)
            test_board_1 = test_board_1.at[11, pawn_col, CHANNEL_OWNER].set(RED)
            test_board_1 = test_board_1.at[12, pawn_col, CHANNEL_PIECE_TYPE].set(0)

            in_check_1 = is_in_check(
                test_board_1,
                state.king_positions[RED],
                RED,
                state.player_active,
                valid_mask
            )
            legal_1 = not bool(in_check_1)
        else:
            legal_1 = False
            in_check_1 = None

        # Test 2 squares forward
        if can_move_2:
            test_board_2 = board.copy()
            test_board_2 = test_board_2.at[10, pawn_col, CHANNEL_PIECE_TYPE].set(PAWN)
            test_board_2 = test_board_2.at[10, pawn_col, CHANNEL_OWNER].set(RED)
            test_board_2 = test_board_2.at[12, pawn_col, CHANNEL_PIECE_TYPE].set(0)

            in_check_2 = is_in_check(
                test_board_2,
                state.king_positions[RED],
                RED,
                state.player_active,
                valid_mask
            )
            legal_2 = not bool(in_check_2)
        else:
            legal_2 = False
            in_check_2 = None

        print(f"  Legal moves: 1 sq={legal_1}, 2 sq={legal_2}")

        if can_move_1 and not legal_1:
            print(f"    ⚠️  1 square move leaves king in check!")

            # Find which opponent is attacking
            for opponent in range(4):
                if opponent != RED and state.player_active[opponent]:
                    attacked = is_square_attacked(
                        test_board_1,
                        state.king_positions[RED][0],
                        state.king_positions[RED][1],
                        opponent,
                        valid_mask
                    )
                    if attacked:
                        player_names = ["Red", "Blue", "Yellow", "Green"]
                        print(f"       King would be attacked by {player_names[opponent]}")

        if can_move_2 and not legal_2:
            print(f"    ⚠️  2 square move leaves king in check!")

        if can_move_1 and legal_1 and can_move_2 and not legal_2:
            print(f"    🐛 BUG: Can move 1 square but NOT 2 squares!")

        if can_move_2 and legal_2 and can_move_1 and not legal_1:
            print(f"    🐛 BUG: Can move 2 squares but NOT 1 square!")
