"""
Focused test on the pawn at column 9 bug.

Based on user feedback:
- Pawn at [12, 8]: Correctly CANNOT move (Blue queen threatens check) ✓
- Pawn at [12, 9]: Can move 2 squares but NOT 1 square ❌ BUG!
"""

import requests
import json
import jax.numpy as jnp
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    PAWN, KING, QUEEN, CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    RED, BLUE
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check, is_square_attacked
from four_player_chess.board import create_valid_square_mask

# Fetch the game state
url = "https://gist.githubusercontent.com/ericyuxuanye/22ed11686879721da3101c466ce3a91d/raw/ceab36d4ec1bac8e8e05f15cf4afce16ddc67628/gistfile1.txt"
response = requests.get(url)
raw_data = json.loads(response.text)

if 'basic_state' in raw_data:
    game_data = raw_data['basic_state']
    detailed_state = raw_data.get('detailed_state', {})
else:
    game_data = raw_data
    detailed_state = {}

# Convert board
board_data = game_data['board']
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)
for row in range(14):
    for col in range(14):
        cell = board_data[row][col]
        board = board.at[row, col, CHANNEL_PIECE_TYPE].set(cell['piece'])
        board = board.at[row, col, CHANNEL_OWNER].set(cell['owner'])
        board = board.at[row, col, 2].set(1 if cell.get('has_moved', False) else 0)
        board = board.at[row, col, 3].set(1 if cell['valid'] else 0)

# Get king positions - try detailed_state first
if 'king_positions' in detailed_state:
    king_positions = jnp.array(detailed_state['king_positions'])
    print("Using king positions from detailed_state")
else:
    # Find kings on the board
    king_positions = jnp.zeros((4, 2), dtype=jnp.int32)
    for player in range(4):
        found = False
        for row in range(14):
            for col in range(14):
                if int(board[row, col, CHANNEL_PIECE_TYPE]) == KING and int(board[row, col, CHANNEL_OWNER]) == player:
                    king_positions = king_positions.at[player].set([row, col])
                    found = True
                    break
            if found:
                break
    print("Found king positions by scanning board")

state = EnvState(
    board=board,
    current_player=jnp.int32(RED),
    player_scores=jnp.array(game_data.get('player_scores', [0,0,0,0])),
    player_active=jnp.array(game_data.get('active_players', [True, True, True, True])),
    move_count=jnp.int32(game_data.get('move_count', 0)),
    en_passant_square=jnp.array([-1, -1]),
    king_positions=king_positions,
    castling_rights=jnp.array([[True, True], [True, True], [True, True], [True, True]]),
    last_capture_move=jnp.int32(0),
    promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
)

print("=" * 80)
print("FOCUSED ANALYSIS: Pawn at [12, 9]")
print("=" * 80)

print(f"\nRed King Position: {king_positions[RED]}")
print(f"Red King actual location: [{int(king_positions[RED][0])}, {int(king_positions[RED][1])}]")

# Check what's at the king position
king_row, king_col = int(king_positions[RED][0]), int(king_positions[RED][1])
piece_at_king = int(board[king_row, king_col, CHANNEL_PIECE_TYPE])
owner_at_king = int(board[king_row, king_col, CHANNEL_OWNER])
piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
print(f"Piece at king position: {piece_names[piece_at_king]} (owner={owner_at_king})")

# Find Blue queen
print("\nBlue Queen location:")
for row in range(14):
    for col in range(14):
        if int(board[row, col, CHANNEL_PIECE_TYPE]) == QUEEN and int(board[row, col, CHANNEL_OWNER]) == BLUE:
            print(f"  Blue Queen at [{row}, {col}]")
            blue_queen_pos = (row, col)

valid_mask = create_valid_square_mask()

print("\n" + "-" * 80)
print("Pawn at [12, 9] - Move Analysis")
print("-" * 80)

# Get pseudo-legal moves
pseudo = get_pseudo_legal_moves(board, 12, 9, RED, valid_mask, state.en_passant_square)

print(f"\nPseudo-legal moves:")
print(f"  1 square [11, 9]: {bool(pseudo[11, 9])}")
print(f"  2 squares [10, 9]: {bool(pseudo[10, 9])}")

# Test each move
for dest_row, desc in [(11, "1 square [11, 9]"), (10, "2 squares [10, 9]")]:
    if pseudo[dest_row, 9]:
        print(f"\n{desc}:")

        # Simulate move
        test_board = board.copy()
        test_board = test_board.at[dest_row, 9, CHANNEL_PIECE_TYPE].set(PAWN)
        test_board = test_board.at[dest_row, 9, CHANNEL_OWNER].set(RED)
        test_board = test_board.at[dest_row, 9, CHANNEL_HAS_MOVED].set(1)
        test_board = test_board.at[12, 9, CHANNEL_PIECE_TYPE].set(0)
        test_board = test_board.at[12, 9, CHANNEL_OWNER].set(0)

        # Check if king in check
        in_check = is_in_check(
            test_board,
            king_positions[RED],
            RED,
            state.player_active,
            valid_mask
        )

        print(f"  King in check after move: {bool(in_check)}")
        print(f"  Move is LEGAL: {not bool(in_check)}")

        # If in check, find out why
        if in_check:
            print(f"  Finding which opponent is attacking...")
            for opponent in range(4):
                if opponent != RED and state.player_active[opponent]:
                    attacked = is_square_attacked(
                        test_board,
                        king_positions[RED][0],
                        king_positions[RED][1],
                        opponent,
                        valid_mask
                    )
                    if attacked:
                        player_names = ["Red", "Blue", "Yellow", "Green"]
                        print(f"    ⚠️  {player_names[opponent]} is attacking the king at [{int(king_positions[RED][0])}, {int(king_positions[RED][1])}]!")

                        # Find which piece
                        attacking_pieces = []
                        for r in range(14):
                            for c in range(14):
                                if int(test_board[r, c, CHANNEL_OWNER]) == opponent:
                                    from four_player_chess.pieces import can_piece_attack_square
                                    can_attack = can_piece_attack_square(
                                        test_board, r, c,
                                        king_positions[RED][0], king_positions[RED][1],
                                        valid_mask
                                    )
                                    if can_attack:
                                        pt = int(test_board[r, c, CHANNEL_PIECE_TYPE])
                                        attacking_pieces.append((piece_names[pt], r, c))

                        for piece_name, r, c in attacking_pieces:
                            print(f"       {piece_name} at [{r}, {c}] CAN attack king")

                            # Show the path for sliding pieces
                            pt = int(test_board[r, c, CHANNEL_PIECE_TYPE])
                            if pt in [3, 4, 5]:  # Bishop, Rook, Queen
                                kr, kc = int(king_positions[RED][0]), int(king_positions[RED][1])
                                dr = kr - r
                                dc = kc - c
                                # Normalize direction
                                steps = max(abs(dr), abs(dc))
                                if steps > 0:
                                    dr_norm = dr // steps if dr != 0 else 0
                                    dc_norm = dc // steps if dc != 0 else 0
                                    print(f"       Attack line: ", end="")
                                    for i in range(1, steps):
                                        pr = r + i * dr_norm
                                        pc = c + i * dc_norm
                                        piece_there = int(test_board[pr, pc, CHANNEL_PIECE_TYPE])
                                        if piece_there > 0:
                                            print(f"[{pr},{pc}]=BLOCKED ", end="")
                                        else:
                                            print(f"[{pr},{pc}]=clear ", end="")
                                    print()

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if pseudo[11, 9] and pseudo[10, 9]:
    print("Both moves are pseudo-legal. Checking why one might be filtered out...")
else:
    print("One or both moves are not pseudo-legal!")
