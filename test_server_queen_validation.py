"""
Test to simulate what the web server does when validating queen moves.
This will help identify if there's a bug in the check validation logic.
"""
import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import json
from four_player_chess import FourPlayerChessEnv
from four_player_chess.constants import PIECE_NAMES, PLAYER_NAMES
from four_player_chess.board import create_valid_square_mask
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check
from four_player_chess.constants import CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED, KING

# Load the debug state
with open('debug_state.json', 'r') as f:
    debug_data = json.load(f)

# Rebuild the JAX board state from the JSON
def build_jax_board_from_json(json_board):
    """Reconstruct a JAX board array from JSON."""
    board = jnp.zeros((14, 14, 4), dtype=jnp.int32)

    for row in range(14):
        for col in range(14):
            cell = json_board[row][col]
            board = board.at[row, col, 0].set(cell['piece'])  # CHANNEL_PIECE_TYPE
            board = board.at[row, col, 1].set(cell['owner'])  # CHANNEL_OWNER
            board = board.at[row, col, 2].set(0)  # CHANNEL_HAS_MOVED (we don't have this in JSON)
            board = board.at[row, col, 3].set(1 if cell['valid'] else 0)  # Valid square mask

    return board

# Build board
board = build_jax_board_from_json(debug_data['basic_state']['board'])
current_player = debug_data['basic_state']['current_player']
player_active = jnp.array(debug_data['basic_state']['active_players'], dtype=jnp.bool_)
king_positions = jnp.array(debug_data['detailed_state']['king_positions'], dtype=jnp.int32)
valid_mask = create_valid_square_mask()
en_passant_square = jnp.array(debug_data['detailed_state']['en_passant_square'], dtype=jnp.int32)

# Queen position
queen_row, queen_col = 9, 10

print("=" * 70)
print("SIMULATING WEB SERVER QUEEN MOVE VALIDATION")
print("=" * 70)
print(f"\nQueen position: [{queen_row}, {queen_col}]")
print(f"Current player: {current_player} (Red)")
print(f"Red King position: {king_positions[current_player]}")

# Get pseudo-legal moves
piece_type = int(board[queen_row, queen_col, 0])
owner = int(board[queen_row, queen_col, 1])

print(f"\nQueen owner: {owner}")
print(f"Queen piece type: {piece_type} ({PIECE_NAMES.get(piece_type, 'Unknown')})")

# Get pseudo-legal moves for the queen
pseudo_moves = get_pseudo_legal_moves(
    board,
    queen_row,
    queen_col,
    current_player,
    valid_mask,
    en_passant_square
)

print(f"\n🔍 TESTING EACH PSEUDO-LEGAL MOVE:")
print(f"   (Simulating the check validation from /api/valid_moves)\n")

valid_moves = []
test_positions = []

for move_row in range(14):
    for move_col in range(14):
        if pseudo_moves[move_row, move_col]:
            test_positions.append((move_row, move_col))

print(f"Total pseudo-legal moves: {len(test_positions)}\n")

for move_row, move_col in test_positions:
    # Simulate the move - exactly as the web server does
    test_board = board.copy()

    # Move piece to destination
    test_board = test_board.at[move_row, move_col, CHANNEL_PIECE_TYPE].set(piece_type)
    test_board = test_board.at[move_row, move_col, CHANNEL_OWNER].set(current_player)
    test_board = test_board.at[move_row, move_col, CHANNEL_HAS_MOVED].set(1)

    # Clear source square
    test_board = test_board.at[queen_row, queen_col, CHANNEL_PIECE_TYPE].set(0)  # EMPTY
    test_board = test_board.at[queen_row, queen_col, CHANNEL_OWNER].set(0)
    test_board = test_board.at[queen_row, queen_col, CHANNEL_HAS_MOVED].set(0)

    # Update king position if we're moving the king
    test_king_pos = jnp.where(
        piece_type == KING,
        jnp.array([move_row, move_col], dtype=jnp.int32),
        king_positions[current_player]
    )

    # Check if king would be in check after this move
    in_check_after = is_in_check(
        test_board,
        test_king_pos,
        current_player,
        player_active,
        valid_mask
    )

    # Convert to Python bool
    is_still_in_check = bool(in_check_after.item() if hasattr(in_check_after, 'item') else in_check_after)

    # Determine if this move is valid
    if not is_still_in_check:
        valid_moves.append({'row': move_row, 'col': move_col})
        print(f"✓ [{move_row:2d}, {move_col:2d}] - VALID (king not in check)")
    else:
        print(f"✗ [{move_row:2d}, {move_col:2d}] - REJECTED (would leave king in check)")

        # Extra debugging for rejected moves
        # Check which opponent is attacking
        print(f"   Checking attacks after move to [{move_row}, {move_col}]:")
        for opponent in range(4):
            if opponent != current_player and player_active[opponent]:
                from four_player_chess.rules import is_square_attacked
                attacked = is_square_attacked(
                    test_board,
                    test_king_pos[0],
                    test_king_pos[1],
                    opponent,
                    valid_mask
                )
                if attacked:
                    print(f"      ⚠️  {PLAYER_NAMES[opponent]} can attack the king!")

print(f"\n" + "=" * 70)
print(f"SUMMARY:")
print(f"  Pseudo-legal moves: {len(test_positions)}")
print(f"  Valid moves (after check filter): {len(valid_moves)}")
print(f"  Filtered out: {len(test_positions) - len(valid_moves)}")
print("=" * 70)

# Highlight specific square the user mentioned
print(f"\n🎯 SPECIFIC SQUARE CHECK: [9, 13]")
print(f"   (Adjacent to Blue Bishop at [8, 13])")

if [9, 13] in [[m['row'], m['col']] for m in valid_moves]:
    print("   ✓ This move IS included in valid moves")
else:
    print("   ✗ This move is NOT included in valid moves")
    if (9, 13) in test_positions:
        print("   → It WAS in pseudo-legal moves but got filtered out by check validation")
    else:
        print("   → It was NOT even in pseudo-legal moves")
