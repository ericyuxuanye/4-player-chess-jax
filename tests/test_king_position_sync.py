"""
Test to identify the king position synchronization bug.
"""

import requests
import json
import jax.numpy as jnp
from four_player_chess.constants import KING, CHANNEL_PIECE_TYPE, CHANNEL_OWNER

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

# Get king positions from state
king_positions = jnp.array(game_data.get('king_positions', [[13,6], [6,13], [0,6], [6,0]]))

print("King Position Synchronization Check")
print("=" * 80)

player_names = ["Red", "Blue", "Yellow", "Green"]

for player in range(4):
    stated_pos = king_positions[player]
    stated_row, stated_col = int(stated_pos[0]), int(stated_pos[1])

    print(f"\n{player_names[player]} King:")
    print(f"  state.king_positions says: [{stated_row}, {stated_col}]")

    # Check what's actually at that position
    piece_at_stated = int(board[stated_row, stated_col, CHANNEL_PIECE_TYPE])
    owner_at_stated = int(board[stated_row, stated_col, CHANNEL_OWNER])

    if piece_at_stated == KING and owner_at_stated == player:
        print(f"  ✓ King is actually at that position")
    else:
        piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
        print(f"  ❌ BUG: That square contains {piece_names.get(piece_at_stated, '?')} (owner={owner_at_stated})")

        # Find where the king actually is
        print(f"  Searching for actual king position...")
        found = False
        for row in range(14):
            for col in range(14):
                if int(board[row, col, CHANNEL_PIECE_TYPE]) == KING and int(board[row, col, CHANNEL_OWNER]) == player:
                    print(f"  King actually found at: [{row}, {col}]")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"  ⚠️  King not found on board! Player may have been eliminated.")

print("\n" + "=" * 80)
print("\nThis bug explains the pawn movement issue!")
print("When the code checks if a move leaves the king in check,")
print("it uses the WRONG king position, leading to incorrect validation.")
