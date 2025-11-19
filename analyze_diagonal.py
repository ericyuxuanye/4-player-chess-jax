"""
Analyze the diagonal from Blue Queen to Red King
"""

import requests
import json
import jax.numpy as jnp

# Fetch game state
url = "https://gist.githubusercontent.com/ericyuxuanye/22ed11686879721da3101c466ce3a91d/raw/ceab36d4ec1bac8e8e05f15cf4afce16ddc67628/gistfile1.txt"
response = requests.get(url)
raw_data = json.loads(response.text)
game_data = raw_data['basic_state']
board_data = game_data['board']

piece_symbols = {0: "  ·", 1: "  P", 2: "  N", 3: "  B", 4: "  R", 5: "  Q", 6: "  K"}
player_names = {0: "R", 1: "B", 2: "Y", 3: "G"}

print("Diagonal Analysis: Blue Queen [8, 12] to Red King [13, 7]")
print("=" * 80)
print("\nDiagonal equation: row + col = 20")
print("\nSquares on the diagonal:")

diagonal_squares = []
for row in range(14):
    for col in range(14):
        if row + col == 20:
            diagonal_squares.append((row, col))

diagonal_squares.sort()

for row, col in diagonal_squares:
    if 0 <= row < 14 and 0 <= col < 14:
        cell = board_data[row][col]
        piece = cell['piece']
        owner = cell['owner']
        symbol = piece_symbols[piece]
        owner_str = player_names.get(owner, " ") if piece > 0 else " "

        marker = ""
        if row == 8 and col == 12:
            marker = " <- Blue Queen"
        elif row == 13 and col == 7:
            marker = " <- Red King"
        elif row == 12 and col == 8:
            marker = " <- Red Pawn (blocker)"
        elif row == 11 and col == 9:
            marker = " <- PAWN WOULD MOVE HERE (1 sq)"
        elif row == 12 and col == 9:
            marker = " <- Red Pawn (current position)"

        print(f"  [{row:2d}, {col:2d}]: {symbol}{owner_str}{marker}")

print("\n" + "=" * 80)
print("EXPLANATION")
print("=" * 80)
print("""
Currently, the diagonal is blocked by the Red pawn at [12, 8].

If the pawn at [12, 9] moves to [11, 9]:
  - The pawn at [12, 9] is NOT on the diagonal (12+9=21)
  - The destination [11, 9] IS on the diagonal (11+9=20)
  - But [11, 9] is BETWEEN the queen and the blocking pawn at [12, 8]
  - So it doesn't help - the line from [8,12] to [11,9] is still clear!
  - After [11, 9], the queen can't reach the king because [12, 8] still blocks

Wait, that's not right. Let me reconsider...

Actually, let's trace the diagonal from Queen to King:
  [8, 12] Queen
  [9, 11] -> checking...
  [10, 10] -> checking...
  [11, 9] -> pawn would be here
  [12, 8] -> Red pawn blocks
  [13, 7] King

So if there's a blocker at [12, 8], the queen can't reach the king anyway!

Unless... maybe there's ANOTHER line of attack?
""")

print("\nLet me check ALL possible attack lines from Blue Queen [8, 12]:")
print("-" * 80)

# Check all 8 directions from queen
directions = [
    (-1, -1, "up-left diagonal"),
    (-1, 0, "up"),
    (-1, 1, "up-right diagonal"),
    (0, -1, "left"),
    (0, 1, "right"),
    (1, -1, "down-left diagonal"),
    (1, 0, "down"),
    (1, 1, "down-right diagonal"),
]

queen_row, queen_col = 8, 12
king_row, king_col = 13, 7

for dr, dc, direction in directions:
    # Check if king is in this direction
    # Calculate if king is on this line
    if dr == 0:  # horizontal
        if queen_row == king_row:
            print(f"\n{direction}: King IS on this line!")
            # Trace the path
            step = 1 if dc > 0 else -1
            path = []
            for c in range(queen_col + step, king_col, step):
                cell = board_data[queen_row][c]
                if cell['piece'] > 0:
                    path.append(f"[{queen_row},{c}] {piece_symbols[cell['piece']]}{player_names[cell['owner']]}")
            if path:
                print(f"  Blockers: {', '.join(path)}")
            else:
                print(f"  No blockers - KING IN CHECK!")
    elif dc == 0:  # vertical
        if queen_col == king_col:
            print(f"\n{direction}: King IS on this line!")
            step = 1 if dr > 0 else -1
            path = []
            for r in range(queen_row + step, king_row, step):
                cell = board_data[r][queen_col]
                if cell['piece'] > 0:
                    path.append(f"[{r},{queen_col}] {piece_symbols[cell['piece']]}{player_names[cell['owner']]}")
            if path:
                print(f"  Blockers: {', '.join(path)}")
            else:
                print(f"  No blockers - KING IN CHECK!")
    else:  # diagonal
        # Check if on same diagonal
        if (king_row - queen_row) * dc == (king_col - queen_col) * dr:
            print(f"\n{direction}: King IS on this line!")
            steps = abs(king_row - queen_row)
            path = []
            for i in range(1, steps):
                r = queen_row + i * dr
                c = queen_col + i * dc
                if 0 <= r < 14 and 0 <= c < 14:
                    cell = board_data[r][c]
                    if cell['piece'] > 0:
                        path.append(f"[{r},{c}] {piece_symbols[cell['piece']]}{player_names[cell['owner']]}")
            if path:
                print(f"  Blockers: {', '.join(path)}")
                print(f"  Path from queen: ", end="")
                for i in range(1, steps):
                    r = queen_row + i * dr
                    c = queen_col + i * dc
                    if 0 <= r < 14 and 0 <= c < 14:
                        cell = board_data[r][c]
                        sym = piece_symbols[cell['piece']]
                        own = player_names[cell['owner']] if cell['piece'] > 0 else " "
                        print(f"[{r},{c}]{sym}{own} ", end="")
                print()
            else:
                print(f"  No blockers - KING IN CHECK!")
