"""
Check if BETWEEN_4P table is returning correct indices
"""

import jax.numpy as jnp
from four_player_chess.precompute import BETWEEN_4P, COORD_TO_IDX
from four_player_chess.constants import BOARD_SIZE

# Queen at [8, 12], King at [13, 7]
queen_idx = COORD_TO_IDX[8, 12]
king_idx = COORD_TO_IDX[13, 7]

print(f"Queen at [8, 12] -> valid square index {queen_idx}")
print(f"King at [13, 7] -> valid square index {king_idx}")
print()

between_squares = BETWEEN_4P[queen_idx, king_idx]
print(f"BETWEEN_4P[{queen_idx}, {king_idx}] = {between_squares}")
print()

# Convert to row, col
print("Squares between queen and king:")
for i in range(12):
    sq = int(between_squares[i])
    if sq >= 0:
        # BETWEEN_4P returns FLAT BOARD INDICES (0-195), not valid square indices!
        row = sq // BOARD_SIZE
        col = sq % BOARD_SIZE
        print(f"  between_squares[{i}] = {sq} -> [{row}, {col}]")
    else:
        print(f"  between_squares[{i}] = {sq} (end of path)")
        break

print()
print("Expected path: [9, 11], [10, 10], [11, 9], [12, 8]")
