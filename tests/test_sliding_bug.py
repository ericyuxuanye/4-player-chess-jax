"""Test to demonstrate the sliding moves bug."""
import sys
sys.path.append('/home/user/4-player-chess-jax')

import jax.numpy as jnp

# Simulate what happens in get_sliding_moves
# When moving from [9, 10] in direction [0, 1] (right)

BOARD_SIZE = 14
dists = jnp.arange(1, BOARD_SIZE, dtype=jnp.int32)  # [1, 2, 3, ..., 13]

# For direction [0, 1] from position [9, 10]
dr, dc = 0, 1
row, col = 9, 10

new_rows = row + dr * dists  # [9, 9, 9, 9, ...]
new_cols = col + dc * dists  # [11, 12, 13, 14, 15, ..., 23]

print("Distances and resulting positions:")
for i, (r, c) in enumerate(zip(new_rows, new_cols)):
    print(f"  Distance {i+1}: [{r}, {c}]")

# Clamp to board bounds
clamped_rows = jnp.clip(new_rows, 0, BOARD_SIZE - 1)
clamped_cols = jnp.clip(new_cols, 0, BOARD_SIZE - 1)

print("\nAfter clamping:")
for i, (r, c) in enumerate(zip(clamped_rows, clamped_cols)):
    print(f"  Distance {i+1}: [{r}, {c}]")

# Check which are in bounds
in_bounds = (new_rows >= 0) & (new_rows < BOARD_SIZE) & (new_cols >= 0) & (new_cols < BOARD_SIZE)

print("\nIn bounds check:")
for i, ib in enumerate(in_bounds):
    print(f"  Distance {i+1}: {ib}")

# Simulate can_move_here (assuming squares 11, 12, 13 are all empty/capturable)
# For simplicity, assume all in-bounds squares can be moved to
can_move_here = in_bounds

print("\nCan move here:")
for i, cm in enumerate(can_move_here):
    r, c = clamped_rows[i], clamped_cols[i]
    print(f"  Distance {i+1} → [{r}, {c}]: {cm}")

# Now let's see what happens with .at[].set()
moves = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)

# This is what the code does
moves = moves.at[(clamped_rows, clamped_cols)].set(can_move_here)

print(f"\nFinal result at [9, 13]: {moves[9, 13]}")
print("^ This should be True but it's False!")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("Position [9, 13] appears in the flattened arrays at indices:")
for i, (r, c) in enumerate(zip(clamped_rows, clamped_cols)):
    if r == 9 and c == 13:
        print(f"  Index {i} (distance {i+1}): can_move_here = {can_move_here[i]}")

print("\nBecause JAX's .at[].set() uses the LAST value when there are")
print("duplicate indices, the False value from distance 4+ overwrites the")
print("True value from distance 3!")
