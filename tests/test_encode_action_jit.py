"""
Test that encode_action is JIT-compatible.
"""

import jax
import jax.numpy as jnp
from four_player_chess.utils import encode_action
from four_player_chess.board import create_valid_square_mask

print("Testing encode_action JIT compatibility")
print("=" * 60)

# Create valid mask
valid_mask = create_valid_square_mask()

# Test 1: Direct JIT compilation of encode_action
print("\nTest 1: Direct JIT compilation")
print("-" * 60)

# The function is already decorated with @jax.jit, so just call it
action = encode_action(12, 3, 10, 3, 0, valid_mask)
print(f"Action (12,3) -> (10,3): {action}")
print("✓ encode_action is JIT-compatible")

# Test 2: JIT compilation in a larger function
print("\nTest 2: JIT compilation within larger function")
print("-" * 60)

@jax.jit
def compute_multiple_actions(rows, cols, valid_mask):
    """Compute actions for multiple moves."""
    actions = jax.vmap(
        lambda r_src, c_src, r_dst, c_dst: encode_action(
            r_src, c_src, r_dst, c_dst, 0, valid_mask
        )
    )(rows[:, 0], cols[:, 0], rows[:, 1], cols[:, 1])
    return actions

# Test with multiple moves
rows = jnp.array([[12, 10], [12, 11], [11, 10]])
cols = jnp.array([[3, 3], [4, 4], [3, 3]])

actions = compute_multiple_actions(rows, cols, valid_mask)
print(f"Computed {len(actions)} actions via vmap + JIT")
print(f"Actions: {actions}")
print("✓ encode_action works with vmap and JIT")

# Test 3: Verify actions are correct
print("\nTest 3: Verify action values")
print("-" * 60)

# Manually compute expected action for (12,3) -> (10,3)
action1 = encode_action(12, 3, 10, 3, 0, valid_mask)
print(f"Pawn forward 2: {action1}")

# Different move
action2 = encode_action(12, 4, 11, 4, 0, valid_mask)
print(f"Pawn forward 1: {action2}")

assert action1 != action2, "Different moves should have different actions"
print("✓ Actions are correctly differentiated")

print("\n" + "=" * 60)
print("All JIT compatibility tests passed!")
print("encode_action is now fully JIT-compatible")
print("=" * 60)
