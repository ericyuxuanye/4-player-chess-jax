"""Test JIT compilation of the environment."""

import jax
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv

# Create environment
env = FourPlayerChessEnv()

# Generate random key
key = jax.random.PRNGKey(0)

# Reset environment
state, obs = env.reset(key)

print("Initial state created successfully!")
print(f"Current player: {state.current_player}")
print(f"Active players: {state.player_active}")

# Test with a valid move (Red pawn forward 2 squares)
# Source: row 12, col 3 -> Dest: row 10, col 3
# Simple action encoding for testing
# We need to convert board coordinates to flat valid square indices

from four_player_chess.board import create_valid_square_mask
import numpy as np

# Get valid mask
valid_mask = create_valid_square_mask()
mask_flat = valid_mask.flatten()

# Find the flat indices for source and destination
source_flat = 12 * 14 + 3
dest_flat = 10 * 14 + 3

# Count valid squares before source to get source_idx
source_idx = int(np.sum(mask_flat[:source_flat] * (mask_flat[:source_flat] > 0)))

# Count valid squares before dest to get dest_idx
dest_idx = int(np.sum(mask_flat[:dest_flat] * (mask_flat[:dest_flat] > 0)))

# Encode action: source_idx * (160 * 4) + dest_idx * 4 + promotion_type
action = source_idx * (160 * 4) + dest_idx * 4 + 0

print(f"\nTesting action: {action}")
print("Attempting to JIT compile and execute step...")

# Split key for next step
key, subkey = jax.random.split(key)

# Create a wrapper function that can be JIT compiled
def step_wrapper(key, state, action):
    return env.step(key, state, action)

# Execute step (this will trigger JIT compilation)
try:
    # JIT compile the wrapper function
    lowered = jax.jit(step_wrapper).lower(subkey, state, action)
    print(lowered.cost_analysis())
    # jitted_step = lowered.compile()
    # print("✓ JIT compilation successful!")
    # new_state, new_obs, reward, done, info = jitted_step(subkey, state, action)
    # print(f"Move valid: {info['move_valid']}")
    # print(f"Current player after move: {new_state.current_player}")
    # print(f"Reward: {reward}")
except Exception as e:
    print(f"✗ JIT compilation failed: {e}")
    import traceback
    traceback.print_exc()