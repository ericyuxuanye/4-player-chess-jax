"""Diagnostic test to identify FLOP hotspots."""

import jax
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv

# Create environment
env = FourPlayerChessEnv()

# Generate random key
key = jax.random.PRNGKey(0)

# Reset environment
state, obs = env.reset(key)

print("=" * 80)
print("DIAGNOSTIC: Measuring FLOP contribution from each component")
print("=" * 80)

# Prepare action
from four_player_chess.board import create_valid_square_mask
import numpy as np

valid_mask = create_valid_square_mask()
mask_flat = valid_mask.flatten()

source_flat = 12 * 14 + 3
dest_flat = 10 * 14 + 3

source_idx = int(np.sum(mask_flat[:source_flat] * (mask_flat[:source_flat] > 0)))
dest_idx = int(np.sum(mask_flat[:dest_flat] * (mask_flat[:dest_flat] > 0)))

action = source_idx * (160 * 4) + dest_idx * 4 + 0

key, subkey = jax.random.split(key)

print("\n1. Testing FULL step function:")
print("-" * 80)
lowered_full = jax.jit(env.step).lower(subkey, state, action)
cost_full = lowered_full.cost_analysis()
print(f"   Total FLOPs: {cost_full['flops']:,.0f}")
print(f"   Bytes accessed: {cost_full['bytes accessed']:,.0f}")

# Test individual components
from four_player_chess.rules import (
    is_move_legal, is_checkmate, is_stalemate, 
    has_legal_moves, get_all_legal_moves_for_player
)
from four_player_chess.utils import decode_action

print("\n2. Testing decode_action:")
print("-" * 80)
def test_decode():
    return decode_action(action, valid_mask)
lowered = jax.jit(test_decode).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")

print("\n3. Testing is_move_legal:")
print("-" * 80)
source_row, source_col, dest_row, dest_col, promotion_type = decode_action(action, valid_mask)
def test_is_legal():
    return is_move_legal(
        state.board, source_row, source_col, dest_row, dest_col,
        state.current_player, state.king_positions[state.current_player],
        state.player_active, valid_mask, state.en_passant_square
    )
lowered = jax.jit(test_is_legal).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")

print("\n4. Testing has_legal_moves (called 2x in checkmate/stalemate):")
print("-" * 80)
def test_has_legal():
    return has_legal_moves(
        state.board, state.current_player,
        state.king_positions[state.current_player],
        state.player_active, valid_mask, state.en_passant_square
    )
lowered = jax.jit(test_has_legal).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")
print(f"   Called 2x per step = {cost['flops'] * 2:,.0f} FLOPs")

print("\n5. Testing get_all_legal_moves_for_player (nested vmaps):")
print("-" * 80)
def test_all_moves():
    return get_all_legal_moves_for_player(
        state.board, state.current_player,
        state.king_positions[state.current_player],
        state.player_active, valid_mask, state.en_passant_square
    )
lowered = jax.jit(test_all_moves).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")

print("\n6. Testing is_checkmate:")
print("-" * 80)
def test_checkmate():
    return is_checkmate(
        state.board, state.current_player,
        state.king_positions[state.current_player],
        state.player_active, valid_mask, state.en_passant_square
    )
lowered = jax.jit(test_checkmate).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")

print("\n7. Testing is_stalemate:")
print("-" * 80)
def test_stalemate():
    return is_stalemate(
        state.board, state.current_player,
        state.king_positions[state.current_player],
        state.player_active, valid_mask, state.en_passant_square
    )
lowered = jax.jit(test_stalemate).lower()
cost = lowered.cost_analysis()
print(f"   FLOPs: {cost['flops']:,.0f}")

print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS:")
print("=" * 80)
print(f"Full step: {cost_full['flops']:,.0f} FLOPs")
print("\nExpected breakdown:")
print(f"  - is_checkmate + is_stalemate likely account for majority")
print(f"  - Both call has_legal_moves which calls get_all_legal_moves_for_player")
print(f"  - This creates and checks ~38,416 move possibilities")
print("=" * 80)