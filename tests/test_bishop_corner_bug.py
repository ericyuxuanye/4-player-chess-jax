"""
Test for bishop moving through invalid corner squares.

Bug: Bishop at [3, 0] can attack [0, 3] by going through the invalid
corner squares [2, 1] and [1, 2].

This shouldn't be allowed - the path goes through invalid squares!
"""

import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    BISHOP, KING, CHANNEL_PIECE_TYPE, CHANNEL_OWNER,
    GREEN, YELLOW
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.board import create_valid_square_mask

print("Testing Bishop Movement Through Invalid Corners")
print("=" * 80)

env = FourPlayerChessEnv()
valid_mask = create_valid_square_mask()

# Create minimal board
board = jnp.zeros((14, 14, 4), dtype=jnp.int32)
board = board.at[:, :, 3].set(valid_mask)

# Place Green bishop at [3, 0]
board = board.at[3, 0, CHANNEL_PIECE_TYPE].set(BISHOP)
board = board.at[3, 0, CHANNEL_OWNER].set(GREEN)

# Place kings for valid state
board = board.at[13, 7, CHANNEL_PIECE_TYPE].set(KING)
board = board.at[13, 7, CHANNEL_OWNER].set(0)  # Red
board = board.at[7, 13, CHANNEL_PIECE_TYPE].set(KING)
board = board.at[7, 13, CHANNEL_OWNER].set(1)  # Blue
board = board.at[0, 6, CHANNEL_PIECE_TYPE].set(KING)
board = board.at[0, 6, CHANNEL_OWNER].set(2)  # Yellow
board = board.at[6, 0, CHANNEL_PIECE_TYPE].set(KING)
board = board.at[6, 0, CHANNEL_OWNER].set(3)  # Green

state = EnvState(
    board=board,
    current_player=jnp.int32(GREEN),
    player_scores=jnp.zeros(4, dtype=jnp.int32),
    player_active=jnp.ones(4, dtype=jnp.bool_),
    move_count=jnp.int32(0),
    en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
    king_positions=jnp.array([[13, 7], [7, 13], [0, 6], [6, 0]], dtype=jnp.int32),
    castling_rights=jnp.ones((4, 2), dtype=jnp.bool_),
    last_capture_move=jnp.int32(0),
    promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
)

print("\nBoard Setup:")
print(f"  Green Bishop at: [3, 0]")
print(f"  Target square: [0, 3]")
print()

# Check valid mask along the diagonal
print("Diagonal path from [3, 0] to [0, 3]:")
path = [[3, 0], [2, 1], [1, 2], [0, 3]]
for row, col in path:
    is_valid = bool(valid_mask[row, col])
    print(f"  [{row}, {col}]: {'VALID' if is_valid else 'INVALID (corner)'}")

print()

# Get pseudo-legal moves for the bishop
pseudo_moves = get_pseudo_legal_moves(
    board, 3, 0, GREEN, valid_mask, state.en_passant_square
)

can_reach_0_3 = bool(pseudo_moves[0, 3])

print(f"Bishop pseudo-legal moves:")
print(f"  Can move to [0, 3]? {can_reach_0_3}")
print()

if can_reach_0_3:
    print("❌ BUG CONFIRMED: Bishop can move through invalid corner squares!")
    print("   The path from [3,0] to [0,3] goes through:")
    print("   - [2, 1] (INVALID)")
    print("   - [1, 2] (INVALID)")
    print("   This move should NOT be allowed!")
else:
    print("✓ No bug: Bishop correctly cannot move through invalid squares")

# Check all bishop moves to see what it CAN reach
print("\nAll squares the bishop can reach:")
count = 0
for row in range(14):
    for col in range(14):
        if pseudo_moves[row, col]:
            count += 1
            # Check if path is valid
            dr = row - 3
            dc = col - 0
            if abs(dr) == abs(dc) and dr != 0:  # Diagonal move
                print(f"  [{row}, {col}]", end="")
                # Verify path
                steps = abs(dr)
                dr_step = 1 if dr > 0 else -1
                dc_step = 1 if dc > 0 else -1
                path_valid = True
                for i in range(1, steps):
                    check_r = 3 + i * dr_step
                    check_c = 0 + i * dc_step
                    if not valid_mask[check_r, check_c]:
                        path_valid = False
                        print(f" (path through invalid [{check_r},{check_c}]!)", end="")
                        break
                print()

print(f"\nTotal reachable squares: {count}")
