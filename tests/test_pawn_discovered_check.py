"""
Test for pawn movement with discovered check scenarios.

The bug: A pawn can move 2 squares but not 1 square when there's a discovered check.

Example scenario:
- Red King at [13, 7]
- Red Pawn at [12, 7] (directly in front of king)
- Yellow Queen at [0, 7] (attacking down the file)

If the pawn moves 1 square to [11, 7], it's still on the same file - discovered check!
If the pawn moves 2 squares to [10, 7], it's still on the same file - should also be discovered check!

Actually, this doesn't make sense. Both moves should be illegal if they leave the king in check.

Let me think of another scenario:
- Red King at [13, 6]
- Red Pawn at [12, 6] (directly in front of king)
- Yellow Queen at [0, 6] (attacking down the column)

Same issue - both moves should be illegal.

Wait, maybe the scenario is different. Let me think about a DIAGONAL pin:
- Red King at [13, 7]
- Red Pawn at [12, 6] (diagonally in front of king)
- Yellow Bishop at [0, 5] (on diagonal: could attack king if pawn moves away)

Nope, if pawn moves forward (to [11, 6] or [10, 6]), it leaves the diagonal, so discovered check.

Actually, I think I misunderstood the user's question. Let me re-read it.

The user said: "red's pawn whose column is between the bishop and the rook, can only move 2 squares, not one"

Maybe it's a UI bug where the moves are being displayed incorrectly? Or maybe the game state I'm testing doesn't match what they're seeing?

Let me just check if the web UI has access to the correct king positions.
"""

import jax
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    PAWN, KING, QUEEN,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER,
    RED, YELLOW
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check
from four_player_chess.board import create_valid_square_mask

print("Testing discovered check scenarios with pawns...")
print("=" * 80)

env = FourPlayerChessEnv()
valid_mask = create_valid_square_mask()

# Scenario 1: Pawn pinned on same file as king
print("\nScenario 1: Vertical Pin")
print("-" * 80)
print("Setup:")
print("  - Red King at [13, 7]")
print("  - Red Pawn at [12, 7] (directly in front)")
print("  - Yellow Queen at [0, 7] (same column, attacking down)")
print()

board1 = jnp.zeros((14, 14, 4), dtype=jnp.int32)
board1 = board1.at[:, :, 3].set(valid_mask)  # Set valid squares

# Place pieces
board1 = board1.at[13, 7, CHANNEL_PIECE_TYPE].set(KING)
board1 = board1.at[13, 7, CHANNEL_OWNER].set(RED)
board1 = board1.at[12, 7, CHANNEL_PIECE_TYPE].set(PAWN)
board1 = board1.at[12, 7, CHANNEL_OWNER].set(RED)
board1 = board1.at[0, 7, CHANNEL_PIECE_TYPE].set(QUEEN)
board1 = board1.at[0, 7, CHANNEL_OWNER].set(YELLOW)

# All other kings for valid state
board1 = board1.at[7, 13, CHANNEL_PIECE_TYPE].set(KING)
board1 = board1.at[7, 13, CHANNEL_OWNER].set(1)  # Blue
board1 = board1.at[0, 6, CHANNEL_PIECE_TYPE].set(KING)
board1 = board1.at[0, 6, CHANNEL_OWNER].set(2)  # Yellow
board1 = board1.at[6, 0, CHANNEL_PIECE_TYPE].set(KING)
board1 = board1.at[6, 0, CHANNEL_OWNER].set(3)  # Green

state1 = EnvState(
    board=board1,
    current_player=jnp.int32(RED),
    player_scores=jnp.zeros(4, dtype=jnp.int32),
    player_active=jnp.ones(4, dtype=jnp.bool_),
    move_count=jnp.int32(0),
    en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
    king_positions=jnp.array([[13, 7], [7, 13], [0, 6], [6, 0]], dtype=jnp.int32),
    castling_rights=jnp.ones((4, 2), dtype=jnp.bool_),
    last_capture_move=jnp.int32(0),
    promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
)

# Get pseudo-legal moves
pseudo1 = get_pseudo_legal_moves(board1, 12, 7, RED, valid_mask, state1.en_passant_square)

print("Pseudo-legal moves:")
print(f"  1 square forward [11, 7]: {bool(pseudo1[11, 7])}")
print(f"  2 squares forward [10, 7]: {bool(pseudo1[10, 7])}")

# Check if moves are actually legal (don't leave king in check)
for dest_row, desc in [(11, "1 square"), (10, "2 squares")]:
    if pseudo1[dest_row, 7]:
        # Simulate move
        test_board = board1.copy()
        test_board = test_board.at[dest_row, 7, CHANNEL_PIECE_TYPE].set(PAWN)
        test_board = test_board.at[dest_row, 7, CHANNEL_OWNER].set(RED)
        test_board = test_board.at[12, 7, CHANNEL_PIECE_TYPE].set(0)  # Clear source

        in_check = is_in_check(
            test_board,
            state1.king_positions[RED],
            RED,
            state1.player_active,
            valid_mask
        )

        print(f"  {desc:12s}: pseudo-legal={True:5}, leaves king in check={bool(in_check):5}, LEGAL={not bool(in_check)}")
    else:
        print(f"  {desc:12s}: pseudo-legal={False:5}")

print("\nExpected: Both moves should be ILLEGAL (both leave king in check)")
print("=" * 80)
