"""
Test to reproduce the king capture bug where a player is not eliminated
when their king is captured.
"""

import jax
import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState, EnvParams
from four_player_chess.constants import (
    KING, QUEEN, EMPTY, PAWN,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    RED, BLUE, YELLOW, GREEN
)
from four_player_chess.board import create_valid_square_mask

def test_king_capture_eliminates_player():
    """
    Test that capturing a king properly eliminates the player.

    Setup: Red king at (13, 7), Yellow queen at (11, 6)
    Move: Yellow queen captures Red king
    Expected: Red should be marked as inactive
    """
    env = FourPlayerChessEnv()
    key = jax.random.PRNGKey(0)

    # Create a minimal board state
    board = jnp.zeros((14, 14, 4), dtype=jnp.int32)
    valid_mask = create_valid_square_mask()

    # Set valid squares channel
    board = board.at[:, :, 3].set(valid_mask)

    # Place kings for all players
    # Red king at (13, 7)
    board = board.at[13, 7, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[13, 7, CHANNEL_OWNER].set(RED)

    # Blue king at (7, 13)
    board = board.at[7, 13, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[7, 13, CHANNEL_OWNER].set(BLUE)

    # Yellow king at (0, 6)
    board = board.at[0, 6, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[0, 6, CHANNEL_OWNER].set(YELLOW)

    # Green king at (6, 0)
    board = board.at[6, 0, CHANNEL_PIECE_TYPE].set(KING)
    board = board.at[6, 0, CHANNEL_OWNER].set(GREEN)

    # Place Yellow queen at (13, 6) - can capture Red king horizontally
    board = board.at[13, 6, CHANNEL_PIECE_TYPE].set(QUEEN)
    board = board.at[13, 6, CHANNEL_OWNER].set(YELLOW)

    # Create state with Yellow as current player
    state = EnvState(
        board=board,
        current_player=jnp.int32(YELLOW),
        player_scores=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        player_active=jnp.ones(4, dtype=jnp.bool_),
        move_count=jnp.int32(0),
        en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
        king_positions=jnp.array([
            [13, 7],  # Red
            [7, 13],  # Blue
            [0, 6],   # Yellow
            [6, 0]    # Green
        ], dtype=jnp.int32),
        castling_rights=jnp.ones((4, 2), dtype=jnp.bool_),
        last_capture_move=jnp.int32(0),
        promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
    )

    print("Initial state:")
    print(f"  Current player: {state.current_player} (Yellow)")
    print(f"  Active players: {state.player_active}")
    print(f"  Red king at: {state.king_positions[RED]}")
    print(f"  Yellow queen at: (13, 6)")

    # Yellow queen captures Red king: (13, 6) -> (13, 7)
    # Action encoding: we need to manually call _execute_move
    next_state, reward, move_valid = env._execute_move(
        state,
        source_row=13,
        source_col=6,
        dest_row=13,
        dest_col=7,
        promotion_type=0
    )

    print("\nAfter Yellow queen captures Red king:")
    print(f"  Move valid: {move_valid}")
    print(f"  Reward: {reward}")
    print(f"  Current player: {next_state.current_player}")
    print(f"  Active players: {next_state.player_active}")
    print(f"  Red active: {next_state.player_active[RED]}")
    print(f"  Red king position: {next_state.king_positions[RED]}")
    print(f"  Scores: {next_state.player_scores}")

    # Debug: Check if the move should be legal
    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.rules import is_move_legal

    pseudo_legal = get_pseudo_legal_moves(
        state.board, 13, 6, YELLOW, valid_mask, state.en_passant_square
    )
    print(f"\n  Debug: Queen can pseudo-legally move to (13,7)? {pseudo_legal[13, 7]}")

    legal = is_move_legal(
        state.board, 13, 6, 13, 7, YELLOW,
        state.king_positions[YELLOW], state.player_active, valid_mask, state.en_passant_square
    )
    print(f"  Debug: Move is legal? {legal}")
    print(f"  Debug: (13,6) is valid square? {valid_mask[13, 6]}")
    print(f"  Debug: (13,7) is valid square? {valid_mask[13, 7]}")
    print(f"  Debug: Piece at (13,6): type={state.board[13, 6, CHANNEL_PIECE_TYPE]}, owner={state.board[13, 6, CHANNEL_OWNER]}")
    print(f"  Debug: Piece at (13,7): type={state.board[13, 7, CHANNEL_PIECE_TYPE]}, owner={state.board[13, 7, CHANNEL_OWNER]}")

    # Verify Red is eliminated
    assert not next_state.player_active[RED], "BUG: Red should be eliminated after king capture!"
    assert next_state.king_positions[RED, 0] == -1, "Red king position should be marked invalid"
    assert next_state.player_scores[YELLOW] >= 20, "Yellow should get checkmate bonus (20 points)"

    print("\n✓ Test passed: Red is properly eliminated after king capture")

if __name__ == "__main__":
    test_king_capture_eliminates_player()
