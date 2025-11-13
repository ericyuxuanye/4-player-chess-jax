"""Debug script to test why Red's queen cannot capture Blue's knight."""

import jax.numpy as jnp
from four_player_chess.constants import (
    QUEEN, KNIGHT, KING, RED, BLUE, YELLOW, GREEN, PIECE_NAMES, PLAYER_NAMES,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED
)
from four_player_chess.pieces import get_pseudo_legal_moves
from four_player_chess.rules import is_in_check, is_square_attacked
from four_player_chess.board import create_valid_square_mask

def print_board_row(board, row):
    """Print a single row of the board."""
    pieces = {
        0: '.', 1: '♟', 2: '♞', 3: '♝', 4: '♜', 5: '♛', 6: '♚'
    }
    colors = {0: 'R', 1: 'B', 2: 'Y', 3: 'G'}

    row_str = ""
    for col in range(14):
        piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
        owner = int(board[row, col, CHANNEL_OWNER])
        if piece_type == 0:
            row_str += " . "
        else:
            row_str += f"{colors[owner]}{pieces[piece_type]}"
    print(f"Row {row:2d}: {row_str}")

def analyze_queen_move(state, queen_pos, target_pos):
    """Analyze why a queen move might be blocked."""
    valid_mask = create_valid_square_mask()
    qr, qc = queen_pos
    tr, tc = target_pos

    print(f"\n=== Analyzing Queen move from ({qr},{qc}) to ({tc},{tc}) ===")
    print(f"Current player: {PLAYER_NAMES[int(state.current_player)]}")
    print(f"Red King position: {state.king_positions[RED]}")

    # Get pseudo-legal moves
    pseudo_moves = get_pseudo_legal_moves(
        state.board, qr, qc, RED, valid_mask, state.en_passant_square
    )

    print(f"\nIs ({tr},{tc}) pseudo-legal? {bool(pseudo_moves[tr, tc])}")

    if not pseudo_moves[tr, tc]:
        print("  → Move is not even pseudo-legal (blocked or invalid path)")
        return

    # Simulate the move
    test_board = state.board.copy()
    piece_type = int(state.board[qr, qc, CHANNEL_PIECE_TYPE])

    captured_piece = int(test_board[tr, tc, CHANNEL_PIECE_TYPE])
    captured_owner = int(test_board[tr, tc, CHANNEL_OWNER])

    if captured_piece > 0:
        print(f"  Capturing: {PIECE_NAMES[captured_piece]} (owner: {PLAYER_NAMES[captured_owner]})")

    test_board = test_board.at[tr, tc, CHANNEL_PIECE_TYPE].set(piece_type)
    test_board = test_board.at[tr, tc, CHANNEL_OWNER].set(RED)
    test_board = test_board.at[tr, tc, CHANNEL_HAS_MOVED].set(1)
    test_board = test_board.at[qr, qc, CHANNEL_PIECE_TYPE].set(0)
    test_board = test_board.at[qr, qc, CHANNEL_OWNER].set(0)
    test_board = test_board.at[qr, qc, CHANNEL_HAS_MOVED].set(0)

    # Check if in check after move
    in_check_after = is_in_check(
        test_board, state.king_positions[RED], RED, state.player_active, valid_mask
    )

    print(f"\nWould Red king be in check after move? {bool(in_check_after)}")

    if in_check_after:
        print("\n  Checking which opponent is attacking:")
        for opponent in [BLUE, YELLOW, GREEN]:
            if state.player_active[opponent]:
                attacked = is_square_attacked(
                    test_board,
                    state.king_positions[RED][0],
                    state.king_positions[RED][1],
                    opponent,
                    valid_mask
                )
                if attacked:
                    print(f"    {PLAYER_NAMES[opponent]} is attacking the king!")
                    # Find which piece is attacking
                    for row in range(14):
                        for col in range(14):
                            if test_board[row, col, CHANNEL_OWNER] == opponent:
                                pt = int(test_board[row, col, CHANNEL_PIECE_TYPE])
                                if pt > 0:
                                    # Check if this piece can attack the king
                                    from four_player_chess.pieces import can_piece_attack_square
                                    if can_piece_attack_square(
                                        test_board, row, col,
                                        state.king_positions[RED][0],
                                        state.king_positions[RED][1],
                                        valid_mask
                                    ):
                                        print(f"      → {PIECE_NAMES[pt]} at ({row},{col})")

def test_queen_capture():
    """Test if Red's queen can capture Blue's knight."""
    from four_player_chess import FourPlayerChessEnv
    import jax

    env = FourPlayerChessEnv()
    rng_key = jax.random.PRNGKey(0)
    state, obs = env.reset(rng_key)

    print("=== Initial Board State ===")
    for row in range(14):
        print_board_row(state.board, row)

    print("\n=== Finding pieces ===")
    for row in range(14):
        for col in range(14):
            piece_type = int(state.board[row, col, CHANNEL_PIECE_TYPE])
            owner = int(state.board[row, col, CHANNEL_OWNER])
            if piece_type == QUEEN and owner == RED:
                print(f"Red Queen at: ({row}, {col})")
            if piece_type == KNIGHT and owner == BLUE:
                print(f"Blue Knight at: ({row}, {col})")

    print("\nNOTE: This script starts with initial board state.")
    print("To debug your specific game state, you need to:")
    print("1. Run the web UI (python web_ui/app.py)")
    print("2. Load your game state in the browser")
    print("3. Try to select the Red Queen and see the debug output in the server console")

if __name__ == "__main__":
    test_queen_capture()
