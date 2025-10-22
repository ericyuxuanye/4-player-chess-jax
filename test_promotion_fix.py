#!/usr/bin/env python3
"""
Test script to verify the promotion bug fix.
"""

import jax.numpy as jnp
from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.state import EnvState
from four_player_chess.constants import (
    RED, BLUE, YELLOW, GREEN, PAWN, QUEEN, EMPTY,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER
)
from four_player_chess.board import create_initial_board, set_piece_at, clear_square, create_valid_square_mask, get_initial_king_positions
from jax import random

def test_green_pawn_promotion():
    """Test that green pawn promotes at column 10, not row 10."""
    
    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    
    # Create a custom board with a green pawn near promotion
    board = create_initial_board()
    
    # Clear all pieces for clarity
    for r in range(14):
        for c in range(14):
            board = clear_square(board, r, c)
    
    # Set valid mask back
    valid_mask = create_valid_square_mask()
    board = board.at[:, :, 3].set(valid_mask)
    
    # Place green pawn at (row=7, col=9) - one square away from promotion at col=10
    board = set_piece_at(board, 7, 9, PAWN, GREEN, has_moved=True)
    
    # Place kings
    board = set_piece_at(board, 13, 7, 6, RED)  # KING = 6
    board = set_piece_at(board, 7, 13, 6, BLUE)
    board = set_piece_at(board, 0, 6, 6, YELLOW)
    board = set_piece_at(board, 6, 0, 6, GREEN)
    
    # Create state
    state = EnvState(
        board=board,
        current_player=jnp.int32(GREEN),
        player_scores=jnp.zeros(4, dtype=jnp.int32),
        player_active=jnp.ones(4, dtype=jnp.bool_),
        move_count=jnp.int32(0),
        en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
        king_positions=get_initial_king_positions(),
        castling_rights=jnp.ones((4, 2), dtype=jnp.bool_),
        last_capture_move=jnp.int32(0),
        promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
    )
    
    print("=" * 60)
    print("TEST 1: Green pawn at (row=7, col=9)")
    print("=" * 60)
    print("\nBefore move:")
    print(f"  Position: (row=7, col=9)")
    print(f"  Piece type: {board[7, 9, CHANNEL_PIECE_TYPE]} (PAWN={PAWN})")
    print(f"  Owner: {board[7, 9, CHANNEL_OWNER]} (GREEN={GREEN})")
    
    # Simulate move to (7, 10) - this SHOULD promote (correct column)
    new_state, reward, move_valid = env._execute_move(state, 7, 9, 7, 10, 0)
    
    print(f"\nAfter move to (row=7, col=10):")
    print(f"  Move valid: {move_valid}")
    print(f"  Piece type: {new_state.board[7, 10, CHANNEL_PIECE_TYPE]} (QUEEN={QUEEN})")
    print(f"  Expected: QUEEN (promotion at col=10)")
    
    promoted = new_state.board[7, 10, CHANNEL_PIECE_TYPE] == QUEEN
    print(f"\n  ✓ PASS: Promoted to Queen" if promoted else f"  ✗ FAIL: Did not promote")
    
    # Test 2: Green pawn at (row=10, col=3) - should NOT promote
    board2 = create_initial_board()
    for r in range(14):
        for c in range(14):
            board2 = clear_square(board2, r, c)
    board2 = board2.at[:, :, 3].set(valid_mask)
    
    # Place green pawn at (row=10, col=3)
    board2 = set_piece_at(board2, 10, 3, PAWN, GREEN, has_moved=True)
    board2 = set_piece_at(board2, 13, 7, 6, RED)
    board2 = set_piece_at(board2, 7, 13, 6, BLUE)
    board2 = set_piece_at(board2, 0, 6, 6, YELLOW)
    board2 = set_piece_at(board2, 6, 0, 6, GREEN)
    
    state2 = EnvState(
        board=board2,
        current_player=jnp.int32(GREEN),
        player_scores=jnp.zeros(4, dtype=jnp.int32),
        player_active=jnp.ones(4, dtype=jnp.bool_),
        move_count=jnp.int32(0),
        en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
        king_positions=get_initial_king_positions(),
        castling_rights=jnp.ones((4, 2), dtype=jnp.bool_),
        last_capture_move=jnp.int32(0),
        promoted_pieces=jnp.zeros((14, 14), dtype=jnp.bool_)
    )
    
    print("\n" + "=" * 60)
    print("TEST 2: Green pawn at (row=10, col=3) - the buggy position")
    print("=" * 60)
    print("\nBefore move:")
    print(f"  Position: (row=10, col=3)")
    print(f"  Piece type: {board2[10, 3, CHANNEL_PIECE_TYPE]} (PAWN={PAWN})")
    
    # Move right to (10, 4) - should NOT promote (only at col 3)
    new_state2, reward2, move_valid2 = env._execute_move(state2, 10, 3, 10, 4, 0)
    
    print(f"\nAfter move to (row=10, col=4):")
    print(f"  Move valid: {move_valid2}")
    print(f"  Piece type: {new_state2.board[10, 4, CHANNEL_PIECE_TYPE]} (PAWN={PAWN})")
    print(f"  Expected: PAWN (no promotion, col=4 ≠ 10)")
    
    not_promoted = new_state2.board[10, 4, CHANNEL_PIECE_TYPE] == PAWN
    print(f"\n  ✓ PASS: Stayed as Pawn" if not_promoted else f"  ✗ FAIL: Incorrectly promoted")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if promoted and not_promoted:
        print("✓ ALL TESTS PASSED - Bug is fixed!")
        print("  - Green pawn promotes at col=10 ✓")
        print("  - Green pawn does NOT promote at row=10 ✓")
    else:
        print("✗ TESTS FAILED")
        if not promoted:
            print("  - Green pawn should promote at col=10")
        if not not_promoted:
            print("  - Green pawn should NOT promote at row=10")

if __name__ == "__main__":
    test_green_pawn_promotion()