"""
Test castling functionality for 4-player chess.
"""

import jax
import jax.numpy as jnp
from jax import random

from four_player_chess.environment import FourPlayerChessEnv
from four_player_chess.constants import (
    RED, BLUE, YELLOW, GREEN, KING, ROOK,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    INITIAL_KING_POSITIONS, CASTLING_KING_DESTINATIONS,
    QUEENSIDE_CASTLING, KINGSIDE_CASTLING
)
from four_player_chess.rules import is_move_legal
from four_player_chess.pieces import get_king_moves
from four_player_chess.utils import dict_to_jax_array


def test_castling_move_generation():
    """Test that castling moves are generated when path is clear."""
    print("\n=== Testing Castling Move Generation ===")

    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    state, obs = env.reset(key)

    # Red's initial king position: [13, 7]
    print(f"Initial castling rights for Red: {state.castling_rights[RED]}")
    print(f"Initial king position: {state.king_positions[RED]}")

    # Clear the path between king and kingside rook
    # Knight at [13, 9], Bishop at [13, 8]
    test_board = state.board.at[13, 9, CHANNEL_PIECE_TYPE].set(0)  # Clear knight
    test_board = test_board.at[13, 8, CHANNEL_PIECE_TYPE].set(0)   # Clear bishop

    # Get king moves with castling
    king_moves = get_king_moves(
        test_board, 13, 7, RED, env.valid_mask, state.castling_rights
    )

    # Check if castling destination [13, 9] is in the moves
    can_castle_kingside = king_moves[13, 9]

    print(f"Kingside castling move generated: {can_castle_kingside}")

    if can_castle_kingside:
        print("✓ Castling move was correctly generated!")
        return True
    else:
        print("✗ Castling move was NOT generated (expected it to be)")
        return False


def test_red_kingside_castling():
    """Test that Red can castle kingside when conditions are met."""
    print("\n=== Testing Red Kingside Castling Legality ===")

    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    state, obs = env.reset(key)

    # Red's initial king position: [13, 7]
    # Red's kingside rook position: [13, 10]
    # After kingside castling:
    #   King at [13, 9]
    #   Rook at [13, 8]

    print(f"Initial castling rights for Red: {state.castling_rights[RED]}")
    print(f"Initial king position: {state.king_positions[RED]}")

    # Clear the path between king and rook (remove knight and bishop)
    # Knight at [13, 9], Bishop at [13, 8]
    test_board = state.board.at[13, 9, CHANNEL_PIECE_TYPE].set(0)  # Clear knight
    test_board = test_board.at[13, 8, CHANNEL_PIECE_TYPE].set(0)   # Clear bishop

    # Try to castle: move king from [13, 7] to [13, 9]
    source_row, source_col = 13, 7
    dest_row, dest_col = 13, 9

    print(f"Checking if castling move is legal: King from [{source_row}, {source_col}] to [{dest_row}, {dest_col}]")

    # Check if move is legal using is_move_legal
    move_legal = is_move_legal(
        test_board,
        source_row, source_col,
        dest_row, dest_col,
        RED,
        state.king_positions[RED],
        state.player_active,
        env.valid_mask,
        state.en_passant_square,
        state.castling_rights
    )

    print(f"Move is legal: {move_legal}")

    if move_legal:
        print("✓ Castling move is correctly identified as legal!")
        return True
    else:
        print("✗ Castling move should be legal but was marked as illegal!")
        return False


def test_castling_blocked_by_pieces():
    """Test that castling is blocked when pieces are in the way."""
    print("\n=== Testing Castling Blocked by Pieces ===")

    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    state, obs = env.reset(key)

    # Try to castle kingside without clearing the path
    source_row, source_col = 13, 7
    dest_row, dest_col = 13, 9

    print(f"Attempting castling with pieces in the way (knight at [13, 9])...")

    # Check if castling moves are generated (should not be)
    king_moves = get_king_moves(
        state.board, 13, 7, RED, env.valid_mask, state.castling_rights
    )

    can_castle = king_moves[dest_row, dest_col]

    if not can_castle:
        print("✓ Castling was correctly blocked by pieces in the way!")
        return True
    else:
        print("✗ Castling should have been blocked but wasn't!")
        return False


def test_castling_rights_structure():
    """Test that castling rights are properly initialized."""
    print("\n=== Testing Castling Rights Structure ===")

    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    state, obs = env.reset(key)

    print(f"Castling rights shape: {state.castling_rights.shape}")
    print(f"Red castling rights: {state.castling_rights[RED]}")
    print(f"Blue castling rights: {state.castling_rights[BLUE]}")
    print(f"Yellow castling rights: {state.castling_rights[YELLOW]}")
    print(f"Green castling rights: {state.castling_rights[GREEN]}")

    # All players should start with both castling rights
    all_can_castle = jnp.all(state.castling_rights == True)

    if all_can_castle:
        print("✓ All players have castling rights at game start!")
        return True
    else:
        print("✗ Some players missing castling rights!")
        return False


def main():
    """Run all castling tests."""
    print("\n" + "="*60)
    print("CASTLING FUNCTIONALITY TESTS")
    print("="*60)

    results = []

    # Run tests
    results.append(("Castling Rights Structure", test_castling_rights_structure()))
    results.append(("Castling Move Generation", test_castling_move_generation()))
    results.append(("Red Kingside Castling Legality", test_red_kingside_castling()))
    results.append(("Castling Blocked by Pieces", test_castling_blocked_by_pieces()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
