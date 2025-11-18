"""Test script to visualize board and debug queen moves."""
import json
import sys

# Paste your debug state JSON here
DEBUG_STATE = """
{JSON_PLACEHOLDER}
"""

def visualize_board(board_data, highlight_positions=None):
    """Visualize the board with player colors and pieces."""
    piece_symbols = {
        0: '  ',  # empty
        1: '♟ ',  # pawn
        2: '♞ ',  # knight
        3: '♝ ',  # bishop
        4: '♜ ',  # rook
        5: '♛ ',  # queen
        6: '♚ '   # king
    }

    player_colors = {
        0: '\033[91m',  # Red
        1: '\033[94m',  # Blue
        2: '\033[93m',  # Yellow
        3: '\033[92m',  # Green
    }
    reset_color = '\033[0m'

    print("\n" + "=" * 60)
    print("BOARD VISUALIZATION (with coordinates)")
    print("=" * 60)
    print("   ", end="")
    for col in range(14):
        print(f" {col:2d}", end="")
    print()

    for row in range(14):
        print(f"{row:2d} ", end="")
        for col in range(14):
            cell = board_data[row][col]

            if not cell['valid']:
                print(' XX', end="")
            else:
                piece_type = cell['piece']
                owner = cell['owner']
                symbol = piece_symbols.get(piece_type, '? ')

                # Highlight specific positions
                if highlight_positions and [row, col] in highlight_positions:
                    print(f'\033[7m', end="")  # Reverse video

                if piece_type > 0:
                    color = player_colors.get(owner, reset_color)
                    print(f"{color}{symbol}{reset_color}", end="")
                else:
                    print(f" . ", end="")

                if highlight_positions and [row, col] in highlight_positions:
                    print(f'\033[0m', end="")  # Reset reverse video
        print()
    print()


def get_queen_theoretical_moves(queen_pos):
    """Calculate all theoretical queen moves (rook + bishop moves)."""
    row, col = queen_pos
    moves = []

    # Rook-like moves (horizontal and vertical)
    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (-1, -1), # up-left diagonal
        (-1, 1),  # up-right diagonal
        (1, -1),  # down-left diagonal
        (1, 1),   # down-right diagonal
    ]

    for dr, dc in directions:
        for i in range(1, 14):
            new_row = row + dr * i
            new_col = col + dc * i
            if 0 <= new_row < 14 and 0 <= new_col < 14:
                moves.append([new_row, new_col])
            else:
                break  # Out of bounds

    return moves


def is_valid_square(row, col):
    """Check if a square is valid (part of the cross-shaped board)."""
    if row < 0 or row >= 14 or col < 0 or col >= 14:
        return False

    # Top-left corner (invalid)
    if row < 3 and col < 3:
        return False
    # Top-right corner (invalid)
    if row < 3 and col > 10:
        return False
    # Bottom-left corner (invalid)
    if row > 10 and col < 3:
        return False
    # Bottom-right corner (invalid)
    if row > 10 and col > 10:
        return False

    return True


def main():
    # Load the debug state
    data = json.loads(DEBUG_STATE)
    board = data['basic_state']['board']

    # Find the Red Queen
    red_queen_pos = None
    for detail in data['board_details']:
        if detail['piece_name'] == 'Queen' and detail['owner_name'] == 'Red':
            red_queen_pos = detail['position']
            break

    if not red_queen_pos:
        print("Red Queen not found!")
        return

    print(f"\n🔴 RED QUEEN POSITION: {red_queen_pos}")

    # Visualize the board with queen highlighted
    visualize_board(board, highlight_positions=[red_queen_pos])

    # Get theoretical moves
    theoretical_moves = get_queen_theoretical_moves(red_queen_pos)
    print(f"\n📊 THEORETICAL QUEEN MOVES (before filtering): {len(theoretical_moves)}")

    # Filter by valid squares
    valid_theoretical = [m for m in theoretical_moves if is_valid_square(m[0], m[1])]
    print(f"📊 ON VALID SQUARES: {len(valid_theoretical)}")

    # Filter by obstructions
    queen_row, queen_col = red_queen_pos
    current_player = data['basic_state']['current_player']  # Should be 0 (Red)

    print(f"\n🎯 Current player: {current_player} (Red)")
    print(f"\n🔍 ANALYZING EACH DIRECTION:")

    directions = [
        ((-1, 0), "Up"),
        ((1, 0), "Down"),
        ((0, -1), "Left"),
        ((0, 1), "Right"),
        ((-1, -1), "Up-Left diagonal"),
        ((-1, 1), "Up-Right diagonal"),
        ((1, -1), "Down-Left diagonal"),
        ((1, 1), "Down-Right diagonal"),
    ]

    for (dr, dc), direction_name in directions:
        print(f"\n  {direction_name} ({dr:+d}, {dc:+d}):")
        for i in range(1, 14):
            new_row = queen_row + dr * i
            new_col = queen_col + dc * i

            if not (0 <= new_row < 14 and 0 <= new_col < 14):
                print(f"    Step {i}: [{new_row:2d}, {new_col:2d}] - OUT OF BOUNDS")
                break

            if not is_valid_square(new_row, new_col):
                print(f"    Step {i}: [{new_row:2d}, {new_col:2d}] - INVALID SQUARE (corner)")
                break

            cell = board[new_row][new_col]
            piece = cell['piece']
            owner = cell['owner']

            if piece == 0:
                print(f"    Step {i}: [{new_row:2d}, {new_col:2d}] ✓ Empty - VALID MOVE")
            else:
                owner_names = ['Red', 'Blue', 'Yellow', 'Green']
                piece_names = ['Empty', 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']

                if owner == current_player:
                    print(f"    Step {i}: [{new_row:2d}, {new_col:2d}] ✗ Own {piece_names[piece]} - BLOCKED")
                    break
                else:
                    print(f"    Step {i}: [{new_row:2d}, {new_col:2d}] ⚔️  Enemy {owner_names[owner]} {piece_names[piece]} - CAPTURE (then stop)")
                    break

    # Show piece positions around the queen
    print(f"\n🎯 SQUARES AROUND THE QUEEN:")
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            check_row = queen_row + dr
            check_col = queen_col + dc
            if 0 <= check_row < 14 and 0 <= check_col < 14:
                cell = board[check_row][check_col]
                piece_names = ['Empty', 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
                owner_names = ['Red', 'Blue', 'Yellow', 'Green']
                if cell['piece'] > 0:
                    print(f"  [{check_row:2d}, {check_col:2d}]: {owner_names[cell['owner']]} {piece_names[cell['piece']]}")
                else:
                    print(f"  [{check_row:2d}, {check_col:2d}]: Empty")


if __name__ == '__main__':
    # Try to read from debug_state.json
    try:
        with open('debug_state.json', 'r') as f:
            DEBUG_STATE = f.read()
        main()
    except FileNotFoundError:
        print("debug_state.json not found. Please create it with your debug state.")
        sys.exit(1)
