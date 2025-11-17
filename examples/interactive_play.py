"""
Interactive 4-player chess game.

Allows manual input of moves and displays ASCII board after each move.
"""

import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.rendering import render_board, print_game_summary
from four_player_chess.utils import encode_action
from four_player_chess.constants import PLAYER_NAMES, PLAYER_COLORS


def print_help():
    """Print help message about how to play."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    HOW TO PLAY                                ║
╚═══════════════════════════════════════════════════════════════╝

MOVE FORMAT:
  Enter moves in chess notation: <source>,<dest>
  Example: e12,e10  (moves piece from e12 to e10)

  You can also use the legacy format:
  Example: 12,7,10,7  (row,col,row,col)

COMMANDS:
  help     - Show this help message
  show     - Redisplay the current board
  quit     - Exit the game
  hint     - Show valid moves for a piece

COORDINATES:
  - Rows: 0-13 (numbers on the left)
  - Cols: a-h for main columns (3-10), or 0-2, 11-13 for extensions
  - Look at the board labels to find coordinates
  - Format: column+row (e.g., e12, d7, a10)

PLAYERS:
  🔴 Red (0)    - Bottom (rows 11-13)
  🔵 Blue (1)   - Right (cols 11-13)
  🟡 Yellow (2) - Top (rows 0-2)
  🟢 Green (3)  - Left (cols 0-2)

TIPS:
  - Make sure you're moving YOUR pieces (current player shown at top)
  - Invalid moves will be rejected and you can try again
  - Empty squares are shown as ·
    """)


def format_coord(col, row):
    """Format a coordinate in chess notation (e.g., e12)."""
    if 3 <= col <= 10:
        col_str = chr(ord('a') + col - 3)
    else:
        col_str = str(col)
    return f"{col_str}{row}"


def get_piece_at(board, row, col):
    """Get piece info at a square."""
    piece_type = int(board[row, col, 0])
    owner = int(board[row, col, 1])
    return piece_type, owner


def show_available_moves_hint(state, source_row, source_col):
    """Show which squares a piece can potentially move to."""
    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.constants import EMPTY
    
    piece_type, owner = get_piece_at(state.board, source_row, source_col)
    
    if piece_type == EMPTY:
        print(f"  No piece at {format_coord(source_col, source_row)}")
        return

    if owner != state.current_player:
        print(f"  That piece belongs to player {owner}, not player {int(state.current_player)}")
        return
    
    # Get pseudo-legal moves
    from four_player_chess.board import create_valid_square_mask
    valid_mask = create_valid_square_mask()
    
    moves = get_pseudo_legal_moves(
        state.board, source_row, source_col, 
        state.current_player, valid_mask, state.en_passant_square
    )
    
    # Find all destination squares
    destinations = []
    for r in range(14):
        for c in range(14):
            if moves[r, c]:
                destinations.append((r, c))
    
    if destinations:
        print(f"\n  Possible moves for piece at {format_coord(source_col, source_row)}:")
        for r, c in destinations[:20]:  # Show first 20
            print(f"    -> {format_coord(c, r)}")
        if len(destinations) > 20:
            print(f"    ... and {len(destinations) - 20} more")
    else:
        print(f"  No legal moves for piece at {format_coord(source_col, source_row)}")


def parse_move(move_str):
    """
    Parse a move string into coordinates.

    Supports two formats:
    1. Chess notation: e13,e10 (col+row for source and dest)
    2. Legacy format: 13,7,10,7 (row,col,row,col)

    Returns:
        Tuple of (source_row, source_col, dest_row, dest_col) or None if invalid
    """
    def parse_col(col_str):
        """Parse a column - either a letter (a-h for cols 3-10) or a number."""
        col_str = col_str.strip().lower()
        if len(col_str) == 1 and 'a' <= col_str <= 'h':
            return ord(col_str) - ord('a') + 3  # a=3, b=4, ..., h=10
        try:
            col = int(col_str)
            return col if 0 <= col < 14 else None
        except:
            return None

    def parse_square(square_str):
        """Parse a square like 'e13' or '7,13' into (row, col)."""
        square_str = square_str.strip()

        # Try chess notation (letter+number like 'e13')
        if len(square_str) >= 2:
            if square_str[0].isalpha():
                col = parse_col(square_str[0])
                if col is not None:
                    try:
                        row = int(square_str[1:])
                        if 0 <= row < 14:
                            return row, col
                    except:
                        pass

        return None

    try:
        parts = move_str.strip().split(',')

        # Try new format: e13,e10 (2 parts)
        if len(parts) == 2:
            source = parse_square(parts[0])
            dest = parse_square(parts[1])
            if source and dest:
                return source[0], source[1], dest[0], dest[1]

        # Try legacy format: 13,7,10,7 (4 parts)
        if len(parts) == 4:
            try:
                source_row = int(parts[0].strip())
                source_col = int(parts[1].strip())
                dest_row = int(parts[2].strip())
                dest_col = int(parts[3].strip())

                # Validate ranges
                if not (0 <= source_row < 14 and 0 <= source_col < 14):
                    return None
                if not (0 <= dest_row < 14 and 0 <= dest_col < 14):
                    return None

                return source_row, source_col, dest_row, dest_col
            except:
                pass

        return None
    except:
        return None


def debug_board_state(state, row, col):
    """Show detailed info about a square."""
    from four_player_chess.constants import EMPTY
    piece_type, owner = get_piece_at(state.board, row, col)
    piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
    
    print(f"\n🔍 DEBUG: Square ({row},{col})")
    print(f"   Piece: {piece_names.get(piece_type, 'Unknown')}")
    print(f"   Owner: {owner if piece_type != EMPTY else 'N/A'}")
    print(f"   Has moved: {bool(state.board[row, col, 2])}")
    print(f"   Valid square: {bool(state.board[row, col, 3])}")


def play_interactive():
    """
    Main interactive game loop.
    """
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║           4-PLAYER CHESS - INTERACTIVE MODE                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("Type 'help' for instructions on how to play.")
    print("Type 'debug <row> <col>' to see detailed info about a square.")
    print()
    
    # Initialize environment
    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    
    # Reset environment
    key, reset_key = random.split(key)
    state, obs = env.reset(reset_key)

    step_fn = jax.jit(env.step)
    
    # Show initial board
    print(render_board(state, use_unicode=True, use_color=True))
    
    move_num = 0
    
    while True:
        current_player = int(state.current_player)
        
        # Check if game is over
        active_count = int(jnp.sum(state.player_active))
        if active_count <= 1:
            print("\n" + "="*70)
            print("GAME OVER!")
            print("="*70)
            print(print_game_summary(state))
            break
        
        # Prompt for move
        print(f"\n{'─'*70}")
        print(f"Move {move_num + 1} - {PLAYER_COLORS[current_player]} {PLAYER_NAMES[current_player]}'s turn")
        print(f"{'─'*70}")
        
        move_input = input("Enter move (or 'help'/'quit'): ").strip().lower()
        
        # Handle commands
        if move_input == 'quit' or move_input == 'q':
            print("\nThanks for playing!")
            break
        
        if move_input == 'help' or move_input == 'h':
            print_help()
            continue
        
        if move_input == 'show' or move_input == 's':
            print("\n" + render_board(state, use_unicode=True, use_color=True))
            continue
        
        if move_input.startswith('hint'):
            # Extract coordinates if provided
            parts = move_input.split()
            if len(parts) == 3:
                try:
                    hint_row = int(parts[1])
                    hint_col = int(parts[2])
                    show_available_moves_hint(state, hint_row, hint_col)
                except:
                    print("  Invalid coordinates. Usage: hint <row> <col>")
            else:
                print("  Usage: hint <row> <col>")
                print("  Example: hint 12 3")
            continue
        
        if move_input.startswith('debug'):
            # Extract coordinates if provided
            parts = move_input.split()
            if len(parts) == 3:
                try:
                    debug_row = int(parts[1])
                    debug_col = int(parts[2])
                    debug_board_state(state, debug_row, debug_col)
                except:
                    print("  Invalid coordinates. Usage: debug <row> <col>")
            else:
                print("  Usage: debug <row> <col>")
                print("  Example: debug 12 3")
            continue
        
        # Parse move
        parsed = parse_move(move_input)
        if parsed is None:
            print("❌ Invalid move format. Use chess notation: <source>,<dest>")
            print("   Example: e12,e10")
            print("   Or legacy format: 12,7,10,7 (row,col,row,col)")
            continue
        
        source_row, source_col, dest_row, dest_col = parsed
        
        # Check if there's a piece at source
        piece_type, piece_owner = get_piece_at(state.board, source_row, source_col)

        if piece_type == 0:  # EMPTY
            print(f"❌ No piece at {format_coord(source_col, source_row)}")
            continue
        
        if piece_owner != current_player:
            print(f"❌ That piece belongs to player {piece_owner}, not you (player {current_player})")
            continue
        
        # Encode action (using default promotion to queen)
        # Simplified encoding for interactive mode
        try:
            # For interactive mode, we'll create a simpler encoding
            # that directly maps source/dest to action space
            from four_player_chess.board import create_valid_square_mask
            
            valid_mask = create_valid_square_mask()
            
            # Convert board coords to valid square indices
            def coord_to_index(r, c):
                """Convert (row, col) to flat index in valid squares (0-159)"""
                count = 0
                for i in range(14):
                    for j in range(14):
                        if valid_mask[i, j]:
                            if i == r and j == c:
                                return count
                            count += 1
                return -1
            
            source_idx = coord_to_index(source_row, source_col)
            dest_idx = coord_to_index(dest_row, dest_col)
            
            if source_idx < 0 or dest_idx < 0:
                print("❌ Invalid coordinates (not on valid squares)")
                continue
            
            # Encode: action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
            promotion_type = 0  # Queen
            action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
            
        except Exception as e:
            print(f"❌ Error encoding move: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Show what we're trying to do
        print(f"\n🔍 Attempting move: {PLAYER_NAMES[current_player]} piece from {format_coord(source_col, source_row)} to {format_coord(dest_col, dest_row)}")
        
        # Show piece info
        from four_player_chess.constants import PIECE_NAMES
        piece_names = {0: "Empty", 1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
        print(f"   Piece: {piece_names.get(piece_type, 'Unknown')}")
        
        # Try to execute the move
        key, step_key = random.split(key)
        next_state, next_obs, reward, done, info = step_fn(step_key, state, action)
        
        # Check if move was valid
        if not info['move_valid']:
            print("❌ Invalid move! That move is not legal.")
            print("   Possible reasons:")
            print("   - Move doesn't follow piece rules")
            print("   - Path is blocked")
            print("   - Would leave your king in check")
            print("   - Destination is invalid")
            print()
            
            # Try to get more specific info
            from four_player_chess.rules import is_move_legal
            from four_player_chess.pieces import get_pseudo_legal_moves
            from four_player_chess.board import create_valid_square_mask
            
            # Check pseudo-legal moves
            valid_mask = create_valid_square_mask()
            pseudo_legal = get_pseudo_legal_moves(
                state.board, source_row, source_col,
                current_player, valid_mask, state.en_passant_square
            )
            
            if pseudo_legal[dest_row, dest_col]:
                print("   ℹ️  Move follows piece rules but leaves king in check")
            else:
                print("   ℹ️  Move doesn't follow piece movement rules")
                print()
                # Show what moves ARE available for this piece
                valid_dests = []
                for r in range(14):
                    for c in range(14):
                        if pseudo_legal[r, c]:
                            valid_dests.append((r, c))
                
                if valid_dests:
                    print(f"   Valid destinations for this piece (showing first 10):")
                    for r, c in valid_dests[:10]:
                        print(f"      -> {format_coord(c, r)}")
                    if len(valid_dests) > 10:
                        print(f"      ... and {len(valid_dests) - 10} more")
                else:
                    print("   This piece has no valid moves!")
            
            print()
            print("💡 Try 'hint <row> <col>' to see valid moves for a piece")
            continue
        
        # Move was successful!
        move_num += 1
        state = next_state

        print(f"\n✅ Move executed: {format_coord(source_col, source_row)} → {format_coord(dest_col, dest_row)}")
        if reward > 0:
            print(f"   🎉 Reward: +{float(reward):.0f} points!")
        
        # Check for eliminations
        if 'player_eliminated' in info or jnp.sum(state.player_active) < active_count:
            for i in range(4):
                if not state.player_active[i] and i != current_player:
                    print(f"\n   💀 {PLAYER_COLORS[i]} {PLAYER_NAMES[i]} has been eliminated!")
        
        # Show updated board
        print()
        print(render_board(state, use_unicode=True, use_color=True))
        
        # Check if game is over
        if done:
            print("\n" + "="*70)
            print("GAME OVER!")
            print("="*70)
            print(print_game_summary(state))
            break
    
    print("\nFinal board state:")
    print(render_board(state, use_unicode=True, use_color=True))


if __name__ == "__main__":
    try:
        play_interactive()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
