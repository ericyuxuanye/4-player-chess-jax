"""
Rendering utilities for 4-player chess environment.

Defaults to simple ASCII for wide compatibility, but supports
Unicode chess pieces and ANSI colors when requested.
"""

import jax.numpy as jnp
import chex

from four_player_chess.constants import (
    BOARD_SIZE, EMPTY,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_VALID_SQUARE,
    ASCII_PIECES_SIMPLE, ASCII_PIECES,
    PLAYER_NAMES, PLAYER_COLORS
)


def render_board(state, *, use_unicode: bool = False, use_color: bool = False) -> str:
    """
    Render the current board state as ASCII art.
    
    Args:
        state: EnvState object
        use_unicode: When True, render with Unicode chess glyphs.
        use_color: When True, colorize pieces with ANSI colors.
    
    Returns:
        String representation of the board
    """
    board = state.board
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append(f"4-Player Chess - Move {int(state.move_count)}")
    lines.append(f"Current Player: {PLAYER_COLORS[int(state.current_player)]} {PLAYER_NAMES[int(state.current_player)]}")
    lines.append("=" * 60)
    
    # Show scores
    score_line = "Scores: "
    for i in range(4):
        active = "✓" if state.player_active[i] else "✗"
        score_line += f"{PLAYER_COLORS[i]} {int(state.player_scores[i])}{active}  "
    lines.append(score_line)
    lines.append("")
    
    # Column labels (0-2, a-h for 3-10, 11-13)
    def col_label(i):
        if 3 <= i <= 10:
            return chr(ord('a') + i - 3)  # Map 3-10 to a-h
        return str(i)

    col_labels = "    " + "".join([f"{col_label(i):^3}" for i in range(BOARD_SIZE)])
    lines.append(col_labels)
    lines.append("   " + "─" * (BOARD_SIZE * 3 + 1))
    
    # Board rows
    for row in range(BOARD_SIZE):
        row_str = f"{row:2d} │"
        
        for col in range(BOARD_SIZE):
            is_valid = board[row, col, CHANNEL_VALID_SQUARE] > 0
            
            if not is_valid:
                row_str += "   "
            else:
                piece_type = int(board[row, col, CHANNEL_PIECE_TYPE])
                owner = int(board[row, col, CHANNEL_OWNER])
                
                if piece_type == EMPTY:
                    # Show empty square
                    row_str += " · "
                else:
                    # Show piece
                    if use_unicode:
                        glyph = ASCII_PIECES.get((owner, piece_type), "?")
                        if use_color:
                            # Keep fixed width: space + colored glyph + space
                            color_map = {
                                0: "\x1b[31m",  # Red
                                1: "\x1b[34m",  # Blue
                                2: "\x1b[33m",  # Yellow
                                3: "\x1b[32m",  # Green
                            }
                            color = color_map.get(owner, "")
                            reset = "\x1b[0m"
                            piece_cell = f" {color}{glyph}{reset} "
                        else:
                            # No color: still ensure 3-wide cell
                            piece_cell = f" {glyph} "
                    else:
                        # Simple ASCII two-char codes are inherently width 2
                        base_piece = ASCII_PIECES_SIMPLE.get((owner, piece_type), "??")
                        if use_color:
                            color_map = {
                                0: "\x1b[31m",
                                1: "\x1b[34m",
                                2: "\x1b[33m",
                                3: "\x1b[32m",
                            }
                            color = color_map.get(owner, "")
                            reset = "\x1b[0m"
                            piece_cell = f" {color}{base_piece}{reset}"
                        else:
                            piece_cell = f" {base_piece}"

                    row_str += piece_cell
        
        lines.append(row_str)
    
    lines.append("   " + "─" * (BOARD_SIZE * 3 + 1))
    lines.append("")
    
    # Show king positions
    def format_col(c):
        if 3 <= c <= 10:
            return chr(ord('a') + c - 3)
        return str(c)

    lines.append("King Positions:")
    for i in range(4):
        if state.player_active[i]:
            kr, kc = int(state.king_positions[i, 0]), int(state.king_positions[i, 1])
            lines.append(f"  {PLAYER_COLORS[i]} {PLAYER_NAMES[i]}: ({format_col(kc)}{kr})")
    
    # Show en passant if any
    ep_row, ep_col = int(state.en_passant_square[0]), int(state.en_passant_square[1])
    if ep_row >= 0 and ep_col >= 0:
        lines.append(f"En Passant Square: {format_col(ep_col)}{ep_row}")
    
    lines.append("")
    return "\n".join(lines)


def render_move(source_row: int, source_col: int, dest_row: int, dest_col: int, promotion: int = -1) -> str:
    """
    Render a move in human-readable format.
    
    Args:
        source_row: Source square row
        source_col: Source square column
        dest_row: Destination square row
        dest_col: Destination square column
        promotion: Promotion piece type (-1 if none)
    
    Returns:
        String representation like "e2-e4" or "e7-e8=Q"
    """
    move_str = f"({source_row},{source_col})->({dest_row},{dest_col})"
    
    if promotion >= 0:
        promo_names = {0: "Q", 1: "R", 2: "B", 3: "N"}
        move_str += f"={promo_names.get(promotion, '?')}"
    
    return move_str


def print_game_summary(state) -> str:
    """
    Print a summary of the game state.
    
    Args:
        state: EnvState object
    
    Returns:
        String summary
    """
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("GAME SUMMARY")
    lines.append("=" * 60)
    
    # Final scores
    lines.append("\nFinal Scores:")
    scores_with_players = [(int(state.player_scores[i]), i) for i in range(4)]
    scores_with_players.sort(reverse=True)
    
    for rank, (score, player) in enumerate(scores_with_players, 1):
        status = "Active" if state.player_active[player] else "Eliminated"
        lines.append(f"  {rank}. {PLAYER_COLORS[player]} {PLAYER_NAMES[player]}: {score} points ({status})")
    
    # Game stats
    lines.append(f"\nTotal Moves: {int(state.move_count)}")
    lines.append(f"Active Players: {int(jnp.sum(state.player_active))}")
    
    lines.append("=" * 60 + "\n")
    return "\n".join(lines)