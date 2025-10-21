"""
ASCII rendering for 4-player chess environment.
"""

import jax.numpy as jnp
import chex

from four_player_chess.constants import (
    BOARD_SIZE, EMPTY,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_VALID_SQUARE,
    ASCII_PIECES_SIMPLE, PLAYER_NAMES, PLAYER_COLORS
)


def render_board(state) -> str:
    """
    Render the current board state as ASCII art.
    
    Args:
        state: EnvState object
    
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
    
    # Column labels
    col_labels = "    " + "".join([f"{i:>3}" for i in range(BOARD_SIZE)])
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
                    piece_str = ASCII_PIECES_SIMPLE.get((owner, piece_type), "??")
                    row_str += f" {piece_str}"
        
        lines.append(row_str)
    
    lines.append("   " + "─" * (BOARD_SIZE * 3 + 1))
    lines.append("")
    
    # Show king positions
    lines.append("King Positions:")
    for i in range(4):
        if state.player_active[i]:
            kr, kc = int(state.king_positions[i, 0]), int(state.king_positions[i, 1])
            lines.append(f"  {PLAYER_COLORS[i]} {PLAYER_NAMES[i]}: ({kr}, {kc})")
    
    # Show en passant if any
    ep_row, ep_col = int(state.en_passant_square[0]), int(state.en_passant_square[1])
    if ep_row >= 0 and ep_col >= 0:
        lines.append(f"En Passant Square: ({ep_row}, {ep_col})")
    
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