"""
State representation for 4-player chess environment.
"""

from typing import NamedTuple
import jax.numpy as jnp
import chex


class EnvState(NamedTuple):
    """
    Complete game state for 4-player chess.
    
    All arrays are JAX arrays for efficient GPU computation.
    """
    
    # Board state: (14, 14, 4)
    # Channels: [piece_type, owner, has_moved, valid_square]
    board: chex.Array
    
    # Current player (0=Red, 1=Blue, 2=Yellow, 3=Green)
    current_player: chex.Numeric
    
    # Score for each player (4,)
    player_scores: chex.Array
    
    # Whether each player is still active (4,)
    player_active: chex.Array
    
    # Total moves made in the game
    move_count: chex.Numeric
    
    # En passant target square: (2,) with [row, col] or [-1, -1] if none
    en_passant_square: chex.Array
    
    # King positions for each player: (4, 2) with [row, col]
    king_positions: chex.Array
    
    # Castling rights: (4, 2) with [queenside, kingside] for each player
    castling_rights: chex.Array
    
    # Moves since last capture or pawn move (for 50-move rule)
    last_capture_move: chex.Numeric
    
    # Track if pieces are promoted queens (for scoring): (14, 14) bool array
    promoted_pieces: chex.Array
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        active_players = jnp.where(self.player_active)[0]
        return (
            f"EnvState(\n"
            f"  current_player={int(self.current_player)},\n"
            f"  move_count={int(self.move_count)},\n"
            f"  active_players={active_players.tolist()},\n"
            f"  scores={self.player_scores.tolist()}\n"
            f")"
        )


class EnvParams(NamedTuple):
    """
    Environment parameters (hyperparameters for the game).
    These can be modified to create variants of the game.
    """

    # Maximum number of moves before declaring a draw
    max_moves: int = 500

    # Number of moves without progress (capture/pawn move) before draw
    no_progress_limit: int = 50

    # Whether to use the full scoring system or simplified
    use_full_scoring: bool = True

    # Whether to allow castling
    allow_castling: bool = True

    # Whether to allow en passant
    allow_en_passant: bool = True

    # Whether to use player-relative (ego-centric) coordinates for actions
    # When True, actions are interpreted in each player's coordinate frame
    use_relative_coordinates: bool = False