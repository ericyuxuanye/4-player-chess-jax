"""
Main environment class for 4-player chess.

Implements a Gymnax-style interface for JAX-based RL training.
"""

import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple, Dict, Any

from four_player_chess.state import EnvState, EnvParams
from four_player_chess.constants import (
    BOARD_SIZE, NUM_PLAYERS, NUM_OBSERVATION_CHANNELS,
    ACTION_SPACE_SIZE, NUM_VALID_SQUARES,
    EMPTY, PAWN, KING, QUEEN, ROOK,
    RED, BLUE, YELLOW, GREEN,
    CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED,
    PROMOTE_QUEEN, PROMOTION_RANKS
)
from four_player_chess.board import (
    create_initial_board, create_valid_square_mask,
    get_initial_king_positions, get_piece_at, set_piece_at, clear_square
)
from four_player_chess.utils import (
    dict_to_jax_array, get_next_active_player, count_active_players,
    decode_action, is_game_over
)
from four_player_chess.rules import (
    is_move_legal, is_checkmate, is_stalemate,
    is_in_check, has_legal_moves
)
from four_player_chess.scoring import (
    calculate_capture_points, update_scores_for_elimination
)
from four_player_chess.rendering import render_board


class FourPlayerChessEnv:
    """
    4-Player Chess Environment with JAX backend.
    
    Supports parallel rollouts via vmap and GPU acceleration.
    """
    
    def __init__(self, params: EnvParams = None):
        """
        Initialize the environment.
        
        Args:
            params: Environment parameters (uses defaults if None)
        """
        self.params = params if params is not None else EnvParams()
        self.valid_mask = create_valid_square_mask()
        
    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters."""
        return EnvParams()
    
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        """
        Reset the environment to initial state.
        
        Args:
            key: JAX random key
        
        Returns:
            Tuple of (initial_state, initial_observation)
        """
        # Create initial board
        board = create_initial_board()
        
        # Initialize state
        state = EnvState(
            board=board,
            current_player=jnp.int32(0),  # Red starts
            player_scores=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
            player_active=jnp.ones(NUM_PLAYERS, dtype=jnp.bool_),
            move_count=jnp.int32(0),
            en_passant_square=jnp.array([-1, -1], dtype=jnp.int32),
            king_positions=get_initial_king_positions(),
            castling_rights=jnp.ones((NUM_PLAYERS, 2), dtype=jnp.bool_),
            last_capture_move=jnp.int32(0),
            promoted_pieces=jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
        )
        
        # Generate initial observation
        obs = self._get_observation(state)
        
        return state, obs
    
    def step(
        self, 
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.Numeric
    ) -> Tuple[EnvState, chex.Array, chex.Numeric, chex.Array, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            key: JAX random key
            state: Current environment state
            action: Action to take (integer encoding move)
        
        Returns:
            Tuple of (next_state, observation, reward, done, info)
        """
        # Decode action
        source_row, source_col, dest_row, dest_col, promotion_type = decode_action(
            action, self.valid_mask
        )
        
        # Execute move and get new state
        next_state, reward, move_valid = self._execute_move(
            state, source_row, source_col, dest_row, dest_col, promotion_type
        )
        
        # Check if game is over
        done = is_game_over(
            next_state.player_active,
            next_state.move_count,
            self.params.max_moves
        )
        
        # Generate observation
        obs = self._get_observation(next_state)
        
        # Build info dict
        info = {
            'current_player': next_state.current_player,
            'move_count': next_state.move_count,
            'active_players': next_state.player_active,
            'scores': next_state.player_scores,
            'move_valid': move_valid,
            'done': done
        }
        
        return next_state, obs, reward, done, info
    
    def _execute_move(
        self,
        state: EnvState,
        source_row: chex.Numeric,
        source_col: chex.Numeric,
        dest_row: chex.Numeric,
        dest_col: chex.Numeric,
        promotion_type: chex.Numeric
    ) -> Tuple[EnvState, chex.Numeric, chex.Array]:
        """
        Execute a move and update state.
        
        Returns:
            Tuple of (new_state, reward, move_was_valid)
        """
        current_player = state.current_player
        
        # Check if move is legal
        move_legal = is_move_legal(
            state.board,
            source_row, source_col,
            dest_row, dest_col,
            current_player,
            state.king_positions[current_player],
            state.player_active,
            self.valid_mask,
            state.en_passant_square
        )
        
        # Get piece being moved
        piece_type, piece_owner = get_piece_at(state.board, source_row, source_col)
        
        # Get captured piece (if any)
        captured_piece, captured_owner = get_piece_at(state.board, dest_row, dest_col)
        
        # Execute the move on the board (will be selected later based on move_legal)
        new_board = state.board
        new_board = set_piece_at(new_board, dest_row, dest_col, piece_type, current_player, has_moved=True)
        new_board = clear_square(new_board, source_row, source_col)
        
        # Handle pawn promotion - use jnp.where instead of if
        # RED and YELLOW move vertically (check row), BLUE and GREEN move horizontally (check col)
        promotion_rank = dict_to_jax_array(PROMOTION_RANKS).at[current_player].get(mode="fill", fill_value=-1)
        
        # Check the correct coordinate based on player orientation
        # RED (0) and YELLOW (2) move vertically → check dest_row
        # BLUE (1) and GREEN (3) move horizontally → check dest_col
        is_vertical_player = (current_player == RED) | (current_player == YELLOW)
        reached_promotion_square = jnp.where(
            is_vertical_player,
            dest_row == promotion_rank,
            dest_col == promotion_rank
        )
        is_pawn_promotion = jnp.logical_and(piece_type == PAWN, reached_promotion_square)
        
        # Apply promotion conditionally
        promoted_piece_type = jnp.where(is_pawn_promotion, QUEEN, piece_type)
        new_board = new_board.at[dest_row, dest_col, CHANNEL_PIECE_TYPE].set(promoted_piece_type)
        
        new_promoted_pieces = jnp.where(
            is_pawn_promotion,
            state.promoted_pieces.at[dest_row, dest_col].set(True),
            state.promoted_pieces
        )
        
        # Update king position if king moved - use jnp.where
        is_king_move = piece_type == KING
        new_king_pos = jnp.array([dest_row, dest_col], dtype=jnp.int32)
        updated_king_positions = state.king_positions.at[current_player].set(new_king_pos)
        new_king_positions = jnp.where(
            is_king_move,
            updated_king_positions,
            state.king_positions
        )
        
        # Calculate reward for capture - use jnp.where
        is_capture = captured_piece != EMPTY
        is_promoted_piece = state.promoted_pieces[dest_row, dest_col]
        capture_points = calculate_capture_points(captured_piece, is_promoted_piece)
        
        capture_reward = jnp.where(is_capture, jnp.float32(capture_points), jnp.float32(0.0))
        new_scores = jnp.where(
            is_capture,
            state.player_scores.at[current_player].add(capture_points),
            state.player_scores
        )
        
        # Update move counter
        new_move_count = state.move_count + 1
        
        # Clear en passant (simplified - not fully implementing en passant for now)
        new_en_passant = jnp.array([-1, -1], dtype=jnp.int32)
        
        # Advance to next player
        next_player = get_next_active_player(current_player, state.player_active)
        
        # Always check for checkmate/stalemate after every move
        # This ensures we never miss an elimination situation

        # Check if next player is in check (relatively cheap operation)
        in_check = is_in_check(
            new_board,
            new_king_positions[next_player],
            next_player,
            state.player_active,
            self.valid_mask
        )

        # Check if next player has legal moves (expensive but necessary)
        has_moves = has_legal_moves(
            new_board,
            next_player,
            new_king_positions[next_player],
            state.player_active,
            self.valid_mask,
            new_en_passant
        )

        # Checkmate = in check AND no legal moves
        is_next_checkmated = in_check & (~has_moves)

        # Stalemate = NOT in check AND no legal moves
        is_next_stalemated = (~in_check) & (~has_moves)
        
        # Handle elimination - use jnp.where for conditional updates
        is_eliminated = jnp.logical_or(is_next_checkmated, is_next_stalemated)
        
        # Conditionally eliminate the next player
        new_player_active = jnp.where(
            is_eliminated,
            state.player_active.at[next_player].set(False),
            state.player_active
        )
        
        # Calculate elimination bonus
        elimination_bonus = jnp.where(
            is_next_checkmated,
            jnp.float32(20.0),
            jnp.where(is_next_stalemated, jnp.float32(10.0), jnp.float32(0.0))
        )
        
        # Update scores for elimination
        num_remaining = count_active_players(new_player_active)
        scores_after_elim = update_scores_for_elimination(
            new_scores,
            current_player,
            next_player,
            is_next_checkmated,
            num_remaining
        )
        new_scores = jnp.where(is_eliminated, scores_after_elim, new_scores)
        
        # Add elimination bonus to reward
        reward = capture_reward + jnp.where(is_eliminated, elimination_bonus, jnp.float32(0.0))
        
        # Find next active player (only if eliminated)
        final_next_player = jnp.where(
            is_eliminated,
            get_next_active_player(next_player, new_player_active),
            next_player
        )
        
        # Create new state
        new_state = EnvState(
            board=new_board,
            current_player=final_next_player,
            player_scores=new_scores,
            player_active=new_player_active,
            move_count=new_move_count,
            en_passant_square=new_en_passant,
            king_positions=new_king_positions,
            castling_rights=state.castling_rights,
            last_capture_move=state.last_capture_move,
            promoted_pieces=new_promoted_pieces
        )
        
        # Use jax.lax.cond to select between valid move state and invalid move state
        # If move is illegal, return unchanged state with 0 reward
        def valid_move_fn(_):
            return new_state, reward, jnp.bool_(True)
        
        def invalid_move_fn(_):
            return state, jnp.float32(0.0), jnp.bool_(False)
        
        return jax.lax.cond(
            move_legal,
            valid_move_fn,
            invalid_move_fn,
            None
        )
    
    def _get_observation(self, state: EnvState) -> chex.Array:
        """
        Generate observation for the current player.
        
        OPTIMIZED: Uses efficient array operations and pre-allocated shapes.
        
        Returns:
            Array of shape (14, 14, NUM_OBSERVATION_CHANNELS)
        """
        # Pre-allocate observation array (avoids expensive concatenations)
        obs = jnp.zeros((BOARD_SIZE, BOARD_SIZE, NUM_OBSERVATION_CHANNELS), dtype=jnp.float32)
        
        # Directly set channels using slicing (much faster than concatenate)
        obs = obs.at[:, :, 0].set(state.board[:, :, CHANNEL_PIECE_TYPE].astype(jnp.float32))
        obs = obs.at[:, :, 1].set(state.board[:, :, CHANNEL_OWNER].astype(jnp.float32))
        obs = obs.at[:, :, 2].set(state.board[:, :, CHANNEL_HAS_MOVED].astype(jnp.float32))
        
        # Additional channels remain as zeros (pre-allocated)
        return obs
    
    def get_legal_actions(self, state: EnvState) -> chex.Array:
        """
        Get a boolean mask of legal actions for the current player.
        
        Returns:
            Boolean array of shape (ACTION_SPACE_SIZE,)
        """
        # For simplicity, return all actions as potentially legal
        # In a full implementation, this would check all possible moves
        return jnp.ones(ACTION_SPACE_SIZE, dtype=jnp.bool_)
    
    def render(self, state: EnvState) -> str:
        """
        Render the current state as ASCII art.
        
        Args:
            state: Current environment state
        
        Returns:
            String representation of the board
        """
        return render_board(state)
    
    @property
    def observation_space(self) -> Tuple[int, int, int]:
        """Return observation space shape."""
        return (BOARD_SIZE, BOARD_SIZE, NUM_OBSERVATION_CHANNELS)
    
    @property
    def action_space(self) -> int:
        """Return action space size."""
        return ACTION_SPACE_SIZE
    
    @property
    def num_players(self) -> int:
        """Return number of players."""
        return NUM_PLAYERS