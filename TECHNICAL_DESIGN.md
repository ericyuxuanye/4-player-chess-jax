# Technical Design: 4-Player Chess Performance Optimization

## Overview

This document provides implementation-level details for optimizing the 4-player chess environment from 704B FLOPs to 1-5B FLOPs per step.

---

## Phase 1: Precomputed Lookup Tables

### 1.1 Legal Destinations Table

**Purpose:** O(1) lookup of possible destinations for each piece type from each square

```python
# four_player_chess/precompute.py

import jax.numpy as jnp
from four_player_chess.constants import (
    BOARD_SIZE, NUM_VALID_SQUARES, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    RED, BLUE, YELLOW, GREEN
)
from four_player_chess.board import create_valid_square_mask

def _compute_legal_destinations():
    """
    Precompute all legal destinations for each piece type from each valid square.
    
    Returns:
        LEGAL_DEST: shape (7, 160, 50) - for each piece type and source square,
                    list of possible destination squares (padded with -1)
        MAX_DESTS: shape (7,) - maximum destinations for each piece type
    """
    valid_mask = create_valid_square_mask()
    valid_squares = jnp.where(valid_mask.flatten())[0]  # 160 valid squares
    
    # Map from (row, col) to flat valid index
    coord_to_idx = -jnp.ones((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int32)
    for idx, sq in enumerate(valid_squares):
        r, c = sq // BOARD_SIZE, sq % BOARD_SIZE
        coord_to_idx = coord_to_idx.at[r, c].set(idx)
    
    # Compute max possible destinations for each piece type
    max_dests = {
        EMPTY: 0,
        PAWN: 4,     # forward, double forward, 2 captures
        KNIGHT: 8,   # L-shaped moves
        BISHOP: 13,  # diagonals (max on 14x14 cross)
        ROOK: 13,    # orthogonals
        QUEEN: 27,   # diagonals + orthogonals
        KING: 8,     # adjacent squares
    }
    max_dest = max(max_dests.values())
    
    # Initialize table
    LEGAL_DEST = -jnp.ones((7, NUM_VALID_SQUARES, max_dest), dtype=jnp.int32)
    
    # Populate for each piece type
    for piece_type in range(1, 7):  # Skip EMPTY
        for src_idx, src_sq in enumerate(valid_squares):
            src_row, src_col = src_sq // BOARD_SIZE, src_sq % BOARD_SIZE
            dests = []
            
            if piece_type == PAWN:
                # Note: Pawn moves are player-dependent, so we'll need separate logic
                # For now, store all possible pawn destinations (will filter by player)
                for dr in [-2, -1, 1, 2]:
                    for dc in [-1, 0, 1]:
                        if abs(dr) == 2 and dc != 0:
                            continue  # double move is only forward
                        dest_row, dest_col = src_row + dr, src_col + dc
                        if (0 <= dest_row < BOARD_SIZE and 
                            0 <= dest_col < BOARD_SIZE and
                            valid_mask[dest_row, dest_col]):
                            dest_idx = coord_to_idx[dest_row, dest_col]
                            if dest_idx >= 0:
                                dests.append(dest_idx)
            
            elif piece_type == KNIGHT:
                knight_moves = [
                    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)
                ]
                for dr, dc in knight_moves:
                    dest_row, dest_col = src_row + dr, src_col + dc
                    if (0 <= dest_row < BOARD_SIZE and 
                        0 <= dest_col < BOARD_SIZE and
                        valid_mask[dest_row, dest_col]):
                        dest_idx = coord_to_idx[dest_row, dest_col]
                        if dest_idx >= 0:
                            dests.append(dest_idx)
            
            elif piece_type in [BISHOP, ROOK, QUEEN]:
                directions = []
                if piece_type in [BISHOP, QUEEN]:
                    directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                if piece_type in [ROOK, QUEEN]:
                    directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
                
                for dr, dc in directions:
                    for dist in range(1, BOARD_SIZE):
                        dest_row = src_row + dr * dist
                        dest_col = src_col + dc * dist
                        if not (0 <= dest_row < BOARD_SIZE and 
                                0 <= dest_col < BOARD_SIZE):
                            break
                        if not valid_mask[dest_row, dest_col]:
                            break
                        dest_idx = coord_to_idx[dest_row, dest_col]
                        if dest_idx >= 0:
                            dests.append(dest_idx)
            
            elif piece_type == KING:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        dest_row, dest_col = src_row + dr, src_col + dc
                        if (0 <= dest_row < BOARD_SIZE and 
                            0 <= dest_col < BOARD_SIZE and
                            valid_mask[dest_row, dest_col]):
                            dest_idx = coord_to_idx[dest_row, dest_col]
                            if dest_idx >= 0:
                                dests.append(dest_idx)
            
            # Store destinations
            for i, dest in enumerate(dests[:max_dest]):
                LEGAL_DEST = LEGAL_DEST.at[piece_type, src_idx, i].set(dest)
    
    return LEGAL_DEST, jnp.array([max_dests[i] for i in range(7)])


def _compute_between_squares():
    """
    Precompute squares between source and destination for sliding pieces.
    
    Returns:
        BETWEEN: shape (160, 160, 12) - squares between source and dest
                 (padded with -1 for non-sliding paths)
    """
    valid_mask = create_valid_square_mask()
    valid_squares = jnp.where(valid_mask.flatten())[0]
    
    coord_to_idx = -jnp.ones((BOARD_SIZE, BOARD_SIZE), dtype=jnp.int32)
    for idx, sq in enumerate(valid_squares):
        r, c = sq // BOARD_SIZE, sq % BOARD_SIZE
        coord_to_idx = coord_to_idx.at[r, c].set(idx)
    
    BETWEEN = -jnp.ones((NUM_VALID_SQUARES, NUM_VALID_SQUARES, 12), dtype=jnp.int32)
    
    for src_idx, src_sq in enumerate(valid_squares):
        src_row, src_col = src_sq // BOARD_SIZE, src_sq % BOARD_SIZE
        
        for dest_idx, dest_sq in enumerate(valid_squares):
            if src_idx == dest_idx:
                continue
            
            dest_row, dest_col = dest_sq // BOARD_SIZE, dest_sq % BOARD_SIZE
            dr = dest_row - src_row
            dc = dest_col - src_col
            
            # Check if it's a sliding move (same row, col, or diagonal)
            is_sliding = False
            if dr == 0 or dc == 0:  # Rook move
                is_sliding = True
            elif abs(dr) == abs(dc):  # Bishop move
                is_sliding = True
            
            if not is_sliding:
                continue
            
            # Compute step direction
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            
            # Collect squares between
            between_squares = []
            r, c = src_row + step_r, src_col + step_c
            while (r, c) != (dest_row, dest_col):
                if valid_mask[r, c]:
                    between_idx = coord_to_idx[r, c]
                    if between_idx >= 0:
                        between_squares.append(between_idx)
                r += step_r
                c += step_c
            
            # Store in BETWEEN table
            for i, sq_idx in enumerate(between_squares[:12]):
                BETWEEN = BETWEEN.at[src_idx, dest_idx, i].set(sq_idx)
    
    return BETWEEN


def _compute_can_move_table():
    """
    Precompute boolean table: can piece P move from square A to square B?
    (ignoring other pieces on the board)
    
    Returns:
        CAN_MOVE: shape (7, 160, 160) - boolean array
    """
    LEGAL_DEST, _ = _compute_legal_destinations()
    CAN_MOVE = jnp.zeros((7, NUM_VALID_SQUARES, NUM_VALID_SQUARES), dtype=jnp.bool_)
    
    for piece_type in range(1, 7):
        for src_idx in range(NUM_VALID_SQUARES):
            dests = LEGAL_DEST[piece_type, src_idx]
            valid_dests = dests[dests >= 0]
            CAN_MOVE = CAN_MOVE.at[piece_type, src_idx, valid_dests].set(True)
    
    return CAN_MOVE


def _separate_near_far_moves():
    """
    Separate moves into near (king/knight) and far (sliding pieces).
    
    Returns:
        LEGAL_DEST_NEAR: shape (160, 16) - near moves
        LEGAL_DEST_FAR: shape (160, 34) - far moves (queen moves minus king moves)
    """
    LEGAL_DEST, _ = _compute_legal_destinations()
    
    LEGAL_DEST_NEAR = -jnp.ones((NUM_VALID_SQUARES, 16), dtype=jnp.int32)
    LEGAL_DEST_FAR = -jnp.ones((NUM_VALID_SQUARES, 34), dtype=jnp.int32)
    
    for src_idx in range(NUM_VALID_SQUARES):
        # Near moves: union of king and knight moves
        king_dests = set(LEGAL_DEST[KING, src_idx][LEGAL_DEST[KING, src_idx] >= 0].tolist())
        knight_dests = set(LEGAL_DEST[KNIGHT, src_idx][LEGAL_DEST[KNIGHT, src_idx] >= 0].tolist())
        near_dests = list(king_dests | knight_dests)
        
        for i, dest in enumerate(near_dests[:16]):
            LEGAL_DEST_NEAR = LEGAL_DEST_NEAR.at[src_idx, i].set(dest)
        
        # Far moves: queen moves minus king moves
        queen_dests = set(LEGAL_DEST[QUEEN, src_idx][LEGAL_DEST[QUEEN, src_idx] >= 0].tolist())
        far_dests = list(queen_dests - king_dests)
        
        for i, dest in enumerate(far_dests[:34]):
            LEGAL_DEST_FAR = LEGAL_DEST_FAR.at[src_idx, i].set(dest)
    
    return LEGAL_DEST_NEAR, LEGAL_DEST_FAR


# Generate all precomputed tables at module import
LEGAL_DEST_4P, MAX_DESTS_4P = _compute_legal_destinations()
BETWEEN_4P = _compute_between_squares()
CAN_MOVE_4P = _compute_can_move_table()
LEGAL_DEST_NEAR_4P, LEGAL_DEST_FAR_4P = _separate_near_far_moves()
```

---

## Phase 2: Sparse Piece Iteration

### 2.1 Get My Piece Positions

```python
# four_player_chess/sparse_utils.py

import jax.numpy as jnp
from four_player_chess.constants import CHANNEL_PIECE_TYPE, CHANNEL_OWNER, EMPTY

def get_my_piece_positions(board: jnp.ndarray, player: int, max_pieces: int = 16):
    """
    Get positions of all pieces belonging to a player using sparse iteration.
    
    Args:
        board: Board state (14, 14, 4)
        player: Player ID (0-3)
        max_pieces: Maximum pieces per player (default 16)
    
    Returns:
        Array of flat indices (up to max_pieces), padded with -1
    """
    # Flatten board to (196,) for piece type and owner
    piece_type_flat = board[:, :, CHANNEL_PIECE_TYPE].flatten()
    owner_flat = board[:, :, CHANNEL_OWNER].flatten()
    
    # Create mask for my pieces
    is_my_piece = (piece_type_flat > EMPTY) & (owner_flat == player)
    
    # Use jnp.nonzero with fixed size for JAX compatibility
    positions = jnp.nonzero(is_my_piece, size=max_pieces, fill_value=-1)[0]
    
    return positions


def get_opponent_piece_positions(board: jnp.ndarray, player: int, max_pieces: int = 48):
    """
    Get positions of all opponent pieces (up to 3 opponents).
    
    Args:
        board: Board state (14, 14, 4)
        player: Player ID (0-3)
        max_pieces: Maximum opponent pieces (default 48 = 16*3)
    
    Returns:
        Array of flat indices (up to max_pieces), padded with -1
    """
    piece_type_flat = board[:, :, CHANNEL_PIECE_TYPE].flatten()
    owner_flat = board[:, :, CHANNEL_OWNER].flatten()
    
    is_opponent_piece = (piece_type_flat > EMPTY) & (owner_flat != player)
    
    positions = jnp.nonzero(is_opponent_piece, size=max_pieces, fill_value=-1)[0]
    
    return positions
```

---

## Phase 3: Optimized Legal Move Generation

### 3.1 Core Legal Action Mask Function

```python
# four_player_chess/legal_moves_optimized.py

import jax
import jax.numpy as jnp
from jax import lax
from four_player_chess.sparse_utils import get_my_piece_positions
from four_player_chess.precompute import (
    LEGAL_DEST_4P, BETWEEN_4P, CAN_MOVE_4P
)
from four_player_chess.constants import (
    ACTION_SPACE_SIZE, EMPTY, PAWN, CHANNEL_PIECE_TYPE
)

@jax.jit
def compute_legal_action_mask(state):
    """
    Compute legal action mask for current player using sparse iteration.
    
    This is the core optimization that replaces has_legal_moves().
    
    Returns:
        Boolean array of shape (ACTION_SPACE_SIZE,)
    """
    player = state.current_player
    
    # Get my piece positions (sparse iteration - only 16 pieces max)
    my_pieces = get_my_piece_positions(
        state.board, player, max_pieces=16
    )
    
    # For each piece, generate pseudo-legal moves
    def generate_moves_from_piece(piece_pos_flat):
        # Handle -1 padding (no piece)
        is_valid_piece = piece_pos_flat >= 0
        
        # Get piece type
        piece_row = piece_pos_flat // 14
        piece_col = piece_pos_flat % 14
        piece_type = state.board[piece_row, piece_col, CHANNEL_PIECE_TYPE]
        
        # Get legal destinations from lookup table
        dests = LEGAL_DEST_4P[piece_type, piece_pos_flat]
        
        # For each destination, check if move is legal
        def check_destination(dest_idx):
            is_valid_dest = (dest_idx >= 0) & is_valid_piece
            
            # Check if path is clear (for sliding pieces)
            between_squares = BETWEEN_4P[piece_pos_flat, dest_idx]
            is_path_clear = (
                (between_squares < 0) |  # no squares between
                (state.board[between_squares // 14, between_squares % 14, CHANNEL_PIECE_TYPE] == EMPTY)
            ).all()
            
            # Check destination square
            dest_row = dest_idx // 14
            dest_col = dest_idx % 14
            dest_piece = state.board[dest_row, dest_col, CHANNEL_PIECE_TYPE]
            dest_owner = state.board[dest_row, dest_col, CHANNEL_OWNER]
            
            # Can move if destination is empty or contains opponent piece
            can_capture = (dest_piece > EMPTY) & (dest_owner != player)
            is_empty_dest = dest_piece == EMPTY
            
            # Additional checks for pawns
            is_pawn = piece_type == PAWN
            # TODO: Add pawn-specific logic (forward must be empty, diagonal must capture)
            
            # Move is legal if all conditions met
            is_legal = is_valid_dest & is_path_clear & (is_empty_dest | can_capture)
            
            # Final check: doesn't leave king in check
            # TODO: Implement check simulation
            
            return is_legal
        
        # Check all destinations
        legal_dests = jax.vmap(check_destination)(dests)
        
        return legal_dests, piece_pos_flat, dests
    
    # Generate moves for all pieces
    all_moves = jax.vmap(generate_moves_from_piece)(my_pieces)
    
    # Convert to action mask
    # TODO: Map (piece_pos, dest) tuples to action indices
    
    mask = jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_)
    # TODO: Set mask bits for legal actions
    
    return mask
```

---

## Phase 4: Optimized Check Detection

### 4.1 Efficient is_square_attacked

```python
# four_player_chess/check_optimized.py

import jax
import jax.numpy as jnp
from four_player_chess.sparse_utils import get_opponent_piece_positions
from four_player_chess.precompute import (
    LEGAL_DEST_NEAR_4P, LEGAL_DEST_FAR_4P, BETWEEN_4P, CAN_MOVE_4P
)
from four_player_chess.constants import CHANNEL_PIECE_TYPE, EMPTY

@jax.jit
def is_square_attacked_optimized(state, target_pos, by_player):
    """
    Check if a square is attacked by any piece of the given player.
    
    Uses sparse iteration over opponent pieces only.
    
    Args:
        state: Game state
        target_pos: Flat index of target square (0-195)
        by_player: Player ID to check attacks from
    
    Returns:
        Boolean: True if square is attacked
    """
    # Get opponent pieces (sparse - only up to 16 pieces)
    opponent_pieces = get_opponent_piece_positions(
        state.board, player=-1, max_pieces=16  # TODO: fix logic
    )
    # Filter for specific opponent
    opponent_pieces = opponent_pieces[
        state.board[opponent_pieces // 14, opponent_pieces % 14, CHANNEL_OWNER] == by_player
    ]
    
    # Check near attacks (king, knight, pawn)
    def check_near_attack(piece_pos):
        is_valid = piece_pos >= 0
        piece_type = state.board[piece_pos // 14, piece_pos % 14, CHANNEL_PIECE_TYPE]
        
        # Can this piece attack the target using near moves?
        can_attack = CAN_MOVE_4P[piece_type, piece_pos, target_pos]
        
        # Special handling for pawns (only diagonal captures)
        # TODO: Add pawn attack logic
        
        return is_valid & can_attack
    
    # Check far attacks (sliding pieces)
    def check_far_attack(piece_pos):
        is_valid = piece_pos >= 0
        piece_type = state.board[piece_pos // 14, piece_pos % 14, CHANNEL_PIECE_TYPE]
        
        # Can this piece type reach the target?
        can_attack = CAN_MOVE_4P[piece_type, piece_pos, target_pos]
        
        # Check if path is clear
        between_squares = BETWEEN_4P[piece_pos, target_pos]
        is_path_clear = (
            (between_squares < 0) |
            (state.board[between_squares // 14, between_squares % 14, CHANNEL_PIECE_TYPE] == EMPTY)
        ).all()
        
        return is_valid & can_attack & is_path_clear
    
    # Check both near and far attacks
    near_attacks = jax.vmap(check_near_attack)(opponent_pieces)
    far_attacks = jax.vmap(check_far_attack)(opponent_pieces)
    
    return near_attacks.any() | far_attacks.any()


@jax.jit
def is_in_check_optimized(state, king_pos, player):
    """
    Check if player's king is in check.
    
    Args:
        state: Game state
        king_pos: King position [row, col]
        player: Player ID
    
    Returns:
        Boolean: True if in check
    """
    # Convert king position to flat index
    king_flat = king_pos[0] * 14 + king_pos[1]
    
    # Check if attacked by any opponent
    in_check = jnp.bool_(False)
    for opponent in range(4):
        if opponent != player and state.player_active[opponent]:
            in_check |= is_square_attacked_optimized(state, king_flat, opponent)
    
    return in_check
```

---

## Phase 5: Refactored Environment

### 5.1 Updated EnvState

```python
# four_player_chess/state.py (modified)

class EnvState(NamedTuple):
    # ... existing fields ...
    
    # NEW: Cached legal action mask
    legal_action_mask: chex.Array  # shape (ACTION_SPACE_SIZE,)
```

### 5.2 Updated Environment Step

```python
# four_player_chess/environment.py (modified)

def step(self, key, state, action):
    """Execute one step - OPTIMIZED VERSION."""
    
    # Decode action
    source_row, source_col, dest_row, dest_col, promotion_type = decode_action(
        action, self.valid_mask
    )
    
    # Check if action is legal (using cached mask)
    is_legal = state.legal_action_mask[action]
    
    # Execute move
    def valid_move_fn(_):
        new_state = _execute_move_optimized(state, source_row, source_col, dest_row, dest_col, promotion_type)
        
        # Compute legal action mask for NEXT player
        new_state = new_state._replace(
            legal_action_mask=compute_legal_action_mask(new_state)
        )
        
        reward = _calculate_reward(state, new_state)
        return new_state, reward
    
    def invalid_move_fn(_):
        return state, jnp.float32(0.0)
    
    new_state, reward = lax.cond(is_legal, valid_move_fn, invalid_move_fn, None)
    
    # Check if game is over (using cached mask)
    has_legal_moves = new_state.legal_action_mask.any()
    is_checkmated = (~has_legal_moves) & is_in_check_optimized(new_state, ...)
    is_stalemated = (~has_legal_moves) & (~is_in_check_optimized(new_state, ...))
    
    done = is_game_over(new_state.player_active, new_state.move_count, self.params.max_moves)
    
    obs = self._get_observation(new_state)
    info = {'move_valid': is_legal, 'done': done}
    
    return new_state, obs, reward, done, info
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_precompute.py

def test_legal_dest_table():
    """Verify LEGAL_DEST table is correct."""
    # Test knight from center
    # Test rook from corner
    # Test bishop from diagonal
    pass

def test_between_table():
    """Verify BETWEEN table is correct."""
    # Test straight line
    # Test diagonal
    # Test non-sliding moves (should be -1)
    pass

def test_sparse_iteration():
    """Verify sparse piece iteration."""
    # Test with full board
    # Test with sparse board
    # Test with no pieces
    pass
```

### Integration Tests

```python
# tests/test_optimized_environment.py

def test_same_behavior_as_original():
    """Verify optimized version behaves identically to original."""
    # Play same game with both versions
    # Compare states at each step
    pass
```

---

## Performance Benchmarks

```python
# benchmarks/benchmark_optimized.py

import jax
from four_player_chess.environment import FourPlayerChessEnv

env = FourPlayerChessEnv()
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)

# Benchmark step function
lowered = jax.jit(env.step).lower(key, state, action)
cost = lowered.cost_analysis()
print(f"Total FLOPs: {cost['flops']:,.0f}")
```
