"""
Precomputed lookup tables for optimized move generation.

These tables are computed once at module import time and provide O(1) lookups
for legal move generation, replacing expensive O(n²) calculations.

Inspired by pgx chess implementation.
"""

import jax.numpy as jnp
import numpy as np
from four_player_chess.constants import (
    BOARD_SIZE, NUM_VALID_SQUARES, 
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
)
from four_player_chess.board import create_valid_square_mask


def _build_coord_mappings():
    """
    Build mappings between (row, col) coordinates and flat valid indices.
    
    Returns:
        coord_to_idx: (14, 14) array mapping (row, col) to valid index (0-159) or -1
        idx_to_coord: (160,) array mapping valid index to flat board index (0-195)
    """
    valid_mask = np.array(create_valid_square_mask())
    coord_to_idx = -np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    idx_to_coord = np.zeros(NUM_VALID_SQUARES, dtype=np.int32)
    
    valid_idx = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if valid_mask[row, col]:
                flat_board_idx = row * BOARD_SIZE + col
                coord_to_idx[row, col] = valid_idx
                idx_to_coord[valid_idx] = flat_board_idx
                valid_idx += 1
    
    return coord_to_idx, idx_to_coord


def _compute_legal_destinations():
    """
    Precompute all possible destinations for each piece type from each square.
    
    This table contains ALL geometrically possible moves, ignoring:
    - Whether pieces are blocking the path
    - Whether destination contains own piece
    - Whether move leaves king in check
    
    These conditions are checked at runtime using other precomputed tables.
    
    Returns:
        LEGAL_DEST: (7, 160, 50) - list of destination indices for each (piece_type, source)
                    Padded with -1 for unused slots
    """
    valid_mask = np.array(create_valid_square_mask())
    coord_to_idx, idx_to_coord = _build_coord_mappings()
    
    # Maximum destinations for each piece type (conservative estimate)
    max_dests = 50  # Queen can reach ~27 squares, use 50 for safety
    LEGAL_DEST = -np.ones((7, NUM_VALID_SQUARES, max_dests), dtype=np.int32)
    
    for src_idx in range(NUM_VALID_SQUARES):
        src_flat = idx_to_coord[src_idx]
        src_row, src_col = src_flat // BOARD_SIZE, src_flat % BOARD_SIZE
        
        # KNIGHT moves (L-shaped)
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        dests = []
        for dr, dc in knight_moves:
            r, c = src_row + dr, src_col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and valid_mask[r, c]:
                dest_idx = coord_to_idx[r, c]
                if dest_idx >= 0:
                    dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[KNIGHT, src_idx, i] = dest
        
        # KING moves (8 adjacent squares)
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        dests = []
        for dr, dc in king_moves:
            r, c = src_row + dr, src_col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and valid_mask[r, c]:
                dest_idx = coord_to_idx[r, c]
                if dest_idx >= 0:
                    dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[KING, src_idx, i] = dest
        
        # BISHOP moves (diagonals)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        dests = []
        for dr, dc in directions:
            for dist in range(1, BOARD_SIZE):
                r, c = src_row + dr * dist, src_col + dc * dist
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    break
                if not valid_mask[r, c]:
                    break
                dest_idx = coord_to_idx[r, c]
                if dest_idx >= 0:
                    dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[BISHOP, src_idx, i] = dest
        
        # ROOK moves (orthogonals)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dests = []
        for dr, dc in directions:
            for dist in range(1, BOARD_SIZE):
                r, c = src_row + dr * dist, src_col + dc * dist
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    break
                if not valid_mask[r, c]:
                    break
                dest_idx = coord_to_idx[r, c]
                if dest_idx >= 0:
                    dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[ROOK, src_idx, i] = dest
        
        # QUEEN moves (diagonals + orthogonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        dests = []
        for dr, dc in directions:
            for dist in range(1, BOARD_SIZE):
                r, c = src_row + dr * dist, src_col + dc * dist
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    break
                if not valid_mask[r, c]:
                    break
                dest_idx = coord_to_idx[r, c]
                if dest_idx >= 0:
                    dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[QUEEN, src_idx, i] = dest
        
        # PAWN moves - special handling needed per player
        # Store all possible pawn destinations (will filter by player direction at runtime)
        dests = []
        for dr in [-2, -1, 1, 2]:
            for dc in [-1, 0, 1]:
                # Skip invalid pawn moves
                if abs(dr) == 2 and dc != 0:
                    continue  # Double move must be straight
                r, c = src_row + dr, src_col + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and valid_mask[r, c]:
                    dest_idx = coord_to_idx[r, c]
                    if dest_idx >= 0:
                        dests.append(dest_idx)
        for i, dest in enumerate(dests):
            LEGAL_DEST[PAWN, src_idx, i] = dest
    
    return jnp.array(LEGAL_DEST, dtype=jnp.int32)


def _compute_between_squares():
    """
    Precompute squares between source and destination for sliding pieces.
    
    For sliding pieces (Bishop, Rook, Queen), we need to check if the path
    is clear. This table stores all intermediate squares.
    
    Returns:
        BETWEEN: (160, 160, 12) - squares between source and dest
                 Padded with -1 if no squares between or not a sliding move
    """
    valid_mask = np.array(create_valid_square_mask())
    coord_to_idx, idx_to_coord = _build_coord_mappings()
    
    BETWEEN = -np.ones((NUM_VALID_SQUARES, NUM_VALID_SQUARES, 12), dtype=np.int32)
    
    for src_idx in range(NUM_VALID_SQUARES):
        src_flat = idx_to_coord[src_idx]
        src_row, src_col = src_flat // BOARD_SIZE, src_flat % BOARD_SIZE
        
        for dest_idx in range(NUM_VALID_SQUARES):
            if src_idx == dest_idx:
                continue
            
            dest_flat = idx_to_coord[dest_idx]
            dest_row, dest_col = dest_flat // BOARD_SIZE, dest_flat % BOARD_SIZE
            
            dr = dest_row - src_row
            dc = dest_col - src_col
            
            # Check if it's a sliding move (straight line or diagonal)
            is_straight = (dr == 0) or (dc == 0)
            is_diagonal = (dr != 0) and (dc != 0) and (abs(dr) == abs(dc))
            
            if not (is_straight or is_diagonal):
                continue
            
            # Compute step direction
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            
            # Collect squares between
            between_list = []
            r, c = src_row + step_r, src_col + step_c
            while (r, c) != (dest_row, dest_col):
                if valid_mask[r, c]:
                    between_idx = coord_to_idx[r, c]
                    if between_idx >= 0:
                        between_list.append(between_idx)
                r += step_r
                c += step_c
                
                # Safety check
                if len(between_list) >= 12:
                    break
            
            # Store in BETWEEN table
            for i, sq_idx in enumerate(between_list):
                BETWEEN[src_idx, dest_idx, i] = sq_idx
    
    return jnp.array(BETWEEN, dtype=jnp.int32)


def _compute_can_move_table():
    """
    Precompute boolean table: can piece P move from square A to square B?
    (Ignoring other pieces on the board)
    
    This provides O(1) lookup for geometric move possibility.
    
    Returns:
        CAN_MOVE: (7, 160, 160) - boolean array
    """
    LEGAL_DEST = _compute_legal_destinations()
    CAN_MOVE = np.zeros((7, NUM_VALID_SQUARES, NUM_VALID_SQUARES), dtype=bool)
    
    for piece_type in range(1, 7):  # Skip EMPTY
        for src_idx in range(NUM_VALID_SQUARES):
            dests = LEGAL_DEST[piece_type, src_idx]
            valid_dests = dests[dests >= 0]
            if len(valid_dests) > 0:
                CAN_MOVE[piece_type, src_idx, valid_dests] = True
    
    return jnp.array(CAN_MOVE, dtype=jnp.bool_)


def _separate_near_far_moves():
    """
    Separate moves into "near" (king/knight) and "far" (sliding pieces).
    
    This optimization allows faster check detection:
    - Near moves: O(1) lookup, no path checking needed
    - Far moves: O(path_length) with path checking
    
    Returns:
        LEGAL_DEST_NEAR: (160, 16) - near moves (king + knight)
        LEGAL_DEST_FAR: (160, 34) - far moves (queen moves minus king moves)
    """
    LEGAL_DEST = _compute_legal_destinations()
    
    LEGAL_DEST_NEAR = -np.ones((NUM_VALID_SQUARES, 16), dtype=np.int32)
    LEGAL_DEST_FAR = -np.ones((NUM_VALID_SQUARES, 34), dtype=np.int32)
    
    for src_idx in range(NUM_VALID_SQUARES):
        # Near moves: union of king and knight moves
        king_dests = LEGAL_DEST[KING, src_idx]
        knight_dests = LEGAL_DEST[KNIGHT, src_idx]
        king_set = set(king_dests[king_dests >= 0].tolist())
        knight_set = set(knight_dests[knight_dests >= 0].tolist())
        near_dests = list(king_set | knight_set)
        
        for i, dest in enumerate(near_dests[:16]):
            LEGAL_DEST_NEAR[src_idx, i] = dest
        
        # Far moves: queen moves minus king moves
        queen_dests = LEGAL_DEST[QUEEN, src_idx]
        queen_set = set(queen_dests[queen_dests >= 0].tolist())
        far_dests = list(queen_set - king_set)
        
        for i, dest in enumerate(far_dests[:34]):
            LEGAL_DEST_FAR[src_idx, i] = dest
    
    return jnp.array(LEGAL_DEST_NEAR, dtype=jnp.int32), jnp.array(LEGAL_DEST_FAR, dtype=jnp.int32)


# ============================================================================
# Generate all precomputed tables at module import time (one-time cost)
# ============================================================================

print("Generating precomputed tables for 4-player chess...")

# Coordinate mappings
COORD_TO_IDX, IDX_TO_COORD = _build_coord_mappings()
COORD_TO_IDX = jnp.array(COORD_TO_IDX, dtype=jnp.int32)
IDX_TO_COORD = jnp.array(IDX_TO_COORD, dtype=jnp.int32)

# Legal destinations for each piece type
LEGAL_DEST_4P = _compute_legal_destinations()

# Squares between source and destination (for path checking)
BETWEEN_4P = _compute_between_squares()

# Quick boolean lookup for geometric move possibility
CAN_MOVE_4P = _compute_can_move_table()

# Separate near and far moves for efficient check detection
LEGAL_DEST_NEAR_4P, LEGAL_DEST_FAR_4P = _separate_near_far_moves()

print(f"✓ Precomputed tables generated:")
print(f"  - LEGAL_DEST_4P: {LEGAL_DEST_4P.shape} ({LEGAL_DEST_4P.nbytes / 1024:.1f} KB)")
print(f"  - BETWEEN_4P: {BETWEEN_4P.shape} ({BETWEEN_4P.nbytes / 1024:.1f} KB)")
print(f"  - CAN_MOVE_4P: {CAN_MOVE_4P.shape} ({CAN_MOVE_4P.nbytes / 1024:.1f} KB)")
print(f"  - LEGAL_DEST_NEAR_4P: {LEGAL_DEST_NEAR_4P.shape} ({LEGAL_DEST_NEAR_4P.nbytes / 1024:.1f} KB)")
print(f"  - LEGAL_DEST_FAR_4P: {LEGAL_DEST_FAR_4P.shape} ({LEGAL_DEST_FAR_4P.nbytes / 1024:.1f} KB)")
total_kb = (LEGAL_DEST_4P.nbytes + BETWEEN_4P.nbytes + CAN_MOVE_4P.nbytes + 
            LEGAL_DEST_NEAR_4P.nbytes + LEGAL_DEST_FAR_4P.nbytes) / 1024
print(f"  Total memory: {total_kb:.1f} KB")