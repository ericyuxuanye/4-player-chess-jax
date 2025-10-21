# 4-Player Chess JAX Environment - Architecture Plan

## Overview

This document outlines the architecture for a JAX-based 4-player chess environment that supports parallel rollouts for reinforcement learning training.

## Board Structure

### Physical Layout
```
       Yellow (Y)
          ↓
      [3x8 extension]
    ┌─────────────┐
    │             │
    │   8x8 core  │  ← Blue (B)
Green│             │
(G) →│   center    │
    │             │
    │             │
    └─────────────┘
      [3x8 extension]
          ↑
        Red (R)
```

### Coordinate System
- **Board dimensions**: 14x14 grid with corners masked
- **160 valid squares** total
- **Coordinate mapping**:
  - Rows 0-13, Cols 0-13 (0-indexed)
  - Invalid corner regions masked with sentinel value (-1)
  - Each player starts in their 3x8 extension area

### Players
- **Red** (Player 0): Bottom (rows 11-13, cols 3-10)
- **Blue** (Player 1): Right (rows 3-10, cols 11-13)  
- **Yellow** (Player 2): Top (rows 0-2, cols 3-10)
- **Green** (Player 3): Left (rows 3-10, cols 0-2)

## State Representation

### Board State (JAX Array)
```python
# Shape: (14, 14, 4)
# Channels:
#   0: Piece type (0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king)
#   1: Piece owner (0=Red, 1=Blue, 2=Yellow, 3=Green, -1=inactive/eliminated)
#   2: Has moved flag (for castling/pawn double move)
#   3: Valid square mask (1=valid, 0=corner/invalid)
```

### Game State Structure
```python
@chex.dataclass
class EnvState:
    board: jnp.ndarray          # (14, 14, 4) board state
    current_player: jnp.int32   # 0-3 (Red, Blue, Yellow, Green)
    player_scores: jnp.ndarray  # (4,) points for each player
    player_active: jnp.ndarray  # (4,) bool, True if player still in game
    move_count: jnp.int32       # Total moves made
    en_passant_square: jnp.ndarray  # (2,) row, col or [-1, -1]
    king_positions: jnp.ndarray # (4, 2) positions of each king
    castling_rights: jnp.ndarray # (4, 2) queenside/kingside per player
    last_capture_move: jnp.int32 # Moves since last capture (50-move rule)
```

### Observation Space
```python
# For each player: (14, 14, K) where K includes:
# - 7 channels for own pieces (one-hot piece types)
# - 7 channels for each opponent (3 opponents × 7 = 21)
# - 1 channel for valid squares
# - 1 channel for en passant square
# - 4 channels for player active flags
# - 1 channel for current player indicator
# Total: 7 + 21 + 1 + 1 + 4 + 1 = 35 channels
```

## Action Space

### Action Encoding
Each action is encoded as a single integer that combines:
1. **Source square** (160 possibilities)
2. **Destination square** (160 possibilities)  
3. **Promotion piece** (4 options: Queen, Rook, Bishop, Knight)

```python
# Action space size: 160 × 160 × 4 = 102,400 possible actions
# Most will be invalid at any given state

action_id = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
```

### Legal Move Generation
- Pre-compute piece movement patterns (knights, bishops, etc.)
- Use JAX's JIT compilation for fast legal move checking
- Return boolean mask of legal actions: `(102400,)`

## Core Algorithms

### 1. Move Validation
```python
def is_legal_move(state, action) -> bool:
    # 1. Extract source, dest, promotion from action
    # 2. Check if source has current player's piece
    # 3. Check basic piece movement rules
    # 4. Check path is clear (for sliding pieces)
    # 5. Check destination (empty or opponent piece)
    # 6. Validate special moves (castling, en passant)
    # 7. Check that move doesn't leave king in check
    # 8. Return validity
```

### 2. Piece Movement Patterns
```python
# Precomputed move offsets for each piece type
PAWN_MOVES = {player_id: direction_offsets}
KNIGHT_MOVES = [(±1, ±2), (±2, ±1)]  # 8 L-shapes
BISHOP_MOVES = diagonal_rays  # 4 directions
ROOK_MOVES = orthogonal_rays  # 4 directions
QUEEN_MOVES = BISHOP_MOVES + ROOK_MOVES
KING_MOVES = 8 adjacent squares
```

### 3. Check Detection
```python
def is_in_check(state, player_id) -> bool:
    king_pos = state.king_positions[player_id]
    # For each active opponent:
    #   For each opponent piece:
    #     If piece can attack king_pos: return True
    # return False
```

### 4. Checkmate/Stalemate Detection
```python
def is_checkmate(state) -> bool:
    if not is_in_check(state, state.current_player):
        return False
    return not has_legal_moves(state)

def is_stalemate(state) -> bool:
    if is_in_check(state, state.current_player):
        return False
    return not has_legal_moves(state)
```

### 5. Scoring System (Free-for-All)
```python
# Points awarded for:
# - Checkmating opponent: +20
# - Stalemating self: +20  
# - Opponent stalemated: +10 per remaining player
# - Checking 2 kings simultaneously: +5
# - Checking 3 kings simultaneously: +5
# - Checking 2 kings with non-queen: +5
# - Checking 3 kings with non-queen: +20
# - Capturing pieces: +1 pawn, +3 knight/bishop, +5 rook, +9 queen
# - Promoted queen only gives +1 when captured
```

## Environment API (Gymnax-compatible)

### Core Methods
```python
class FourPlayerChessEnv:
    def reset(self, key: PRNGKey) -> Tuple[EnvState, Observation]:
        """Initialize a new game"""
        
    def step(self, key: PRNGKey, state: EnvState, action: int) -> Tuple[EnvState, Observation, Reward, Done, Info]:
        """Execute one move"""
        
    def get_legal_actions(self, state: EnvState) -> jnp.ndarray:
        """Return boolean mask of legal actions"""
        
    def render(self, state: EnvState) -> str:
        """ASCII visualization of current state"""
```

### Parallel Rollouts
```python
# JAX vmap enables parallel environment execution
reset_fn = jax.vmap(env.reset)
step_fn = jax.vmap(env.step)

# Run 1000 parallel games
keys = jax.random.split(key, 1000)
states, obs = reset_fn(keys)
```

## Performance Optimizations

### 1. Precomputation
- Attack/defense tables for each piece type
- Valid move patterns stored as sparse matrices
- Board validity masks cached

### 2. JAX-specific
- All functions pure (no side effects)
- Extensive use of `@jax.jit` for compilation
- `jnp.where` for conditional logic (avoid Python if/else)
- Vectorized operations over loops

### 3. Move Generation
- Use boolean masks for filtering
- Lazy evaluation where possible
- Cache legal move masks when feasible

## File Structure

```
four_player_chess/
├── __init__.py
├── constants.py          # Board dimensions, piece types, etc.
├── state.py              # EnvState dataclass definition
├── board.py              # Board representation and utilities
├── pieces.py             # Piece movement logic
├── rules.py              # Check, checkmate, special moves
├── scoring.py            # Point system implementation
├── environment.py        # Main Gymnax environment class
├── rendering.py          # ASCII visualization
└── utils.py              # Helper functions

tests/
├── test_board.py
├── test_pieces.py
├── test_rules.py
├── test_environment.py
└── test_integration.py

examples/
├── play_game.py          # Test script with ASCII rendering
├── random_rollout.py     # Parallel random games
└── benchmark.py          # Performance testing
```

## Testing Strategy

### Unit Tests
1. **Board representation**: Valid squares, coordinate mapping
2. **Piece movements**: Each piece type's legal moves
3. **Special moves**: Castling, en passant, pawn promotion
4. **Check detection**: Various check scenarios
5. **Checkmate detection**: True/false positives
6. **Scoring**: Point calculations for various actions
7. **Player elimination**: Piece deactivation

### Integration Tests
1. **Full game simulation**: Play to completion
2. **Parallel rollouts**: 1000+ simultaneous games
3. **Edge cases**: Insufficient material, threefold repetition
4. **Performance**: Benchmark JIT compilation times

### Test Game Scenarios
- Opening moves from standard position
- Endgame positions (few pieces remaining)
- Complex tactical positions
- Three-way checks
- Player elimination sequences

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
- State representation
- Board utilities
- Basic piece movement
- Testing framework

### Phase 2: Game Logic (Days 3-4)
- All piece movements
- Special moves
- Check/checkmate detection
- Legal move generation

### Phase 3: Scoring & Elimination (Day 5)
- Point system
- Player elimination
- Game termination conditions

### Phase 4: Environment API (Days 6-7)
- Gymnax interface
- Observation generation
- Parallel rollouts
- ASCII rendering

### Phase 5: Testing & Optimization (Days 8-9)
- Comprehensive tests
- Performance optimization
- Documentation
- Example scripts

### Phase 6: Polish (Day 10)
- Code review
- Additional examples
- Final documentation
- Benchmarking

## Key Design Decisions

### 1. Board Representation
**Decision**: Use dense 14×14 array with masked corners
**Rationale**: Simpler indexing, better for GPU parallelization than sparse representation

### 2. Action Encoding
**Decision**: Flat integer space (102,400 actions)
**Rationale**: Standard for RL, allows masking invalid actions, compatible with policy networks

### 3. Player Elimination
**Decision**: Mark pieces as inactive but keep on board
**Rationale**: Simplifies state management, avoids board recomputation

### 4. Observation Format
**Decision**: 35-channel tensor per player
**Rationale**: Rich representation for neural networks, includes all relevant game state

### 5. Scoring Integration
**Decision**: Track scores in state, return as info dict
**Rationale**: Allows both score-based and checkmate-based training objectives

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.9"
jax = "^0.4.20"
jaxlib = "^0.4.20"
chex = "^0.1.85"
gymnax = "^0.0.6"
numpy = "^1.24.0"
```

## Next Steps

1. Review this architecture with the user
2. Get approval on design decisions
3. Begin implementation in Code mode
4. Iterate based on testing feedback

---

**Note**: This is a complex project requiring careful implementation of chess rules in a purely functional, JAX-compatible way. Estimated total implementation time: 10-15 days for a single developer.