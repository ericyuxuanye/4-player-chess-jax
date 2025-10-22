# 4-Player Chess JAX Optimization Plan

## Executive Summary

**Current Performance:** 704B FLOPs per step  
**Target Performance:** 1-5B FLOPs per step  
**Expected Speedup:** 140-700x faster  
**Strategy:** Full refactor inspired by pgx chess optimizations

---

## Problem Analysis

### Current Bottleneck (from test_jit.py analysis)

The environment calls `has_legal_moves()` twice per step (for checkmate and stalemate detection):

```
Step → is_checkmate + is_stalemate
         ↓                ↓
    has_legal_moves  has_legal_moves
         ↓                ↓
    get_all_legal_moves_for_player (352B FLOPs each)
         ↓
    Nested vmaps over 14×14×14×14 = 38,416 move possibilities
```

**Key Issues:**
1. Checking 38,416 possibilities when typical positions have ~30-40 legal moves
2. Calling expensive check twice per step for rare events (< 1% of moves)
3. No sparse iteration - iterating over empty squares
4. No precomputed lookup tables
5. No early termination in move generation

---

## Optimization Strategy (Inspired by pgx Chess)

### Phase 1: Precomputed Lookup Tables

**What pgx does:**
- Precomputes all legal destinations for each piece type from each square
- Uses `LEGAL_DEST[piece_type, from_square]` arrays
- Separates near moves (king/knight) from far moves (sliding pieces)
- Precomputes `BETWEEN[from, to]` for path checking

**Adapt for 4-player:**
```python
# Precompute legal destinations (shape: [7, 160, max_dests])
LEGAL_DEST_4P = -jnp.ones((7, 160, 50), dtype=jnp.int32)
LEGAL_DEST_NEAR_4P = -jnp.ones((160, 16), dtype=jnp.int32)  # king/knight
LEGAL_DEST_FAR_4P = -jnp.ones((160, 34), dtype=jnp.int32)   # sliding pieces
BETWEEN_4P = -jnp.ones((160, 160, 12), dtype=jnp.int32)     # path checking
CAN_MOVE_4P = jnp.zeros((7, 160, 160), dtype=jnp.bool_)     # quick lookup
```

**Benefits:** O(1) lookup vs O(n²) calculation

---

### Phase 2: Sparse Piece Iteration

**What pgx does:**
```python
possible_piece_positions = jnp.nonzero(state.board > 0, size=16, fill_value=-1)[0]
actions = jax.vmap(legal_normal_moves)(possible_piece_positions).flatten()
```

**Adapt for 4-player:**
```python
# Only iterate over occupied squares (max 32 pieces on 160 squares)
my_pieces = jnp.nonzero(
    (state.board[:, :, CHANNEL_PIECE_TYPE] > 0) & 
    (state.board[:, :, CHANNEL_OWNER] == player),
    size=16,  # max 16 pieces per player
    fill_value=-1
)[0]

# Generate moves only for actual pieces
moves = jax.vmap(generate_legal_moves_from_square)(my_pieces)
```

**Benefits:** 16 iterations instead of 160 (10x reduction)

---

### Phase 3: Lazy Legal Action Mask

**What pgx does:**
- Only computes legal moves when `legal_action_mask()` is called
- Stored in state for reuse
- Step function updates mask AFTER move execution

**Adapt for 4-player:**
```python
class EnvState(NamedTuple):
    # ... existing fields ...
    legal_action_mask: Array  # NEW: cached legal actions (ACTION_SPACE_SIZE,)

def step(self, key, state, action):
    # Execute move (no legality check here)
    next_state = _execute_move(state, action)
    
    # Update legal action mask for NEXT player
    next_state = next_state._replace(
        legal_action_mask=_compute_legal_action_mask(next_state)
    )
    
    return next_state, obs, reward, done, info
```

**Benefits:** Eliminates has_legal_moves() calls entirely

---

### Phase 4: Efficient Check Detection

**What pgx does:**
```python
def _is_attacked(state, pos):
    # Check near attacks (knight/king/pawn)
    by_minor = jax.vmap(attacked_near)(LEGAL_DEST_NEAR[pos]).any()
    
    # Check far attacks (sliding pieces)
    by_major = jax.vmap(attacked_far)(LEGAL_DEST_FAR[pos]).any()
    
    return by_minor | by_major
```

**Adapt for 4-player:**
```python
def _is_square_attacked(state, pos, by_player):
    # Only check opponent pieces (use sparse iteration)
    opponent_pieces = jnp.nonzero(
        (state.board[:, :, CHANNEL_OWNER] == by_player) &
        (state.board[:, :, CHANNEL_PIECE_TYPE] > 0),
        size=16,
        fill_value=-1
    )[0]
    
    # Check if any opponent piece can attack this square
    return jax.vmap(lambda piece_pos: 
        CAN_MOVE_4P[piece_type, piece_pos, pos] & 
        is_path_clear(state, piece_pos, pos)
    )(opponent_pieces).any()
```

**Benefits:** O(pieces) instead of O(board_size²)

---

### Phase 5: Eliminate Redundant Calculations

**Current approach:**
```python
# Called TWICE per step
in_check = is_in_check(...)
has_moves = has_legal_moves(...)  # 352B FLOPs

is_checkmated = in_check & (~has_moves)
is_stalemated = (~in_check) & (~has_moves)
```

**Optimized approach:**
```python
# legal_action_mask already computed in state
has_moves = state.legal_action_mask.any()
in_check = _is_checked(state)  # cheap - just check king

is_checkmated = in_check & (~has_moves)
is_stalemated = (~in_check) & (~has_moves)
```

**Benefits:** 704B → ~1B FLOPs (700x reduction)

---

### Phase 6: Optimized Move Generation

**What pgx does:**
```python
def _legal_action_mask(state):
    # 1. Generate pseudo-legal moves (fast)
    possible_moves = generate_pseudo_legal_moves(state)
    
    # 2. Filter only moves that don't leave king in check
    legal_moves = jnp.where(
        jax.vmap(is_not_checked)(possible_moves),
        possible_moves,
        -1
    )
    
    return mask
```

**Adapt for 4-player:**
```python
def _compute_legal_action_mask(state):
    current_player = state.current_player
    
    # Get occupied squares for current player (sparse)
    my_pieces = _get_my_piece_positions(state, current_player)
    
    # For each piece, get pseudo-legal destinations
    def get_piece_moves(from_square):
        piece_type = state.board[from_square, CHANNEL_PIECE_TYPE]
        to_squares = LEGAL_DEST_4P[piece_type, from_square]
        
        # Check which destinations are actually legal
        return jax.vmap(lambda to: 
            _is_legal_move(state, from_square, to)
        )(to_squares)
    
    # Generate all moves
    moves = jax.vmap(get_piece_moves)(my_pieces)
    
    # Convert to action mask
    return _moves_to_mask(moves)
```

---

## Implementation Roadmap

### Step 1: Create Precomputed Tables (2-3 hours)
- `constants_optimized.py` with LEGAL_DEST_4P, BETWEEN_4P, etc.
- Generate these at import time (one-time cost)
- Add unit tests to verify correctness

### Step 2: Refactor State and Environment (2-3 hours)
- Add `legal_action_mask` to EnvState
- Modify `reset()` to compute initial mask
- Update `step()` to use cached mask

### Step 3: Optimize Legal Move Generation (3-4 hours)
- Implement `_compute_legal_action_mask()` with sparse iteration
- Use precomputed tables
- Add early termination logic

### Step 4: Optimize Check Detection (2-3 hours)
- Rewrite `is_square_attacked()` with sparse iteration
- Use LEGAL_DEST_NEAR/FAR separation
- Optimize path checking with BETWEEN table

### Step 5: Eliminate Redundant Calls (1 hour)
- Remove `has_legal_moves()` from environment.py
- Use `legal_action_mask.any()` instead
- Simplify checkmate/stalemate detection

### Step 6: Testing and Validation (2-3 hours)
- Unit tests for each component
- Integration tests for full game
- Correctness validation vs old implementation
- Performance benchmarking

**Total Estimated Time:** 12-17 hours

---

## Expected Performance Gains

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Action decoding | 10M FLOPs | 1K FLOPs | 10,000x |
| Legal move generation | 352B FLOPs | 500M FLOPs | 700x |
| Check detection | 50B FLOPs | 50M FLOPs | 1,000x |
| Checkmate/stalemate | 704B FLOPs | 100M FLOPs | 7,000x |
| **Total per step** | **704B FLOPs** | **1-2B FLOPs** | **350-700x** |

---

## Key Design Decisions

### 1. Trade Memory for Speed
- Precomputed tables add ~10MB memory
- But eliminate billions of FLOPs per step
- Worth it for RL training (millions of steps)

### 2. Lazy Evaluation
- Don't check for checkmate unless needed
- Store legal_action_mask in state
- Amortize cost across queries

### 3. Sparse Iteration
- Only iterate over occupied squares
- Use jnp.nonzero with fixed size
- Pad with -1 sentinels

### 4. Separation of Concerns
- Near moves (O(1) checks) separate from far moves (O(path_length))
- King/knight logic different from sliding pieces
- Specialized functions for each case

---

## Risk Mitigation

### Correctness Validation
1. Unit tests for each helper function
2. Property-based testing (game rules must hold)
3. Cross-validation against current implementation
4. Manual game playthrough verification

### Performance Validation
1. Benchmark with `test_jit.py`
2. Profile with JAX's cost_analysis
3. Measure end-to-end training speed
4. Compare FLOPs at each step

### Fallback Plan
If optimization breaks correctness:
1. Keep old implementation as reference
2. Add feature flag to switch implementations
3. Debug specific failing cases
4. Incrementally adopt optimizations

---

## Next Steps

1. **Review and approval of plan**
2. **Create feature branch** (`git checkout -b optimize-performance`)
3. **Implement Phase 1** (precomputed tables)
4. **Test Phase 1** thoroughly
5. **Iterate** through phases 2-6
6. **Final validation** and merge

---

## References

- **pgx Chess implementation:** Highly optimized 2-player chess in JAX
- **AlphaZero paper:** 512 step termination limit
- **JAX best practices:** Precompute, vectorize, avoid python loops
- **Current codebase:** Solid foundation, just needs optimization

---

## Questions for Discussion

1. **Priority:** Is 350-700x speedup worth 12-17 hours of work? (Probably yes!)
2. **Testing:** How thorough should validation be before deployment?
3. **Compatibility:** Any existing code depends on current API?
4. **Deployment:** When can we deploy optimized version?
