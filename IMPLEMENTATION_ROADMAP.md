# Implementation Roadmap

## Approved Design Summary

✅ **Single-agent interface** with `state.current_player` tracking whose turn it is
✅ **Sequential turn-based play** (Player 0 → 1 → 2 → 3 → 0...)
✅ **Ego-centric observations** (always from current player's perspective)
✅ **JAX-native implementation** for GPU acceleration and parallel rollouts
✅ **Gymnax-compatible API** for RL training pipelines

## File Structure to Implement

```
four_player_chess/
├── __init__.py
├── constants.py          # Board dimensions, piece types, scoring
├── state.py              # EnvState dataclass
├── board.py              # Board utilities and coordinate system
├── pieces.py             # Piece movement logic
├── rules.py              # Check, checkmate, legal moves
├── scoring.py            # Point system
├── environment.py        # Main Gymnax environment
├── rendering.py          # ASCII visualization
└── utils.py              # Helper functions

examples/
└── play_game.py          # Test script with ASCII rendering
```

## Implementation Order

### Phase 1: Foundation
1. **constants.py** - Define all constants (piece types, board dimensions, etc.)
2. **state.py** - Create EnvState dataclass
3. **board.py** - Board representation, valid squares, coordinate mapping
4. **utils.py** - Helper functions for array manipulation

### Phase 2: Game Logic
5. **pieces.py** - Movement patterns for all piece types
6. **rules.py** - Check detection, legal move validation
7. **scoring.py** - Point calculation system

### Phase 3: Environment
8. **environment.py** - Main Gymnax interface (reset, step, legal actions)
9. **rendering.py** - ASCII visualization

### Phase 4: Testing
10. **play_game.py** - Interactive test script
11. Verify all game mechanics work correctly

## Key Implementation Notes

- All functions must be pure (JAX requirement)
- Use `jnp.where()` instead of Python conditionals
- Extensive use of `@jax.jit` for performance
- Board uses dense 14×14 array with masked corners
- Action space: 160 × 160 × 4 = 102,400 actions
- Observations: 35-channel (14, 14, 35) tensor

## Success Criteria

✅ Environment can be initialized and reset
✅ Legal moves are correctly generated
✅ Pieces move according to chess rules
✅ Check/checkmate detection works
✅ Player elimination handled properly
✅ Scoring system implemented
✅ ASCII rendering shows game state
✅ Can run parallel rollouts with vmap
✅ Test script plays complete games

---

Ready to begin implementation in Code mode!