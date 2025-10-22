# 4-Player Chess JAX Environment

A JAX-native implementation written by Claude of a 4-player chess environment that supports parallel rollouts for end-to-end reinforcement learning training pipelines.

## Features

- ✅ **JAX-Native**: Fully implemented in JAX for GPU acceleration
- ✅ **Parallel Rollouts**: Supports batched execution via `vmap`
- ✅ **Gymnax-Compatible**: Standard RL environment interface
- ✅ **4-Player Chess Rules**: Based on chess.com's free-for-all variant
- ✅ **ASCII Rendering**: Visual debugging and gameplay
- ✅ **Scoring System**: Comprehensive point-based scoring
- ✅ **Pure Functions**: All logic is JIT-compilable

## Installation

```bash
# Clone the repository
git clone https://github.com/ericyuxuanye/four-player-chess-jax.git
cd four-player-chess-jax

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `jax>=0.4.20`
- `jaxlib>=0.4.20`
- `chex>=0.1.85`
- `numpy>=1.24.0`

## Quick Start

### Basic Usage

```python
import jax
from jax import random
from four_player_chess import FourPlayerChessEnv

# Initialize environment
env = FourPlayerChessEnv()

# Reset environment
key = random.PRNGKey(0)
state, obs = env.reset(key)

# Take a step
key, subkey = random.split(key)
action = 0  # Your action here
next_state, next_obs, reward, done, info = env.step(subkey, state, action)

# Render the board
print(env.render(state))
```

### Interactive Play

Run the interactive play script to manually input moves:
```bash
python examples/interactive_play.py
```

### Playing a Test Game

Run the example scripts to see the environment in action:

```bash
# 🎮 INTERACTIVE MODE - Play with manual move input (NEW!)
python examples/interactive_play.py

# Automatic random game (50 moves, render every move)
python examples/play_game.py --mode auto --max-moves 50 --render-every 1

# Step-by-step mode with random moves
python examples/play_game.py --mode step

# Fast automatic game (render every 5 moves)
python examples/play_game.py --mode auto --max-moves 100 --render-every 5
```

**Interactive Mode Instructions:**
- Enter moves as: `source_row,source_col,dest_row,dest_col`
- Use `hint <row> <col>` to see valid moves for a piece
- Type `help` for full instructions

## Architecture

### Board Structure

The board is a **14×14 cross-shaped grid** with 160 valid squares:

```
       Yellow (Y)
          ↓
      [3×8 extension]
     ┌─────────────┐
     │             │
     │   8×8 core  │  ← Blue (B)
Green│             │
(G) →│   center    │
     │             │
     │             │
     └─────────────┘
      [3×8 extension]
          ↑
        Red (R)
```

- **Players**: Red (0), Blue (1), Yellow (2), Green (3)
- **Turn Order**: Sequential clockwise (Red → Blue → Yellow → Green → Red...)
- **Starting Position**: Each player has standard chess pieces in their extension

### State Representation

```python
EnvState(
    board: (14, 14, 4),              # [piece_type, owner, has_moved, valid_square]
    current_player: int,             # 0-3 (whose turn it is)
    player_scores: (4,),             # Points for each player
    player_active: (4,),             # Boolean, True if player still in game
    move_count: int,                 # Total moves made
    en_passant_square: (2,),         # En passant target or [-1, -1]
    king_positions: (4, 2),          # King positions [row, col]
    castling_rights: (4, 2),         # Castling rights per player
    last_capture_move: int,          # For 50-move rule
    promoted_pieces: (14, 14)        # Track promoted queens
)
```

### Action Space

Actions are encoded as single integers:
- **Action Space Size**: 102,400 (160 squares × 160 destinations × 4 promotion types)
- **Encoding**: `action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type`

### Observation Space

Observations are **ego-centric** (from current player's perspective):
- **Shape**: (14, 14, 35)
- **Channels**: Piece types, player indicators, game state flags

## Game Rules

### Standard Chess Rules Apply

- Pieces move according to standard chess rules
- Pawns promote on reaching the 8th rank (opponent's territory)
- Check, checkmate, and stalemate detection

### 4-Player Specific Rules

**Elimination**:
- Players are eliminated via checkmate or stalemate
- Eliminated players' pieces become inactive (grayed out)
- Game continues until ≤1 player remains

**Scoring** (Free-for-All Mode):
- **Capturing pieces**: +1 (pawn), +3 (knight/bishop), +5 (rook), +9 (queen)
- **Checkmate opponent**: +20 points
- **Stalemate self**: +20 points
- **Stalemate opponent**: +10 points × remaining players
- **Promoted queens**: Only worth +1 when captured

**Turn Order**:
- Sequential turns (one player at a time)
- Automatically skips eliminated players
- Each `step()` advances to next active player

## API Reference

### Environment Methods

#### `reset(key: PRNGKey) -> (EnvState, Observation)`
Initialize a new game.

```python
state, obs = env.reset(key)
```

#### `step(key: PRNGKey, state: EnvState, action: int) -> (EnvState, Observation, Reward, Done, Info)`
Execute one move.

```python
next_state, obs, reward, done, info = env.step(key, state, action)
```

#### `render(state: EnvState) -> str`
Get ASCII representation of the board.

```python
print(env.render(state))
```

### Parallel Rollouts

Execute multiple games simultaneously:

```python
import jax

# Vectorize environment operations
batch_size = 1024
keys = jax.random.split(key, batch_size)

# Reset all environments in parallel
reset_fn = jax.vmap(env.reset)
states, obs = reset_fn(keys)

# Step all environments in parallel
step_fn = jax.vmap(env.step, in_axes=(0, 0, 0))
actions = jax.random.randint(keys, (batch_size,), 0, env.action_space)
next_states, next_obs, rewards, dones, infos = step_fn(keys, states, actions)
```

## File Structure

```
four_player_chess/
├── __init__.py           # Package initialization
├── constants.py          # Game constants and piece definitions
├── state.py              # State dataclass definitions
├── board.py              # Board representation and utilities
├── pieces.py             # Piece movement logic
├── rules.py              # Check, checkmate, legal move validation
├── scoring.py            # Scoring system implementation
├── environment.py        # Main environment class
├── rendering.py          # ASCII visualization
└── utils.py              # Helper functions

examples/
└── play_game.py          # Test script with interactive gameplay

docs/
├── ARCHITECTURE.md       # Detailed architecture documentation
├── API_DESIGN.md         # API design decisions
└── IMPLEMENTATION_ROADMAP.md  # Development roadmap
```

## Current Status

### ✅ Implemented

- Core board representation and coordinate system
- All standard piece movement rules (pawn, knight, bishop, rook, queen, king)
- Check and checkmate detection
- Stalemate detection
- Player elimination handling
- Basic scoring system
- Sequential turn-based play
- ASCII rendering
- Test script for gameplay

### 🚧 Partially Implemented

- En passant (structure in place, logic simplified)
- Castling (structure in place, not fully validated)
- Comprehensive legal move generation (simplified for performance)
- Full observation space (currently simplified)

### 📝 Future Enhancements

- Complete en passant implementation
- Full castling validation
- Optimized legal move caching
- Rich observation encoding for neural networks
- Multi-check bonus scoring
- Threefold repetition detection
- Draw by insufficient material

## Performance

The environment is designed for high-performance RL training:

- **JIT Compilation**: All core functions use `@jax.jit`
- **GPU Acceleration**: Native JAX arrays enable GPU computation
- **Vectorization**: Supports batched parallel execution
- **Pure Functions**: No side effects, enabling safe parallelization

## Development

### Running Tests

```bash
# Test basic gameplay
python examples/play_game.py --mode auto --max-moves 20

# Interactive testing
python examples/play_game.py --mode step
```

### Contributing

This is a functional prototype. Contributions welcome for:
- Performance optimizations
- Complete special move implementations
- Enhanced observations for RL
- Comprehensive test suite
- Additional game variants

## References

- [Chess.com 4-Player Chess Rules](https://www.chess.com/terms/4-player-chess)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Gymnax](https://github.com/RobertTLange/gymnax)
- [Pgx](https://github.com/sotetsuk/pgx)