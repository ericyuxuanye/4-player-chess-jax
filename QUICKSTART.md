# Quick Start Guide - 4-Player Chess JAX Environment

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Your First Game

```bash
# Run a test game with ASCII rendering
python examples/play_game.py --mode auto --max-moves 20
```

This will play 20 moves of a game with random actions and display the board after each move.

## Basic Usage Example

```python
import jax
from jax import random
from four_player_chess import FourPlayerChessEnv

# Create environment
env = FourPlayerChessEnv()

# Initialize with random seed
key = random.PRNGKey(0)
state, obs = env.reset(key)

# Print initial board
print(env.render(state))

# Play one move (random action for demo)
key, subkey = random.split(key)
action = random.randint(subkey, (), 0, env.action_space)
next_state, obs, reward, done, info = env.step(subkey, state, action)

# Check results
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Current player: {info['current_player']}")
```

## Test Parallel Rollouts

```bash
# Run 1000 parallel games for 50 steps each
python examples/parallel_rollout_demo.py --mode benchmark --num-games 1000 --num-steps 50

# Compare sequential vs parallel performance
python examples/parallel_rollout_demo.py --mode compare --num-games 100
```

## Interactive Play

### Full Manual Control (NEW!)

```bash
# Play with manual move input
python examples/interactive_play.py
```

Enter moves as: `source_row,source_col,dest_row,dest_col`
Example: `12,3,10,3` moves piece from (12,3) to (10,3)

Commands:
- `help` - Show help and instructions
- `hint <row> <col>` - Show valid moves for a piece
- `show` - Redisplay current board
- `quit` - Exit game

### Step-by-Step with Random Moves

```bash
# Step-by-step mode with random moves
python examples/play_game.py --mode step
```

Press Enter after each move to advance the game.

## Environment Properties

```python
# Get environment information
print(f"Observation space: {env.observation_space}")  # (14, 14, 35)
print(f"Action space size: {env.action_space}")       # 102,400
print(f"Number of players: {env.num_players}")        # 4
```

## Understanding the Board

The board is displayed in ASCII with each player having their own color:
- **Red (rK, rQ, etc.)**: Player 0 - bottom
- **Blue (bK, bQ, etc.)**: Player 1 - right  
- **Yellow (yK, yQ, etc.)**: Player 2 - top
- **Green (gK, gQ, etc.)**: Player 3 - left

Pieces:
- `K` = King
- `Q` = Queen
- `R` = Rook
- `B` = Bishop
- `N` = Knight
- `P` = Pawn
- `·` = Empty square

## Game Rules Summary

1. **Turn Order**: Red → Blue → Yellow → Green (clockwise)
2. **Elimination**: Players eliminated by checkmate or stalemate
3. **Winning**: Last player standing or highest score when game ends
4. **Scoring**:
   - Capture pieces: +1 to +9 points
   - Checkmate opponent: +20 points
   - Stalemate: varies by situation

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Review [API_DESIGN.md](API_DESIGN.md) for API decisions
4. Explore the code in `four_player_chess/` directory

## Troubleshooting

**Import errors**: Make sure you're in the project root directory and dependencies are installed.

**Slow performance**: The first run will be slower due to JAX JIT compilation. Subsequent runs will be much faster.

**Type warnings**: You may see Pylance type warnings - these are common with JAX and don't affect functionality.

## For RL Training

```python
# Example: Multiple parallel environments
batch_size = 1024
keys = random.split(key, batch_size)

# Vectorize reset
reset_fn = jax.vmap(env.reset)
states, observations = reset_fn(keys)

# Vectorize step  
step_fn = jax.vmap(env.step, in_axes=(0, 0, 0))
actions = random.randint(keys, (batch_size,), 0, env.action_space)
next_states, next_obs, rewards, dones, infos = step_fn(keys, states, actions)

# Now you have 1024 parallel games!
```

Happy training! 🎮♟️