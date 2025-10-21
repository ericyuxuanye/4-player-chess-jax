# 4-Player Chess Environment API Design

## Sequential Turn-Based Play

The environment implements **sequential turns** (Player 0 → 1 → 2 → 3 → 0...), where each player observes the state **after** the previous player's move, not simultaneously.

## API Options

### Option 1: Single-Agent Interface (RECOMMENDED)

The environment presents a **unified interface** where the current player is tracked in the state:

```python
# Step function
def step(self, key: PRNGKey, state: EnvState, action: int) 
    -> Tuple[EnvState, Observation, Reward, Done, Info]:
    """
    Execute one move for the current player.
    Returns the new state after advancing to next player's turn.
    """
    # 1. Validate it's a legal move for current_player
    # 2. Apply the move
    # 3. Update scores
    # 4. Check for elimination/game end
    # 5. Advance to next active player
    # 6. Return new state with updated current_player
```

#### Key Characteristics:
- `state.current_player` indicates whose turn it is (0-3)
- Each `step()` call advances the turn to the next active player
- Observations are **player-relative** (always from current player's perspective)
- Rewards are returned for the player who just moved
- Works naturally with self-play training

#### Example Usage:
```python
# Initialize environment
env = FourPlayerChessEnv()
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)

print(f"Current player: {state.current_player}")  # 0 (Red starts)

# Player 0 (Red) moves
key, subkey = jax.random.split(key)
action_0 = select_action(obs)  # Agent chooses move
state, obs, reward, done, info = env.step(subkey, state, action_0)

print(f"Current player: {state.current_player}")  # 1 (Blue's turn)
print(f"Player 0 reward: {reward}")

# Player 1 (Blue) moves  
key, subkey = jax.random.split(key)
action_1 = select_action(obs)
state, obs, reward, done, info = env.step(subkey, state, action_1)

print(f"Current player: {state.current_player}")  # 2 (Yellow's turn)
```

#### Reward Structure:
```python
# Reward returned is for the player who JUST moved
{
    'player_0': score_change_for_red,
    'player_1': score_change_for_blue,
    'player_2': score_change_for_yellow,
    'player_3': score_change_for_green,
    'mover_reward': reward_for_player_who_just_moved  # Primary reward
}
```

#### Observation Space:
Observations are **ego-centric** (from current player's viewpoint):
```python
# Shape: (14, 14, N_CHANNELS)
# Piece channels are arranged as:
# - Channels 0-6: Current player's pieces (one-hot by piece type)
# - Channels 7-13: Next player clockwise (opponent 1)
# - Channels 14-20: Two players ahead (opponent 2)  
# - Channels 21-27: Three players ahead (opponent 3)
# - Channel 28: Valid squares mask
# - Channel 29: En passant target square
# - Channels 30-33: Player active flags (relative ordering)
# - Channel 34: Whose turn it is (always 1 for current player)
```

This ensures the observation is always from the **current player's perspective**, making self-play training straightforward.

### Option 2: Multi-Agent Interface (Alternative)

Each player has a separate agent ID:

```python
def step(self, key: PRNGKey, state: EnvState, action: int, agent_id: int) 
    -> Tuple[EnvState, Dict[int, Observation], Dict[int, Reward], Done, Info]:
    """
    Execute move for specified agent.
    Only valid if agent_id == state.current_player.
    """
```

**Pros**: More explicit about which player is moving  
**Cons**: More complex API, harder to parallelize, not standard for Gymnax

### Option 3: Batched Multi-Agent (Not Recommended)

All 4 players provide actions simultaneously, but only current player's action is executed.

**Cons**: Wasteful, confusing semantics, doesn't match chess.com variant

## Recommended Design: Option 1

**Rationale:**
1. **Clean API**: Single observation/action per step
2. **Self-play friendly**: Natural for training a single policy to play all positions
3. **Efficient**: No wasted computation
4. **Standard**: Similar to AlphaZero approach for multi-player games
5. **Parallel rollouts**: Easy to vmap across multiple games

## Handling Player Elimination

When a player is eliminated (checkmated/stalemated):
1. Mark `state.player_active[player_id] = False`
2. Gray out their pieces (set owner to -1 in board state)
3. Skip their turns: `current_player` advances to next active player
4. Game ends when only 1 or 0 players remain active

```python
def advance_to_next_player(state: EnvState) -> EnvState:
    """Find next active player in clockwise order"""
    for i in range(1, 5):  # Check next 4 players
        next_player = (state.current_player + i) % 4
        if state.player_active[next_player]:
            return state.replace(current_player=next_player)
    # No active players remain - game over
    return state
```

## Self-Play Training Workflow

```python
# Training loop with sequential play
for episode in range(num_episodes):
    state, obs = env.reset(key)
    
    while not done:
        # Single policy network takes current player's perspective
        action, value = policy_network(obs)
        
        # Step environment
        state, obs, reward, done, info = env.step(key, state, action)
        
        # Store experience for the player who just moved
        player_id = (state.current_player - 1) % 4  # Previous player
        store_transition(player_id, obs, action, reward, next_obs)
```

## Parallel Rollout Example

```python
# Vectorize across multiple games
batch_size = 1024

# Each game proceeds sequentially, but we run 1024 games in parallel
keys = jax.random.split(key, batch_size)
states, obs = jax.vmap(env.reset)(keys)

# All games step in parallel (each at their own current_player)
actions = jax.vmap(policy_network)(obs)
keys = jax.random.split(key, batch_size)
states, obs, rewards, dones, infos = jax.vmap(env.step)(keys, states, actions)

# Result: 1024 games, each advanced by one player move
# Some games might be at Player 0's turn, others at Player 2's turn, etc.
```

## State Consistency

After each `step()`:
- `state.current_player` is updated to next active player
- `state.move_count` is incremented
- `state.player_scores` may be updated based on action taken
- Board state reflects the move that was just executed
- Observation is generated from **new** current player's perspective

## Action Masking

Legal actions must be checked for the **current player**:

```python
# Get legal action mask for current player
legal_mask = env.get_legal_actions(state)  # Shape: (102400,)

# Policy network can mask invalid actions
logits = policy_network(obs)
masked_logits = jnp.where(legal_mask, logits, -jnp.inf)
action = jax.random.categorical(key, masked_logits)
```

## Info Dictionary

The `info` dict contains additional metadata:

```python
{
    'current_player': 0,          # ID of player whose turn it is NOW
    'last_player': 3,              # ID of player who just moved
    'move_count': 42,              # Total moves in game
    'active_players': [0, 1, 3],  # Which players are still in game
    'player_scores': [15, 8, 12, 0],  # Current scores
    'is_check': True,              # Is current player in check?
    'legal_actions_count': 23,     # Number of legal moves available
    'last_move': 'e4-e5',         # Human-readable last move (optional)
}
```

## Summary

The API follows a **single-agent, turn-based, ego-centric** design:
- ✅ One step = one player move
- ✅ Observation always from current player's view
- ✅ State tracks whose turn it is
- ✅ Players take turns sequentially
- ✅ Efficient for self-play and parallel rollouts
- ✅ Handles elimination gracefully

This design matches the chess.com variant's sequential nature while being JAX-friendly and suitable for RL training.