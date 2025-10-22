"""
Test script for 4-player chess environment.

Plays a game with random moves and renders the board state after each move.
"""

import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from four_player_chess import FourPlayerChessEnv
from four_player_chess.rendering import render_board, print_game_summary
from four_player_chess.utils import decode_action
from four_player_chess.rules import get_all_legal_moves_for_player


def sample_random_legal_action(key, state, env):
    """
    Sample a random legal action for the current player.
    
    This function samples uniformly from all legal moves using a JAX-compatible approach.
    
    Args:
        key: JAX random key
        state: Current game state
        env: Environment instance
    
    Returns:
        A legal action (encoded as integer), or None if no legal moves exist
    """
    current_player = state.current_player
    
    # Get all legal moves for current player as a boolean array (14, 14, 14, 14)
    legal_moves = get_all_legal_moves_for_player(
        state.board,
        current_player,
        state.king_positions[current_player],
        state.player_active,
        env.valid_mask,
        state.en_passant_square
    )
    
    # Flatten to (14*14*14*14,) for easier sampling
    legal_moves_flat = legal_moves.flatten()
    
    # Count number of legal moves
    num_legal = jnp.sum(legal_moves_flat.astype(jnp.int32))
    
    # If no legal moves, return None
    if int(num_legal) == 0:
        return None
    
    # Create a cumulative sum for sampling
    cumsum = jnp.cumsum(legal_moves_flat.astype(jnp.int32))
    
    # Sample a random value between 1 and num_legal
    sample_val = random.randint(key, (), 1, int(num_legal) + 1)
    
    # Find the first position where cumsum >= sample_val
    chosen_idx = int(jnp.argmax(cumsum >= sample_val))
    
    # Decode the flattened index to (sr, sc, dr, dc)
    sr = chosen_idx // (14 * 14 * 14)
    remainder = chosen_idx % (14 * 14 * 14)
    sc = remainder // (14 * 14)
    remainder = remainder % (14 * 14)
    dr = remainder // 14
    dc = remainder % 14
    
    # Now properly encode using the valid square indexing
    # Get the valid square mask and find valid indices
    mask_flat = env.valid_mask.flatten()
    valid_indices = jnp.where(mask_flat, size=160, fill_value=-1)[0]
    
    # Convert board coordinates to flat board indices
    source_board_flat = sr * 14 + sc
    dest_board_flat = dr * 14 + dc
    
    # Find which valid square index corresponds to our source and dest
    source_idx = int(jnp.argmax(valid_indices == source_board_flat))
    dest_idx = int(jnp.argmax(valid_indices == dest_board_flat))
    
    # Encode action: source_idx (0-159) * (160 * 4) + dest_idx (0-159) * 4 + promotion_type (0-3)
    promotion_type = 0  # No promotion for simplicity
    action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
    
    return int(action)


def play_random_game(seed: int = 0, max_moves: int = 100, render_every: int = 1):
    """
    Play a game with random legal moves.
    
    Args:
        seed: Random seed
        max_moves: Maximum number of moves before stopping
        render_every: Render board every N moves
    """
    print("=" * 60)
    print("4-PLAYER CHESS - RANDOM GAME TEST")
    print("=" * 60)
    print()
    
    # Initialize environment
    env = FourPlayerChessEnv()
    key = random.PRNGKey(seed)
    
    # Reset environment
    key, reset_key = random.split(key)
    state, obs = env.reset(reset_key)
    
    print("Initial Board State:")
    print(render_board(state))
    print()
    
    input("Press Enter to start the game...")
    print()
    
    move_num = 0
    done = False

    step_fn = jax.jit(env.step)
    
    while not done and move_num < max_moves:
        move_num += 1
        
        # Get random legal action
        key, action_key = random.split(key)
        action = sample_random_legal_action(action_key, state, env)
        
        if action is None:
            print(f"No legal moves available. Game over.")
            break
        
        # Step environment
        key, step_key = random.split(key)
        next_state, next_obs, reward, done, info = step_fn(step_key, state, action)
        
        # Render every N moves
        if move_num % render_every == 0:
            print(f"\n{'='*60}")
            print(f"Move {move_num}")
            print(f"{'='*60}")
            
            # Show move details
            source_row, source_col, dest_row, dest_col, promo = decode_action(
                action, env.valid_mask
            )
            print(f"Player {state.current_player} moved: ({source_row},{source_col}) -> ({dest_row},{dest_col})")
            if reward > 0:
                print(f"Reward: {float(reward):.1f}")
            
            print()
            print(render_board(next_state))
            
            # Check for eliminations
            if jnp.sum(state.player_active) > jnp.sum(next_state.player_active):
                eliminated = []
                for i in range(4):
                    if state.player_active[i] and not next_state.player_active[i]:
                        eliminated.append(i)
                print(f"\n🚨 Player(s) eliminated: {eliminated}")
            
            if done:
                print("\n" + "="*60)
                print("GAME OVER!")
                print("="*60)
                break
            
            # Wait for user input
            if move_num < max_moves:
                input("Press Enter for next move...")
        
        # Update state
        state = next_state
    
    # Print final summary
    print(print_game_summary(state))
    
    return state


def play_step_by_step(seed: int = 0):
    """
    Play a game step by step with user control.
    
    Args:
        seed: Random seed
    """
    print("=" * 60)
    print("4-PLAYER CHESS - STEP-BY-STEP MODE")
    print("=" * 60)
    print()
    
    # Initialize environment
    env = FourPlayerChessEnv()
    key = random.PRNGKey(seed)
    
    # Reset environment
    key, reset_key = random.split(key)
    state, obs = env.reset(reset_key)
    
    print("Initial Board State:")
    print(render_board(state))
    print()
    
    move_num = 0
    done = False
    
    while not done:
        print(f"\nMove {move_num + 1}")
        print(f"Current player: {int(state.current_player)}")
        print()
        
        # Get action from user or random
        choice = input("Enter 'r' for random move, 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Quitting game.")
            break
        
        # Get a random legal move
        key, action_key = random.split(key)
        action = sample_random_legal_action(action_key, state, env)
        
        if action is None:
            print(f"No legal moves available. Game may be over.")
            break
        
        # Step environment
        key, step_key = random.split(key)
        next_state, next_obs, reward, done, info = env.step(step_key, state, action)
        
        # Show move
        source_row, source_col, dest_row, dest_col, promo = decode_action(
            action, env.valid_mask
        )
        print(f"\nMove: ({source_row},{source_col}) -> ({dest_row},{dest_col})")
        if reward > 0:
            print(f"Reward: {float(reward):.1f}")
        
        # Render board
        print()
        print(render_board(next_state))
        
        # Check for eliminations
        if jnp.sum(state.player_active) > jnp.sum(next_state.player_active):
            print("\n🚨 A player was eliminated!")
        
        if done:
            print("\n" + "="*60)
            print("GAME OVER!")
            print("="*60)
            break
        
        move_num += 1
        state = next_state
    
    # Print final summary
    print(print_game_summary(state))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 4-player chess environment")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "step"],
                        help="Play mode: 'auto' for automatic random game, 'step' for step-by-step")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-moves", type=int, default=50, help="Maximum moves (auto mode)")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N moves (auto mode)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "auto":
            play_random_game(
                seed=args.seed,
                max_moves=args.max_moves,
                render_every=args.render_every
            )
        else:
            play_step_by_step(seed=args.seed)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()