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
    
    while not done and move_num < max_moves:
        move_num += 1
        
        # Get random action
        # For simplicity, we'll just try random actions until we find a valid one
        max_attempts = 1000
        action_found = False
        
        for attempt in range(max_attempts):
            key, action_key = random.split(key)
            action = random.randint(action_key, (), 0, env.action_space)
            
            # Decode action to check if it's reasonable
            try:
                source_row, source_col, dest_row, dest_col, promo = decode_action(
                    action, env.valid_mask
                )
                
                # Quick sanity check: source must have current player's piece
                piece_owner = state.board[source_row, source_col, 1]  # CHANNEL_OWNER = 1
                if piece_owner == state.current_player:
                    action_found = True
                    break
            except:
                continue
        
        if not action_found:
            print(f"Could not find valid action after {max_attempts} attempts. Ending game.")
            break
        
        # Step environment
        key, step_key = random.split(key)
        next_state, next_obs, reward, done, info = env.step(step_key, state, action)
        
        # Check if move was valid
        if not info['move_valid']:
            print(f"Move {move_num}: Invalid move attempted, skipping...")
            continue
        
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
        
        # Try to find a valid random move
        max_attempts = 1000
        action_found = False
        
        for attempt in range(max_attempts):
            key, action_key = random.split(key)
            action = random.randint(action_key, (), 0, env.action_space)
            
            try:
                source_row, source_col, dest_row, dest_col, promo = decode_action(
                    action, env.valid_mask
                )
                
                piece_owner = state.board[source_row, source_col, 1]
                if piece_owner == state.current_player:
                    action_found = True
                    break
            except:
                continue
        
        if not action_found:
            print(f"Could not find valid action. Game may be over.")
            break
        
        # Step environment
        key, step_key = random.split(key)
        next_state, next_obs, reward, done, info = env.step(step_key, state, action)
        
        if not info['move_valid']:
            print("Invalid move, trying again...")
            continue
        
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