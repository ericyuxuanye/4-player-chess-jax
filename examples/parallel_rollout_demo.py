"""
Demonstration of parallel rollouts with 4-player chess environment.

Shows how to run multiple games simultaneously using JAX vmap.
"""

import jax
import jax.numpy as jnp
from jax import random
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from four_player_chess import FourPlayerChessEnv


def benchmark_parallel_rollouts(num_games: int = 1000, num_steps: int = 50, seed: int = 0):
    """
    Benchmark parallel execution of multiple games.
    
    Args:
        num_games: Number of parallel games to run
        num_steps: Number of steps per game
        seed: Random seed
    """
    print("=" * 70)
    print(f"PARALLEL ROLLOUT BENCHMARK - {num_games} GAMES")
    print("=" * 70)
    print()
    
    env = FourPlayerChessEnv()
    
    # Generate random keys for each game
    key = random.PRNGKey(seed)
    keys = random.split(key, num_games)
    
    print(f"Setting up {num_games} parallel environments...")
    
    # Vectorize reset function
    reset_fn = jax.vmap(env.reset)
    
    # Reset all environments
    print("Resetting all environments...")
    start_time = time.time()
    states, observations = reset_fn(keys)
    reset_time = time.time() - start_time
    
    print(f"✓ Reset completed in {reset_time:.4f}s")
    print(f"  State shape: {states.board.shape}")
    print(f"  Observation shape: {observations.shape}")
    print()
    
    # Vectorize step function
    step_fn = jax.vmap(env.step, in_axes=(0, 0, 0))
    
    # Run steps
    print(f"Running {num_steps} steps for each game...")
    step_keys = random.split(key, num_games)
    
    total_steps = 0
    total_rewards = jnp.zeros(num_games)
    games_done = jnp.zeros(num_games, dtype=jnp.bool_)
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate random actions for all games
        action_keys = random.split(random.fold_in(key, step), num_games)
        actions = jax.random.randint(action_keys, (num_games,), 0, env.action_space)
        
        # Step all environments in parallel
        states, observations, rewards, dones, infos = step_fn(step_keys, states, actions)
        
        # Update statistics
        total_steps += num_games
        total_rewards += rewards
        games_done = games_done | dones
        
        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            num_done = int(jnp.sum(games_done))
            avg_reward = float(jnp.mean(total_rewards))
            print(f"  Step {step + 1}/{num_steps}: {num_done}/{num_games} games done, "
                  f"avg reward: {avg_reward:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # Print final statistics
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total games:          {num_games}")
    print(f"Steps per game:       {num_steps}")
    print(f"Total steps:          {total_steps}")
    print(f"Time elapsed:         {elapsed_time:.4f}s")
    print(f"Steps per second:     {total_steps / elapsed_time:.0f}")
    print(f"Games per second:     {num_games / elapsed_time:.2f}")
    print()
    print(f"Games completed:      {int(jnp.sum(games_done))}/{num_games}")
    print(f"Average reward:       {float(jnp.mean(total_rewards)):.2f}")
    print(f"Max reward:           {float(jnp.max(total_rewards)):.2f}")
    print(f"Min reward:           {float(jnp.min(total_rewards)):.2f}")
    print()
    
    # Print final game states for a few games
    print("Sample game states (first 3 games):")
    for i in range(min(3, num_games)):
        active_players = int(jnp.sum(states.player_active[i]))
        scores = states.player_scores[i]
        print(f"  Game {i+1}: {active_players} players active, "
              f"scores: {[int(s) for s in scores]}, "
              f"move count: {int(states.move_count[i])}")
    
    print("=" * 70)
    print()
    
    return states, total_rewards


def compare_sequential_vs_parallel(num_games: int = 100, num_steps: int = 20):
    """
    Compare sequential vs parallel execution performance.
    
    Args:
        num_games: Number of games to run
        num_steps: Number of steps per game
    """
    print("=" * 70)
    print("SEQUENTIAL vs PARALLEL COMPARISON")
    print("=" * 70)
    print()
    
    env = FourPlayerChessEnv()
    key = random.PRNGKey(0)
    
    # Sequential execution
    print(f"Running {num_games} games sequentially...")
    start_time = time.time()
    
    for i in range(num_games):
        game_key = random.fold_in(key, i)
        state, obs = env.reset(game_key)
        
        for step in range(num_steps):
            step_key = random.fold_in(game_key, step)
            action = random.randint(step_key, (), 0, env.action_space)
            state, obs, reward, done, info = env.step(step_key, state, action)
    
    sequential_time = time.time() - start_time
    sequential_sps = (num_games * num_steps) / sequential_time
    
    print(f"✓ Sequential time: {sequential_time:.4f}s ({sequential_sps:.0f} steps/sec)")
    print()
    
    # Parallel execution
    print(f"Running {num_games} games in parallel...")
    keys = random.split(key, num_games)
    
    start_time = time.time()
    
    # Reset all
    reset_fn = jax.vmap(env.reset)
    states, observations = reset_fn(keys)
    
    # Step all
    step_fn = jax.vmap(env.step, in_axes=(0, 0, 0))
    
    for step in range(num_steps):
        step_keys = random.split(random.fold_in(key, step), num_games)
        actions = random.randint(step_keys, (num_games,), 0, env.action_space)
        states, observations, rewards, dones, infos = step_fn(step_keys, states, actions)
    
    parallel_time = time.time() - start_time
    parallel_sps = (num_games * num_steps) / parallel_time
    
    print(f"✓ Parallel time: {parallel_time:.4f}s ({parallel_sps:.0f} steps/sec)")
    print()
    
    # Comparison
    speedup = sequential_time / parallel_time
    print("=" * 70)
    print(f"Speedup: {speedup:.2f}x faster with parallel execution")
    print(f"Efficiency: {(speedup / num_games) * 100:.1f}% (ideal: 100%)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel rollout demonstration")
    parser.add_argument("--mode", type=str, default="benchmark", 
                        choices=["benchmark", "compare"],
                        help="Mode: 'benchmark' or 'compare'")
    parser.add_argument("--num-games", type=int, default=1000,
                        help="Number of parallel games")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of steps per game")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "benchmark":
            benchmark_parallel_rollouts(
                num_games=args.num_games,
                num_steps=args.num_steps,
                seed=args.seed
            )
        else:
            compare_sequential_vs_parallel(
                num_games=min(args.num_games, 100),  # Limit for comparison
                num_steps=args.num_steps
            )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()