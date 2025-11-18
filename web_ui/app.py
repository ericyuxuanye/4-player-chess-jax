"""Flask web server for 4-player chess UI."""
import json
import os
import argparse
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import jax
import jax.numpy as jnp
import sys
sys.path.append('..')

from four_player_chess import FourPlayerChessEnv
from four_player_chess.constants import PIECE_NAMES, PLAYER_NAMES
from four_player_chess.board import create_valid_square_mask, square_to_index, index_to_square

app = Flask(__name__)
CORS(app)

# Global game state
env = None
rng_key = jax.random.PRNGKey(0)
game_state = None
obs = None
valid_mask = create_valid_square_mask()


def get_env():
    """Lazy-load the environment to avoid expensive initialization at import time."""
    global env
    if env is None:
        env = FourPlayerChessEnv()
    return env


def encode_action_simple(from_row, from_col, to_row, to_col, promotion):
    """
    Simple Python version of encode_action for web server use.

    Action = source_idx * (160 * 4) + dest_idx * 4 + promotion_type
    """
    # Convert board positions to flat indices (0-159) using numpy
    import numpy as np
    mask_flat = np.array(valid_mask).flatten().astype(np.int32)

    # Source index - count how many valid squares come before this position
    source_flat = from_row * 14 + from_col
    source_idx = int(np.sum(mask_flat[:source_flat]))

    # Dest index - count how many valid squares come before this position
    dest_flat = to_row * 14 + to_col
    dest_idx = int(np.sum(mask_flat[:dest_flat]))

    # Encode action
    action = source_idx * (160 * 4) + dest_idx * 4 + promotion
    return int(action)


def init_game():
    """Initialize a new game."""
    global game_state, obs, rng_key
    rng_key, reset_key = jax.random.split(rng_key)
    state, obs_data = get_env().reset(reset_key)
    game_state = state
    obs = obs_data

    # Warm up JIT compilation for both step and get_pseudo_legal_moves
    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.rules import is_in_check

    print("Warming up JIT compilation...")
    dummy_key = jax.random.PRNGKey(999)

    # Warm up get_pseudo_legal_moves with several different positions
    try:
        for row in [12, 3, 1]:
            for col in [3, 5, 7]:
                _ = get_pseudo_legal_moves(
                    state.board,
                    row, col,
                    0,  # red player
                    valid_mask,
                    state.en_passant_square
                )
    except:
        pass

    # Warm up is_in_check function
    try:
        for player_id in range(4):
            _ = is_in_check(
                state.board,
                state.king_positions[player_id],
                player_id,
                state.player_active,
                valid_mask
            )
    except:
        pass

    # Warm up step function with a valid pawn move (12,3 -> 10,3)
    try:
        valid_action = encode_action_simple(12, 3, 10, 3, 0)
        _, _, _, _, _ = get_env().step(dummy_key, state, jnp.int32(valid_action))
    except:
        pass

    print("JIT warmup complete!")

    return state


def state_to_dict(state):
    """Convert JAX game state to JSON-serializable dict."""
    board_data = []
    for row in range(14):
        row_data = []
        for col in range(14):
            piece_type = int(state.board[row, col, 0])
            owner = int(state.board[row, col, 1])
            valid_square = bool(state.board[row, col, 3])

            row_data.append({
                'piece': piece_type,
                'owner': owner,
                'valid': valid_square
            })
        board_data.append(row_data)

    return {
        'board': board_data,
        'current_player': int(state.current_player),
        'scores': [int(s) for s in state.player_scores],
        'active_players': [bool(a) for a in state.player_active],
        'move_count': int(state.move_count),
        'game_over': bool(jnp.sum(state.player_active) <= 1)
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the game."""
    state = init_game()
    return jsonify(state_to_dict(state))


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    if game_state is None:
        state = init_game()
    else:
        state = game_state
    return jsonify(state_to_dict(state))


@app.route('/api/move', methods=['POST'])
def make_move():
    """Make a move."""
    global game_state, obs, rng_key

    data = request.json
    from_row = data['from_row']
    from_col = data['from_col']
    to_row = data['to_row']
    to_col = data['to_col']
    promotion = data.get('promotion', 0)  # Default to queen

    # Encode the action
    action = encode_action_simple(from_row, from_col, to_row, to_col, promotion)

    # Convert to JAX array to avoid recompilation
    action_jax = jnp.int32(action)

    # Execute move
    rng_key, step_key = jax.random.split(rng_key)
    try:
        next_state, next_obs, reward, done, info = get_env().step(step_key, game_state, action_jax)
        game_state = next_state
        obs = next_obs

        # Convert JAX arrays to Python types
        def jax_to_python(val):
            """Convert JAX arrays to Python types."""
            if hasattr(val, 'tolist'):
                return val.tolist()
            elif hasattr(val, 'item'):
                return val.item()
            else:
                return val

        return jsonify({
            'state': state_to_dict(next_state),
            'reward': jax_to_python(reward),
            'done': jax_to_python(done),
            'info': {k: jax_to_python(v) for k, v in info.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/valid_moves', methods=['POST'])
def get_valid_moves():
    """Get valid moves for a piece."""
    global game_state

    if game_state is None:
        return jsonify({'error': 'Game not initialized'}), 400

    data = request.json
    row = data['row']
    col = data['col']

    from four_player_chess.pieces import get_pseudo_legal_moves
    from four_player_chess.rules import is_in_check

    piece_type = int(game_state.board[row, col, 0])
    owner = int(game_state.board[row, col, 1])

    # Check if it's this player's piece
    if owner != game_state.current_player or piece_type == 0:
        return jsonify({'moves': []})

    # Get pseudo-legal moves for this piece (returns 14x14 boolean array)
    pseudo_moves = get_pseudo_legal_moves(
        game_state.board,
        row,
        col,
        game_state.current_player,
        valid_mask,
        game_state.en_passant_square
    )

    # Filter out moves that would leave king in check
    valid_moves = []
    for move_row in range(14):
        for move_col in range(14):
            if pseudo_moves[move_row, move_col]:
                # Simulate the move - properly update all board channels
                from four_player_chess.constants import CHANNEL_PIECE_TYPE, CHANNEL_OWNER, CHANNEL_HAS_MOVED

                test_board = game_state.board.copy()

                # Move piece to destination
                test_board = test_board.at[move_row, move_col, CHANNEL_PIECE_TYPE].set(piece_type)
                test_board = test_board.at[move_row, move_col, CHANNEL_OWNER].set(game_state.current_player)
                test_board = test_board.at[move_row, move_col, CHANNEL_HAS_MOVED].set(1)

                # Clear source square
                test_board = test_board.at[row, col, CHANNEL_PIECE_TYPE].set(0)  # EMPTY
                test_board = test_board.at[row, col, CHANNEL_OWNER].set(0)
                test_board = test_board.at[row, col, CHANNEL_HAS_MOVED].set(0)

                # Update king position if we're moving the king
                from four_player_chess.constants import KING
                test_king_pos = jnp.where(
                    piece_type == KING,
                    jnp.array([move_row, move_col], dtype=jnp.int32),
                    game_state.king_positions[game_state.current_player]
                )

                # Check if king would be in check after this move
                # This checks ALL opponents - if ANY opponent is still checking, it returns True
                in_check_after = is_in_check(
                    test_board,
                    test_king_pos,
                    game_state.current_player,
                    game_state.player_active,
                    valid_mask
                )

                # Explicitly convert JAX boolean to Python bool
                # Only include move if it doesn't leave king in check from ANY opponent
                is_still_in_check = bool(in_check_after.item() if hasattr(in_check_after, 'item') else in_check_after)

                if not is_still_in_check:
                    valid_moves.append({'row': int(move_row), 'col': int(move_col)})

    return jsonify({'moves': valid_moves})


@app.route('/api/debug_state', methods=['GET'])
def get_debug_state():
    """Get comprehensive debug state including all game details."""
    global game_state

    if game_state is None:
        return jsonify({'error': 'Game not initialized'}), 400

    # Convert all game state to JSON-serializable format
    debug_info = {
        'basic_state': state_to_dict(game_state),
        'detailed_state': {
            'current_player': int(game_state.current_player),
            'move_count': int(game_state.move_count),
            'player_scores': [int(s) for s in game_state.player_scores],
            'player_active': [bool(a) for a in game_state.player_active],
            'en_passant_square': [int(x) for x in game_state.en_passant_square],
            'king_positions': [[int(x) for x in pos] for pos in game_state.king_positions],
            'castling_rights': [[bool(r) for r in rights] for rights in game_state.castling_rights],
            'last_capture_move': int(game_state.last_capture_move),
        },
        'board_details': []
    }

    # Add detailed board information for each square with a piece
    for row in range(14):
        for col in range(14):
            piece_type = int(game_state.board[row, col, 0])
            if piece_type > 0:  # If there's a piece
                debug_info['board_details'].append({
                    'position': [row, col],
                    'piece_type': piece_type,
                    'piece_name': PIECE_NAMES.get(piece_type, 'Unknown'),
                    'owner': int(game_state.board[row, col, 1]),
                    'owner_name': PLAYER_NAMES[int(game_state.board[row, col, 1])],
                    'has_moved': bool(game_state.board[row, col, 2]),
                    'is_promoted': bool(game_state.promoted_pieces[row, col])
                })

    return jsonify(debug_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4-Player Chess Web UI')
    parser.add_argument('--port', type=int, default=None,
                        help='Port to run the server on (default: 8080, or use PORT env var)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    # Priority: command-line arg > environment variable > default (8080)
    port = args.port or int(os.environ.get('PORT', 8080))

    print("Starting 4-Player Chess Web UI...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=True, host=args.host, port=port)
