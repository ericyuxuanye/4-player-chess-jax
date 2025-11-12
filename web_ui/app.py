"""Flask web server for 4-player chess UI."""
import json
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
env = FourPlayerChessEnv()
rng_key = jax.random.PRNGKey(0)
game_state = None
obs = None
valid_mask = create_valid_square_mask()


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
    state, obs_data = env.reset(reset_key)
    game_state = state
    obs = obs_data

    # Warm up JIT compilation with a dummy move to avoid delay on first real move
    dummy_key = jax.random.PRNGKey(999)
    dummy_action = jnp.int32(0)
    try:
        _, _, _, _, _ = env.step(dummy_key, state, dummy_action)
    except:
        pass  # Dummy move might be invalid, that's ok

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
        next_state, next_obs, reward, done, info = env.step(step_key, game_state, action_jax)
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

    piece_type = int(game_state.board[row, col, 0])
    owner = int(game_state.board[row, col, 1])

    # Check if it's this player's piece
    if owner != game_state.current_player or piece_type == 0:
        return jsonify({'moves': []})

    # Get possible moves for this piece (returns 14x14 boolean array)
    moves = get_pseudo_legal_moves(
        game_state.board,
        row,
        col,
        game_state.current_player,
        valid_mask,
        game_state.en_passant_square
    )

    # Convert to row, col format
    valid_moves = []
    for move_row in range(14):
        for move_col in range(14):
            if moves[move_row, move_col]:
                valid_moves.append({'row': int(move_row), 'col': int(move_col)})

    return jsonify({'moves': valid_moves})


if __name__ == '__main__':
    print("Starting 4-Player Chess Web UI...")
    print("Open http://localhost:5000 in your browser")
    init_game()
    app.run(debug=True, host='0.0.0.0', port=5000)
