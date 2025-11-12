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
from four_player_chess.utils import encode_action

app = Flask(__name__)
CORS(app)

# Global game state
env = FourPlayerChessEnv()
rng_key = jax.random.PRNGKey(0)
game_state = None
obs = None
valid_mask = create_valid_square_mask()


def init_game():
    """Initialize a new game."""
    global game_state, obs, rng_key
    rng_key, reset_key = jax.random.split(rng_key)
    state, obs_data = env.reset(reset_key)
    game_state = state
    obs = obs_data
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
    action = encode_action(from_row, from_col, to_row, to_col, promotion, valid_mask)

    # Execute move
    rng_key, step_key = jax.random.split(rng_key)
    try:
        next_state, next_obs, reward, done, info = env.step(step_key, game_state, action)
        game_state = next_state
        obs = next_obs

        return jsonify({
            'state': state_to_dict(next_state),
            'reward': [float(r) for r in reward],
            'done': bool(done),
            'info': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in info.items()}
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

    from four_player_chess.pieces import get_piece_moves

    # Convert row, col to index
    from_idx = square_to_index(row, col, valid_mask)

    piece_type = int(game_state.board[row, col, 0])
    owner = int(game_state.board[row, col, 1])

    # Check if it's this player's piece
    if owner != game_state.current_player or piece_type == 0:
        return jsonify({'moves': []})

    # Get possible moves for this piece
    moves = get_piece_moves(game_state, from_idx)

    # Convert to row, col format
    valid_moves = []
    for move_idx in range(160):
        if moves[move_idx]:
            move_row, move_col = index_to_square(move_idx, valid_mask)
            valid_moves.append({'row': int(move_row), 'col': int(move_col)})

    return jsonify({'moves': valid_moves})


if __name__ == '__main__':
    print("Starting 4-Player Chess Web UI...")
    print("Open http://localhost:5000 in your browser")
    init_game()
    app.run(debug=True, host='0.0.0.0', port=5000)
