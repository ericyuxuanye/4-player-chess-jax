# 4-Player Chess Web UI

A beautiful, interactive web interface for playing 4-player chess!

## Features

- 🎨 Beautiful cross-shaped chess board visualization
- 🖱️ Click-to-select, click-to-move interface
- 👥 Real-time game status and player scores
- ✨ Visual highlighting of selected pieces and valid moves
- 🏆 Winner detection and game over handling
- 🔄 New game button to restart anytime

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Run the server:**
   ```bash
   python app.py
   ```

   Or use a custom port:
   ```bash
   python app.py --port 3000
   ```

   Or set via environment variable:
   ```bash
   PORT=3000 python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:8080` (or your custom port)

4. **Play the game:**
   - Click on a piece to select it (must be your turn and your color)
   - Valid moves will be highlighted in light blue with a green dot
   - Click on a highlighted square to move your piece
   - Click on the selected piece again to deselect it

## Game Rules

This is 4-player chess played on a cross-shaped board:

- **Players:** Red (bottom), Blue (right), Yellow (top), Green (left)
- **Turn Order:** Red → Blue → Yellow → Green (clockwise)
- **Objective:** Be the last player standing!
- **Scoring:**
  - Capture pieces to earn points
  - Eliminate opponents for bonus points
- **Elimination:** When checkmated, you're out of the game

## Board Layout

The board is 14x14 with 160 playable squares forming a cross:

```
    YYY
    YYY
    YYY
GGGCCCCBBB
GGGCCCCBBB
GGGCCCCBBB
    RRR
    RRR
    RRR
```

- R = Red's starting area
- B = Blue's starting area
- Y = Yellow's starting area
- G = Green's starting area
- C = Center area

## Controls

- **Click a piece:** Select it (if it's your turn)
- **Click a valid move:** Move the selected piece there
- **Click selected piece:** Deselect it
- **New Game button:** Start a fresh game
- **Undo button:** (Coming soon!)

## API Endpoints

The Flask server provides these REST API endpoints:

- `GET /api/state` - Get current game state
- `POST /api/reset` - Reset to a new game
- `POST /api/move` - Make a move
- `POST /api/valid_moves` - Get valid moves for a piece

## Technical Details

- **Backend:** Flask + JAX-based game engine
- **Frontend:** Vanilla JavaScript + HTML/CSS
- **Game Engine:** Fully JIT-compiled JAX implementation
- **State Management:** Server-side game state with REST API

## Customization

You can customize the appearance by editing the CSS in `templates/index.html`:

- Change colors in the `.player-name` classes
- Adjust square size (currently 40px)
- Modify the color scheme in the body gradient

Enjoy playing 4-player chess! 🎉
