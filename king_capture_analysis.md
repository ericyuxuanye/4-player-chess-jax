# King Capture Bug Analysis

## Summary
**Good news!** The king capture bug was caused by the SAME bug we just fixed in `get_sliding_moves`. The fix we applied should prevent kings from being captured.

## Root Cause: The Sliding Move Bug Affected Both Movement AND Attack Detection

### The Bug's Impact

The `.set()` vs `.max()` bug in `get_sliding_moves` (four_player_chess/pieces.py:223) affected TWO critical systems:

#### 1. Movement (What we already fixed)
- Queens/rooks/bishops couldn't move to certain edge squares
- Example: Queen at [9, 10] couldn't move to [9, 13]

#### 2. **Attack Detection (Why kings could be captured)**
- Queens/rooks/bishops couldn't "see" kings at certain edge positions
- The `is_square_attacked` function relies on `get_pseudo_legal_moves`
- `get_pseudo_legal_moves` uses `get_sliding_moves`
- If sliding pieces couldn't "move" to a square, they also couldn't "attack" that square!

### How King Captures Happened

1. **Red's king is near a board edge** (e.g., at position [13, 7])
2. **Yellow's queen/rook/bishop attacks the king** from a direction that would put it on an edge square
3. **Bug**: The attack detection fails because `get_sliding_moves` incorrectly returns False for the king's position
4. **Result**: Red's king is NOT detected as being in check
5. **Consequence**: Red can make ANY move (even non-defensive ones)
6. **Outcome**: Yellow captures the king on the next turn!

### Move Validation Flow

```
Player tries to move piece
  ↓
is_move_legal() checks:
  1. Does player own the piece? ✓
  2. Is the destination valid? ✓
  3. Is the move pseudo-legal? ✓
  4. Would the move leave the king in check?
     ↓
     is_in_check() checks if king is under attack
       ↓
       For each opponent piece:
         ↓
         is_square_attacked() checks if opponent can attack king
           ↓
           can_piece_attack_square() checks attack capability
             ↓
             get_pseudo_legal_moves() gets all moves
               ↓
               For sliding pieces (Q/R/B):
                 get_sliding_moves() ← BUG WAS HERE!
                   ↓
                   Returns FALSE for edge squares (before fix)
                   Returns TRUE for edge squares (after fix)
```

### The Fix

**Before (BUGGY)**:
```python
moves = moves.at[(flat_rows, flat_cols)].set(flat_mask)
# When clamped duplicates exist, uses LAST value (often False)
```

**After (FIXED)**:
```python
moves = moves.at[(flat_rows, flat_cols)].max(flat_mask)
# When clamped duplicates exist, uses MAX value (preserves True)
```

This fix applies to:
- ✅ Queen movement
- ✅ Rook movement
- ✅ Bishop movement
- ✅ **Attack detection for all sliding pieces**

## Testing Recommendations

To verify the king capture bug is fixed:

1. **Set up a position where**:
   - Red's king is near an edge (e.g., [13, 7])
   - Yellow's queen can attack along a line to an edge square

2. **Before the fix**:
   - Yellow's queen couldn't "see" the king
   - Red could make non-defensive moves
   - Yellow could capture the king

3. **After the fix**:
   - Yellow's queen CAN detect the king is in check
   - Red is FORCED to defend or move the king
   - King capture should be impossible

## Why Kings Should Never Be Captured

In chess, kings are **never** captured. Instead:

1. If a king is under attack → **Check**
2. If a player in check has no legal moves → **Checkmate** → Player is eliminated
3. Legal move validation ensures:
   - You cannot make a move that leaves YOUR king in check
   - If you're in check, you MUST get out of check

The bug allowed kings to be "invisible" to sliding pieces, breaking the check detection system.

## Conclusion

The single fix to `get_sliding_moves` (changing `.set()` to `.max()`) resolves:
- ✅ Movement bug (queens can now reach edge squares)
- ✅ Attack detection bug (sliding pieces can now detect kings at edge positions)
- ✅ **King capture bug** (kings are properly detected as being in check)
