# Queen Move Validation Issue Analysis

## Board State
- Red Queen at [9, 10]
- Red King at [13, 7]
- Current player: Red (player 0)

## The Issue
The queen should be able to move to [9, 13] (which is adjacent to the Blue Bishop at [8, 13]), but this move is not showing up in the valid moves.

## Theoretical Analysis

### Move: Queen from [9, 10] to [9, 13]
- This is a rook-like move (horizontal, 3 squares to the right)
- Path: [9, 10] → [9, 11] → [9, 12] → [9, 13]
- Squares [9, 11] and [9, 12] are empty
- Square [9, 13] is empty
- This should be a valid pseudo-legal move

### Check Validation
After the queen moves to [9, 13], we need to check if the Red King at [13, 7] would be in check.

Potential attackers:
1. **Blue pieces**:
   - Blue Rook at [3, 13]
   - Blue Rook at [10, 13]
   - Blue Bishop at [8, 13]
   - Blue Bishop at [5, 13]
   - Blue Queen at [6, 13]
   - Blue King at [7, 13]
   - Blue Knight at [8, 11]
   - Blue Pawns at various positions

2. **Yellow pieces**: At the top of the board
3. **Green pieces**: On the left side

## Hypothesis

The problem is likely NOT with the pseudo-legal move generation (the queen CAN move there geometrically), but with the check validation logic incorrectly thinking the Red King would be in check after the queen moves.

Let me analyze if any opponent piece could attack the Red King at [13, 7] after the queen moves to [9, 13]:

### Diagonal Attacks to [13, 7]:
- From [13, 7], diagonal attackers (Bishop/Queen) would need to be on diagonals:
  - Up-Left diagonal: [12,6], [11,5], [10,4], ...
  - Up-Right diagonal: [12,8], [11,9], [10,10], ...
  - Down-Left diagonal: (none, at edge)
  - Down-Right diagonal: (none, at edge)

**WAIT!** The queen is currently at [9, 10], which is NOT on any diagonal to the Red King at [13, 7].

Let me check if it's on a line (rook-like):
- Red King at [13, 7]
- Red Queen at [9, 10]
- These are NOT on the same row or column

So the queen moving away should not expose the king to any attack.

### BUT... Let's check position [9, 10] more carefully

Looking at potential pins:
- Is there any enemy piece that could attack the king through the queen's current position?

From [13, 7] (king):
- Row 13: Goes left/right
- Column 7: Goes up/down to [12, 7], [11, 7], [10, 7], ...

**AH! Column 7:**
- Red King at [13, 7]
- Column 7 going up: [12, 7] empty, [11, 7] empty, [10, 7] has Red Pawn!

So there's no line attack on column 7.

Hmm, let me think differently. Maybe the bug is in a different direction.

## Potential Bug Locations

1. **`is_square_attacked` function** - May be incorrectly detecting attacks
2. **Coordinate system confusion** - Maybe row/col are swapped somewhere
3. **Player index issue** - Maybe checking wrong player's pieces

Given that the user specifically mentions "[9, 13] adjacent to the blue rook and bishop", let me verify:
- Blue Bishop at [8, 13] - YES, [9, 13] is adjacent (one square below)
- Blue Rook... where?
  - [3, 13] - far away
  - [10, 13] - one square below [9, 13]

So [9, 13] is between the Blue Bishop at [8, 13] and Blue Rook at [10, 13]!

This is an empty square that the queen should be able to move to.

## Next Steps
Need to actually run the validation code to see what's happening. The issue is likely in the `is_in_check` or `is_square_attacked` functions incorrectly detecting a check that doesn't exist.
