import random

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TetrisGame:

    TETROMINOES_TYPES = ['I', 'J', 'L', 'O', 'S', 'Z', 'T']
    # Defines the tetrominoes with rotation
    TETROMINOES = {
        'I': [
            np.array([[1, 1, 1, 1]]),  # 0 and 180
            np.array([[1], [1], [1], [1]]),  # 90 and 270
        ],
        'J': [
            np.array([[1, 0, 0],
                      [1, 1, 1]]),  # 0
            np.array([[1, 1],
                      [1, 0],
                      [1, 0]]),  # 90
            np.array([[1, 1, 1],
                      [0, 0, 1]]),  # 180
            np.array([[0, 1],
                      [0, 1],
                      [1, 1]]),  # 270
        ],
        'L': [
            np.array([[0, 0, 1],
                      [1, 1, 1]]),  # 0
            np.array([[1, 0],
                      [1, 0],
                      [1, 1]]),  # 90
            np.array([[1, 1, 1],
                      [1, 0, 0]]),  # 180
            np.array([[1, 1],
                      [0, 1],
                      [0, 1]]),  # 270
        ],
        'O': [
            np.array([[1, 1],
                    [1, 1]]),  # 0, 90, 180, 270
        ],
        'S': [
            np.array([[0, 1, 1],
                      [1, 1, 0]]),  # 0 and 180
            np.array([[1, 0],
                      [1, 1],
                      [0, 1]]),  # 90 and 270
        ],
        'Z': [
            np.array([[1, 1, 0],
                      [0, 1, 1]]),  # 0 and 180
            np.array([[0, 1],
                      [1, 1],
                      [1, 0]]),  # 90 and 270
        ],
        'T': [
            np.array([[0, 1, 0],
                      [1, 1, 1]]),  # 0
            np.array([[1, 0],
                      [1, 1],
                      [1, 0]]),  # 90
            np.array([[1, 1, 1],
                      [0, 1, 0]]),  # 180
            np.array([[0, 1],
                      [1, 1],
                      [0, 1]]),  # 270
        ]
    }

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.reset_board()

    def reset_board(self):
        """Prepares the board and spawn the first two tetrominoes"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.game_over = False
        self.last_action = ''
        self.current_piece = None  # next_piece will push forward by spawn_new_piece
        self.next_piece = self._random_tetromino()

    def spawn_new_piece(self):
        """Generates the next piece and update the current piece from queue"""
        self.current_piece = self.next_piece
        self.next_piece = self._random_tetromino()

    def _random_tetromino(self):
        """Randomly return a piece type and all corresponding rotations"""
        piece_type = random.choice(self.TETROMINOES_TYPES)
        rotations = self.TETROMINOES[piece_type]
        return piece_type, rotations

    def _find_drop_height(self, piece, x):
        h, _ = piece.shape
        last_valid_y = None
        for y in range(self.height - h + 1):
            if not self._valid_position(piece, (y, x)):  # Find the lowest valid position
                break
            last_valid_y = y
        return last_valid_y
    
    def _valid_position(self, piece, position):
        """Check if the piece at the given position is within bounds and does not collide."""
        y, x = position
        h, w = piece.shape
        # stay inside the board
        if x < 0 or x + w > self.width or y < 0 or y + h > self.height:
            return False
        # if piece and board both has 1 => cell occupied
        sub_board = self.board[y:y + h, x:x + w]
        return not np.any(piece & sub_board)

    def get_valid_actions(self):
        """Return a list of all valid (rotation_index, x_position) tuples"""
        piece_type, rotations = self.current_piece
        valid_actions = []

        for rot_idx, piece in enumerate(rotations):
            piece_height, piece_width = piece.shape
            # Try placing the piece at every horizontal position where it fits
            for x in range(self.width - piece_width + 1):
                y = self._find_drop_height(piece, x)
                if y is not None and self._valid_position(piece, (y, x)):
                    valid_actions.append((rot_idx, x))

        return valid_actions

    def _lock_piece(self, piece, position):
        """Lock the given piece into the board at the specified position."""
        y, x = position
        h, w = piece.shape
        sub_board = self.board[y:y+h, x:x+w]
        self.board[y:y+h, x:x+w] = sub_board + piece  # Add the piece to the board (assumes no overlap)


    def _clear_lines(self):
        """Clear full lines from board. Returns the number of lines cleared."""
        full_mask = np.all(self.board == 1, axis=1)  # shape: (height,)
        num_cleared = np.count_nonzero(full_mask)

        if num_cleared > 0:
            self.board = self.board[~full_mask]  # Keep only the rows that are not full
            new_rows = np.zeros((num_cleared, self.width), dtype=int)  # stack empty rows on top
            self.board = np.vstack([new_rows, self.board])

        return num_cleared

    def _compute_score(self, lines_cleared):
        """Compute the actual score and reward for this move"""
        return {0: 0, 1: 2, 2: 5, 3: 15, 4: 60}.get(lines_cleared, 0)

    def update_board(self, rot_idx, x):
        """
        Apply the selected move to the board:
        - Lock the piece at (rot_idx, x)
        - Clear lines
        - Update score using scoring function

        Returns:
            lines_cleared (int): number of rows cleared
            reward (int): reward from scoring function
            drop_y (int): vertical position where piece was placed
            info (dict): optional metadata
        """
        piece_type, rotations = self.current_piece
        piece = rotations[rot_idx]

        y = self._find_drop_height(piece, x)
        if y is None or not self._valid_position(piece, (y, x)):  # additional guard to check y validity
            raise ValueError(f"Invalid move attempted: rotation={rot_idx}, x={x}")

        # binary mask for new piece, used for visualization
        placement_mask = np.zeros_like(self.board)
        h, w = piece.shape
        placement_mask[y:y+h, x:x+w] = piece  

        pre_clear_board = self.board.copy()
        self._lock_piece(piece, (y, x))
        self.last_action = f'({rot_idx}, {x})'
        lines_cleared = self._clear_lines()
        self.score += self._compute_score(lines_cleared)

        output_info = {
            "piece": piece_type,
            "rotation": rot_idx,
            "x": x,
            "y": y,
            "game_over": self.game_over,
            "lines_cleared": lines_cleared,
            "score": self.score,
            "pre_clear_board": pre_clear_board,  # for rendering, shows the board pre-clear line
            "placement_mask": placement_mask  # for rendering, shows the last placed block
        }
        return output_info


    def _draw_piece(self, ax, piece, title, offset_y):
        """Helper function to draw a Tetromino piece"""
        ax.text(0, offset_y, title, fontsize=12)
        offset_y = offset_y - 0.5  # Give text and shape a bit of padding
        h, w = piece.shape
        for y in range(h):
            for x in range(w):
                if piece[y, x]:
                    rect = patches.Rectangle(
                        (x, offset_y - y - 1), 1, 1,
                        linewidth=1, edgecolor='black', facecolor='gray')
                    ax.add_patch(rect)

    def render(self, valid_actions, placement_mask=None, pre_clear_board=None, return_fig=False):
        """
        Render the game using matplotlib:
        - Left: the main board
        - Right: current piece, next piece, score
        """
        # First clear the output in notebook, does nothing in non-notebook environment
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass

        _, rotations = self.current_piece  # rotations: a list of the current type's rotation variants
        piece = rotations[0]  # Preview is 0-rotation
        _, next_rotations = self.next_piece
        next_piece = next_rotations[0]

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        # --- Left: Board ---
        ax_board = fig.add_subplot(gs[0, 0])
        ax_board.set_title("Tetris Board")
        board_to_show = pre_clear_board if pre_clear_board is not None else self.board
        ax_board.imshow(board_to_show, cmap='Greys', origin='upper')

        if placement_mask is not None:
            y_coords, x_coords = np.where(placement_mask)
            for y, x in zip(y_coords, x_coords):
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                ax_board.add_patch(rect)

        ax_board.set_xticks(range(self.width))
        ax_board.set_yticks(range(self.height))
        ax_board.set_xticklabels(range(self.width))
        ax_board.set_yticklabels(range(self.height))
        ax_board.set_xlim(-0.5, self.width - 0.5)
        ax_board.set_ylim(self.height - 0.5, -0.5)

        # --- Right Panel ---
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis('off')

        # Create space for two pieces + score + valid actions
        ax_info.set_xlim(0, 5)
        ax_info.set_ylim(-10, 6)

        # Show pieces
        self._draw_piece(ax_info, piece, "Current Piece", offset_y=5)
        self._draw_piece(ax_info, next_piece, "Next Piece", offset_y=-5)
        # Show action
        ax_info.text(0, 1, "Last Action: " + self.last_action, fontsize=12)
        ax_info.text(0, 0, "Valid Actions:", fontsize=12)
        actions_text = ', '.join(f"({r},{x})" for r, x in valid_actions) if valid_actions else ''
        ax_info.text(0, -0.5, actions_text, fontsize=9, wrap=True, verticalalignment='top')
        # Show score
        ax_info.text(0, -9, f"Score: {self.score}", fontsize=12)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            plt.close(fig)


    def player_input(self, valid_actions):
        """Prompt user until a valid action (rot,x) or 'q' is entered."""
        if not valid_actions:
            self.game_over = True
            return None
        while True:
            raw_input = input("Enter rotation_index and x (e.g. '1 3') or 'q': ").strip()
            if raw_input.lower() == "q":
                self.game_over = True
                return None
            try:
                rot, xpos = map(int, raw_input.split())
                if (rot, xpos) in valid_actions:
                    return rot, xpos
                else:
                    print("Not a valid pair; pick from the valid actions list.")
            except ValueError:
                print("Bad input; try again.")


    def check_game_over(self):
        """The game is over if any cell in the top row is 1 (occupied)."""
        if np.any(self.board[0]):
            self.game_over = True


    def play(self):
        while not self.game_over:
            self.spawn_new_piece()
            valid_actions = self.get_valid_actions()
            self.render(valid_actions=valid_actions)
            player_input = self.player_input(valid_actions=valid_actions)  # Edit this for the robot, move MUST BE VALID
            if not player_input:  # exits the game 
                continue
            info = self.update_board(rot_idx=player_input[0], x=player_input[1])
            self.render(
                valid_actions=valid_actions, 
                placement_mask=info["placement_mask"], 
                pre_clear_board=info["pre_clear_board"]
                )
            self.check_game_over()
        self.render(valid_actions=None)
        print("Final score:", self.score)
        self.reset_board()
