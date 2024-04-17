import config
import tensorflow as tf
import os
from copy import deepcopy
import random
import chess
import chess.svg
from model import get_pretrain_model as get_model




class ChessGPT_API:

    def __init__(self, model_path=config.model_path, user_plays_white=True):
        self.user_plays_white = user_plays_white

        self.model = get_model(model_path)

        # --> Chess Board
        self.board = chess.Board()
        self.move_history = []


    def save_svg(self, filename='board.svg'):
        svg = chess.svg.board(board=self.board, flipped=(not self.user_plays_white))
        full_path = os.path.join(config.results_dir, filename)
        with open(full_path, 'w') as f:
            f.write(svg)

    def new_game(self):
        print('--> STARTING NEW GAME')
        self.board = chess.Board()
        self.move_history = []
        self.save_svg()



    def play_interactive_game(self):
        self.new_game()
        # self.random_position()
        while not self.board.is_game_over():
            print(self.board)
            print(self.move_history)

            if self.board.turn == chess.WHITE and self.user_plays_white or \
                    self.board.turn == chess.BLACK and not self.user_plays_white:
                user_move = input("Your move (in UCI format, e.g. 'e2e4'): ")

                if user_move == 'exit':
                    break
                try:
                    move = chess.Move.from_uci(user_move)
                    if move in self.board.legal_moves:
                        self.move_history.append(user_move)
                        self.board.push(move)
                    else:
                        print("Illegal move. Please enter a valid move:", self.board.legal_moves)
                except ValueError:
                    print("Invalid input. Please enter a valid move in UCI format.")
            else:
                # cpu_move = self.random_move()
                model_move = self.model_move()
                self.move_history.append(model_move.uci())
                print(f"CPU move: {model_move.uci()}")
                self.board.push(model_move)
                self.save_svg()

        print("Game Over.")
        print(self.board.result())

    def random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)




    def model_move(self):
        legal_moves = list(self.board.legal_moves)

        # 2. Add start token
        move_input = deepcopy(self.move_history)
        move_input.insert(0, '[start]')
        move_input_encoded = tf.convert_to_tensor(config.encode(' '.join(move_input)))
        # print('Move input:', move_input)

        # 3. Inference index
        inf_idx = len(move_input) - 1
        print('Inference index:', inf_idx)
        move_input_encoded = tf.expand_dims(move_input_encoded, axis=0)  # Add batch dimension
        pred_logits = self.model(move_input_encoded, training=False)
        move_probs = tf.nn.softmax(pred_logits, axis=-1)
        inf_move_probs = move_probs[:, inf_idx, :]

        #  select top k moves
        k = 5
        top_k_values, top_k_indices = tf.math.top_k(inf_move_probs, k=k)
        top_k_indices = tf.squeeze(top_k_indices, axis=0).numpy().tolist()
        top_k_tokens = [config.id2token[i] for i in top_k_indices]
        print('Top k tokens:', top_k_tokens)
        print('Top k values:', top_k_values.numpy().tolist())
        move_token = top_k_tokens[0]

        # 4. Verify if legal
        legal_moves = list(self.board.legal_moves)
        move = chess.Move.from_uci(move_token)
        if move not in legal_moves:
            print('Illegal move. Randomly selecting a legal move.')
            move_token = random.choice(legal_moves)
            move = chess.Move.from_uci(move_token)

        return move





if __name__ == '__main__':
    interface = ChessGPT_API()
    interface.play_interactive_game()
