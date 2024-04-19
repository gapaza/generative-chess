import chess
import chess.pgn
import tensorflow as tf


def create_pgn_from_uci(uci_moves, filename="game.pgn"):
    game = chess.pgn.Game()
    node = game

    # Setup a chess board
    board = chess.Board()

    # Split the UCI moves string and apply each move to the board
    for move in uci_moves.split():
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
            node = node.add_variation(move_obj)
        else:
            print(f"Invalid move: {move}")
            break

    # Set headers (optional, but useful for metadata)
    game.headers["Event"] = "Example Game"
    game.headers["Site"] = "Internet"
    game.headers["Date"] = "2024.04.13"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player1"
    game.headers["Black"] = "Player2"
    game.headers["Result"] = "*"  # Or use board.result() if the game is over

    print(game)

def padding_func():
    # tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(tensor)
    # max_len = 8
    # seq_len = tf.shape(tensor)[1]
    # padding_size = int(max_len - seq_len)
    #
    # paddings = tf.constant([[0, 0], [0, padding_size]])  # (before1, after1), (before2, after2)
    #
    # padded_tensor = tf.pad(tensor, paddings, "CONSTANT")
    # print(padded_tensor)
    #


    tensor = tf.constant([4, 5, 6])
    max_len = 8
    seq_len = tf.shape(tensor)[0]
    padding_size = int(max_len - seq_len)

    paddings = tf.constant([[0, padding_size]])
    padded_tensor = tf.pad(tensor, paddings, "CONSTANT", constant_values=0)
    print(padded_tensor)





if __name__ == "__main__":
    # padding_func()
    # exit(0)

    # Example UCI move sequence
    uci_moves = """
    
    
d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 g1f3 d5c3 b2c3 f8g7 e2e4 c7c5 f1c4 b8c6 e1g1 c5d4 c3d4 c6d4 a1b1 e8g8 c1b2 e7e5 c4d5 a8b8 f3d4 e5d4 b2d4 b7b6 d4g7 g8g7 d1d4 g7g8 f1d1 c8b7 d4c3 b7d5 d1d5 b8c8 d5d8 c8c3 d8f8 g8f8 b1b2


"""
    uci_moves = uci_moves.strip()
    create_pgn_from_uci(uci_moves)





