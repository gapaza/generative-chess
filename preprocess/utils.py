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
    
    
 b1c3 b7b6 e2e4 c8b7 f2f4 e7e6 d2d3 d7d6 g1f3 b8d7 g2g3 c7c5 f1g2 g8f6 e1g1 f8e7 c1e3 e8g8 h2h3 a7a6 g3g4 h7h6 d1e1 b6b5 a1d1 a8c8 g4g5 h6g5 f3g5 b5b4 c3e2 f6e8 e1g3 g7g6 e3c1 e8g7 g3e3 a6a5 b2b3 e7f6 c1d2 f8e8 d2e1 d8c7 d3d4 c5d4 e3d4 f6d4 e2d4 c7b6 e1f2 b6a6 h3h4 g7h5 d4b5 c8c2 b5c7 a6e2 c7e8 h5f4 e8f6 d7f6 g5f3 f6e4 f3d4 e2g4 d4c2 g4g2

"""
    uci_moves = uci_moves.strip()
    create_pgn_from_uci(uci_moves)





