import chess
import chess.pgn


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



if __name__ == "__main__":

    # Example UCI move sequence
    uci_moves = """
d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1 a7a6 c4d3 c5d4 e3d4 f8e7 b1c3 e8g8 f1e1 b8c6 c1e3 c6b4 d3e2 b4d5 c3d5 f6d5 a2a3 b7b5 a1c1 c8b7 f3e5 d5e3 f2e3 e7g5 e1f2 


"""
    uci_moves = uci_moves.strip()
    create_pgn_from_uci(uci_moves)





