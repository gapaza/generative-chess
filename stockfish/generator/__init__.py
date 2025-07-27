import os
import chess
import chess.pgn
import config



def save_pgn(uci_moves, f_name='game.pgn'):

    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for idx, move in enumerate(uci_moves):
        # print('GAME MOVE:', move)
        move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            board.push(move)
            node = node.add_variation(move)
        else:
            break

    game.headers["Event"] = "Example Game"
    game.headers["Site"] = "Internet"
    game.headers["Date"] = "2024.04.13"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player1"
    game.headers["Black"] = "Player2"
    game.headers["Result"] = "*"  # Or use board.result() if the game is over

    save_dir = config.results_dir
    pgn_path = os.path.join(save_dir, f_name)
    with open(pgn_path, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    # Append UCI move sequence to the end of the PGN file
    with open(pgn_path, 'a') as pgn_file:
        pgn_file.write('\n\n')
        pgn_file.write(' '.join(uci_moves))
        pgn_file.write('\n\n')