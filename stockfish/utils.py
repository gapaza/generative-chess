import config
import chess
import chess.engine
import chess.pgn
import io
from itertools import zip_longest
import os



def get_stockfish(threads=12):
    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    engine.configure({'Threads': threads, "Hash": 4096})
    return engine




def pgn_to_uci(pgn):
    pgn = io.StringIO(pgn)

    # read game from pgn string
    game = chess.pgn.read_game(pgn)

    # create board from game
    board = game.board()

    # iterate through moves and play them on the board
    uci_moves = []
    for move in game.mainline_moves():
        uci_move = move.uci()
        uci_moves.append(uci_move)
    uci_moves = " ".join(uci_moves)
    return uci_moves



def combine_alternate(list1, list2):
    # Combine elements from both lists alternately using zip_longest
    # The fillvalue ensures that if one list is longer, None will not be added
    combined = [item for pair in zip_longest(list1, list2, fillvalue='') for item in pair if
                item is not object()]
    combined = [x for x in combined if x != '']
    return combined



def save_game_pgn(game_moves, save_dir):
    uci_moves = [config.id2token[x] for x in game_moves]
    uci_moves_clipped = [x for x in uci_moves if x not in config.non_move_tokens]

    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for idx, move in enumerate(uci_moves_clipped):
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

    pgn_path = os.path.join(save_dir, 'game.pgn')
    with open(pgn_path, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    # Append UCI move sequence to the end of the PGN file
    with open(pgn_path, 'a') as pgn_file:
        pgn_file.write('\n\n')
        pgn_file.write(' '.join(uci_moves))
        pgn_file.write('\n\n')

