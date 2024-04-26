import config
import chess
import chess.engine
import chess.pgn
import io
from itertools import zip_longest


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




