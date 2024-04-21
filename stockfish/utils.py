import config
import chess
import chess.engine


def get_stockfish(threads=12):
    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    engine.configure({'Threads': threads, "Hash": 4096})
    return engine















