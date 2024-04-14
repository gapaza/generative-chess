import chess
import chess.pgn as chess_pgn
import numpy as np
import re
from tqdm import tqdm
import concurrent.futures
import json
import random
import os
import chess.engine
from chess.engine import PovScore
import pickle
import time
import config
import multiprocessing



class EngineAPI:

    def __init__(self):
        self.threads = 5
        self.lines = 1
        self.nodes = 100000
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': self.threads, "Hash": 1024})


    def evaluate_uci_sequence(self, uci_sequence):
        board = chess.Board()
        uci_moves = uci_sequence.split(" ")
        for uci_move in uci_moves:
            board.push_uci(uci_move)

        return self.evaluate_board(board)


    def get_forced_mate(self, board):
        analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=1)
        line = analysis[0]
        print(line)
        score = line["score"].white()

        if score.is_mate():
            moves = line['pv']
            moves = [move.uci() for move in moves]
            return moves, score
        else:
            return None, None


    def evaluate_board(self, board):
        analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)

        for idx, line in enumerate(analysis):
            print(line)


            # line_top_move = line["pv"][0].uci()
            line_top_move_score = line["score"].white().score()
            if line_top_move_score:
                line_top_move_score_norm = line_top_move_score / 100.0
            else:
                # it must be a forced mate, check if it is
                print(board.fen())
                line_top_move_score_norm = 0.0


            score = line["score"]
            if score.is_mate():
                print(score.white().mate())
                print(f"mate in {score.white().mate()}")
                exit(0)
            else:
                print(f"cp: {score.relative.score()}")


            return line_top_move_score_norm



if __name__ == '__main__':
    game = 'g1f3 d7d5 g2g3 c7c6 f1g2 g8f6 e1g1 g7g6 h2h3 f8g7 d2d3 e8g8 c2c3 c8d7 b1d2 c6c5 e2e4 b8c6 e4e5 f6e8 d3d4 e8c7 f1e1 c5d4 c3d4 c7e6 d2b3 a7a5 c1e3 a5a4 b3d2'

    api = EngineAPI()
    api.evaluate_uci_sequence(game)







