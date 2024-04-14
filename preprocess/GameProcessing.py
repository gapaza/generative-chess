import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import chess
import warnings

import threading
import multiprocessing
import time
import sys
import math
from stockfish.EngineAPI import EngineAPI


uci_dir = os.path.join(config.games_dir, 'chesscom')
# uci_dir = os.path.join(config.games_dir, 'millionsbase')


class GameProcessing:

    def __init__(self):
        self.chunk_uci_dir = uci_dir
        self.uci_files = self.load_uci_files()
        self.engine_api = EngineAPI()



    def load_uci_files(self):
        move_files = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                move_files.append(full_path)
        return move_files




    def run(self):
        game_files = self.uci_files[:1]

        text_dataset = tf.data.TextLineDataset(game_files)

        checkmate_games = 0
        drawn_games = 0
        unknown_games = 0

        end_game_evals = []

        for idx, game in enumerate(tqdm(text_dataset)):

            # Isolate uci move tokens
            uci_moves = game.numpy().decode('utf-8')
            uci_moves = uci_moves.split(' ')
            uci_moves = [move for move in uci_moves if move in config.uci_move_tokens]

            game_end_state, end_eval = self.get_game_end_state(uci_moves)
            end_game_evals.append(end_eval)

            if game_end_state == 'checkmate':
                checkmate_games += 1
            elif game_end_state == 'draw':
                drawn_games += 1
            else:
                unknown_games += 1

            if idx % 100 == 0 and idx > 0:
                print('CHECKMATE:', checkmate_games)
                print('DRAWN:', drawn_games)
                print('UNKNOWN:', unknown_games)
                break


        print('End game evals:', end_game_evals)

        print('CHECKMATE:', checkmate_games)
        print('DRAWN:', drawn_games)
        print('UNKNOWN:', unknown_games)

    def get_game_end_state(self, uci_moves):
        board = chess.Board()
        for move in uci_moves:
            move_obj = chess.Move.from_uci(move)
            if move_obj not in board.legal_moves:
                print('--> MOVE IS NOT LEGAL:', move)
                return None, None  # Game is invalid

            # Make move
            board.push_uci(move)

        # Evaluate board state
        forced_moves, curr_eval = self.engine_api.get_forced_mate(board)
        print('Forced checkmate moves:', forced_moves)

        result = 'unknown'
        if board.is_checkmate():
            result = 'checkmate'
        elif board.is_stalemate():
            result = 'draw'
        elif board.is_insufficient_material():
            result = 'draw'
        elif board.is_seventyfive_moves():
            result = 'draw'
        elif board.is_fivefold_repetition():
            result = 'draw'

        return result, curr_eval




if __name__ == '__main__':
    game_processing = GameProcessing()
    game_processing.run()
