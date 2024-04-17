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
from preprocess.strategies import language_modeling
multiprocessing.set_start_method('spawn', force=True)



uci_dir = os.path.join(config.games_dir, 'combined')
# uci_dir2 = os.path.join(config.games_dir, 'millionsbase')

uci_save_dir = os.path.join(config.games_dir, 'combined')
if not os.path.exists(uci_save_dir):
    os.makedirs(uci_save_dir)

uci_piece_dir = os.path.join(config.games_dir, 'combined_piece')
if not os.path.exists(uci_piece_dir):
    os.makedirs(uci_piece_dir)


class GameProcessing:

    def __init__(self):
        self.chunk_uci_dir = uci_dir
        # self.chunk_uci_dir2 = uci_dir2
        self.uci_files, self.file_names = self.load_uci_files()
        # self.engine_api = EngineAPI()
        self.num_procs = 24

    def load_uci_files(self):
        move_files = []
        file_names = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                file_names.append(file)
                move_files.append(full_path)
        # for file in os.listdir(self.chunk_uci_dir2):
        #     if file.endswith('.txt'):
        #         full_path = os.path.join(self.chunk_uci_dir2, file)
        #         file_names.append(file)
        #         move_files.append(full_path)
        return move_files, file_names


    def run(self):
        file_list = self.uci_files
        params = []
        task_num = 0
        for file_name, file_path in zip(self.file_names, self.uci_files):
            source_file = file_path
            dest_file = os.path.join(uci_save_dir, file_name)
            params.append((source_file, dest_file, task_num))
            task_num += 1

        # Have each file processed by a worker
        with multiprocessing.Pool(self.num_procs) as pool:
            pool.map(GameProcessing.process_uci_file, params)


    def run_piece(self):
        file_list = self.uci_files
        params = []
        task_num = 0
        for file_name, file_path in zip(self.file_names, self.uci_files):
            source_file = file_path
            dest_file = os.path.join(uci_piece_dir, file_name)
            params.append((source_file, dest_file, task_num))
            task_num += 1

        # Have each file processed by a worker
        with multiprocessing.Pool(self.num_procs) as pool:
            pool.map(GameProcessing.process_piece_file, params)




    @staticmethod
    def process_piece_file(inputs):
        src_file, dest_file, task_num = inputs
        print('STARTING TASK:', task_num)
        text_dataset = tf.data.TextLineDataset([src_file])
        # text_dataset = text_dataset.take(1000)
        parsed_games = []

        for text_tensor in text_dataset:
            uci_game = text_tensor.numpy().decode('utf-8')
            piece_string = language_modeling.get_game_piece_encoding_from_str(uci_game)
            parsed_games.append(piece_string)


        # write these lines to dest_file
        with open(dest_file, 'w+') as f:
            for game in parsed_games:
                f.write(game + '\n')

        print('FINISHED TASK:', task_num)






    @staticmethod
    def process_uci_file(inputs):
        src_file, dest_file, task_num = inputs
        print('STARTING TASK:', task_num)
        text_dataset = tf.data.TextLineDataset([src_file])
        # text_dataset = text_dataset.take(1000)
        parsed_games = []

        for text_tensor in text_dataset:
            uci_game = text_tensor.numpy().decode('utf-8')
            uci_game_moves = uci_game.split(' ')
            game = chess.Board()
            parsed_moves = []

            # Single game
            for idx, uci_move in enumerate(uci_game_moves):
                # print('Move:', uci_move)
                if uci_move in config.special_tokens or uci_move == '' or uci_move in config.end_of_game_tokens:
                    parsed_games.append(parsed_moves)
                    break  # Illegal token, end game here
                # print('Move:', uci_move)
                move = chess.Move.from_uci(uci_move)
                if move in game.legal_moves:
                    game.push_uci(uci_move)
                    parsed_moves.append(uci_move)
                    # Check for mate
                    if game.is_checkmate():
                        # if white wins
                        if game.result() == '1-0':
                            parsed_moves.append('[white]')
                        # if black wins
                        elif game.result() == '0-1':
                            parsed_moves.append('[black]')
                        # print('CHECKMATE:', ' '.join(parsed_moves))
                        parsed_games.append(parsed_moves)
                        break  # Checkmate, end game here
                    elif game.is_stalemate() or game.is_insufficient_material() or game.is_seventyfive_moves() or game.is_fivefold_repetition():
                        # print('STALEMATE:', ' '.join(parsed_moves))
                        parsed_moves.append('[draw]')
                        parsed_games.append(parsed_moves)
                        break  # Draw, end game here
                else:
                    # print('Illegal move')
                    parsed_games.append(parsed_moves)
                    break  # Illegal move, end game here

        all_games = []
        for game in parsed_games:
            all_games.append(' '.join(game))

        # write these lines to dest_file
        with open(dest_file, 'w+') as f:
            for game in all_games:
                f.write(game + '\n')

        print('FINISHED TASK:', task_num)






if __name__ == '__main__':
    game_processing = GameProcessing()
    # game_processing.run()
    game_processing.run_piece()