import os
import pickle
import config
import chess

from preprocess.puzzle_processing import load_puzzles
from preprocess.strategies.language_modeling import get_padded_piece_encoding_from_str
import tensorflow as tf
from tqdm import tqdm
import random
from copy import deepcopy
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import multiprocessing
multiprocessing.set_start_method('fork', force=True)





# def process_puzzles(puzzles_path, num_procs=12):
#     puzzles = load_puzzles(puzzles_path)
#     # puzzles = puzzles[:1000]
#     with multiprocessing.Pool(num_procs) as pool:
#         train_dp = pool.map(process_puzzle_a3, puzzles)
#     return train_dp

def process_self_play(self_play_path, num_procs=18):
    game_files = load_game_files(self_play_path)
    game_files = game_files[:1000]  # Limit to 1000 games for testing

    results = []
    with multiprocessing.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(process_games_a3, game_files), total=len(game_files), desc="Processing games"):
            results.append(result)

    # Flatten the list of lists
    all_results = []
    for result in results:
        all_results.extend(result)

    return all_results


def load_game_files(games_path):
    move_files = []
    for file in os.listdir(games_path):
        if file.endswith('.txt'):
            full_path = os.path.join(games_path, file)
            move_files.append(full_path)
    return move_files


    with open(puzzles_path, 'rb') as f:
        puzzles = pickle.load(f)
    return puzzles


def process_games_a3(game_file):
    all_datapoints = []

    with open(game_file, 'r') as f:
        lines = f.readlines()
        # progress_bar = tqdm(lines, desc='Parsing UCI file')
        # print('Processing file...')
        for line in lines:

            # Sanitize the moves
            moves = line.strip().split()
            color = moves[0]  # [white] or [black]
            moves = moves[1:]
            # if len(moves) < 6:
            #     continue

            # --- White / Black Datapoints ---
            self_attn_wp = ['[start]']  # white moves from white perspective
            cross_attn_wp = ['[black]']  # black moves from white perspective

            self_attn_bp = ['[start]']  # black moves from black perspective
            cross_attn_bp = []  # white moves from black perspective

            for idx, move in enumerate(moves):
                if idx % 2 == 0:
                    self_attn_wp.append(move)
                    cross_attn_bp.append(move)
                else:
                    cross_attn_wp.append(move)
                    self_attn_bp.append(move)

            self_attn_inputs_wp = self_attn_wp[:-1]  # All moves except the last one
            self_attn_labels_wp = self_attn_wp[1:]  # All moves except the first one
            sample_weights_wp = [1] * len(self_attn_labels_wp)  # All moves have a sample weight of 1
            while len(sample_weights_wp) < config.seq_length:
                sample_weights_wp.append(0)
            if len(self_attn_inputs_wp) > config.seq_length:
                sample_weights_wp = sample_weights_wp[:config.seq_length]
            white_datapoint = [
                ' '.join(self_attn_inputs_wp),
                ' '.join(self_attn_labels_wp),
                ' '.join(cross_attn_wp),
                True,  # Is white
                sample_weights_wp,  # Sample weights for white moves
            ]

            self_attn_inputs_bp = self_attn_bp[:-1]  # All moves except the last one
            self_attn_labels_bp = self_attn_bp[1:]  # All moves except the first one
            sample_weights_bp = [1] * len(self_attn_labels_bp)  # All moves have a sample weight of 1
            while len(sample_weights_bp) < config.seq_length:
                sample_weights_bp.append(0)
            if len(self_attn_inputs_bp) > config.seq_length:
                sample_weights_bp = sample_weights_bp[:config.seq_length]
            black_datapoint = [
                ' '.join(self_attn_inputs_bp),
                ' '.join(self_attn_labels_bp),
                ' '.join(cross_attn_bp),
                False,  # Is black
                sample_weights_bp,  # Sample weights for black moves
            ]

            if color == '[white]':
                all_datapoints.append(white_datapoint)
            elif color == '[black]':
                all_datapoints.append(black_datapoint)
            else:
                raise ValueError(f"Unexpected color token: {color}. Expected '[white]' or '[black]'.")


    return all_datapoints




if __name__ == '__main__':
    game_path = '/Users/gapaza/repos/gabe/generative-chess/results/100k_1'
    all_datapoints = process_self_play(game_path, num_procs=1)
    print(f"Processed {len(all_datapoints)} datapoints from {game_path}")
    # Print the first datapoint
    print("First datapoint:", all_datapoints[0])


