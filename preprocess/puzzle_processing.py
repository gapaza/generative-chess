import os
import pickle
import config
import chess
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





def process_puzzles(puzzles_path, num_procs=12):
    puzzles = load_puzzles(puzzles_path)
    with multiprocessing.Pool(num_procs) as pool:
        train_dp = pool.map(process_puzzle, puzzles)
    return train_dp


def load_puzzles(puzzles_path):
    with open(puzzles_path, 'rb') as f:
        puzzles = pickle.load(f)
    return puzzles


def process_puzzle(puzzle):
    board = chess.Board()

    white_moves = ['[white]']
    black_moves = ['[black]']

    moves = puzzle['moves']
    moves = moves.split(' ')
    for idx, move in enumerate(moves):
        board.push_uci(move)
        if idx % 2 == 0:
            white_moves.append(move)
        else:
            black_moves.append(move)

    model_plays_white = (board.turn == chess.WHITE)
    if model_plays_white is True:
        start_sample_weight_idx = len(white_moves) - 1
    else:
        start_sample_weight_idx = len(black_moves) - 1

    white_line_moves = []
    black_line_moves = []
    line = puzzle['line']
    line_moves = line.split(' ')
    for idx, move in enumerate(line_moves):
        white_turn = (board.turn == chess.WHITE)
        if white_turn is True:
            white_line_moves.append(move)
        else:
            black_line_moves.append(move)
        board.push_uci(move)

    all_white_moves = white_moves + white_line_moves
    all_black_moves = black_moves + black_line_moves

    white_labels = all_white_moves[1:]
    black_labels = all_black_moves[1:]

    white_inputs = all_white_moves[:-1]
    black_inputs = all_black_moves[:-1]

    if model_plays_white is True:
        model_inputs = white_inputs
        model_labels = white_labels
        cross_inputs = all_black_moves
    else:
        model_inputs = black_inputs
        model_labels = black_labels
        cross_inputs = all_white_moves

    sample_weights = [0 for x in range(config.seq_length)]
    end_sample_weight_idx = len(model_labels)
    for idx in range(start_sample_weight_idx, end_sample_weight_idx):
        sample_weights[idx] = 1

    datapoint = [
        ' '.join(model_inputs),
        ' '.join(model_labels),
        ' '.join(cross_inputs),
        model_plays_white,
        sample_weights
    ]
    return datapoint

