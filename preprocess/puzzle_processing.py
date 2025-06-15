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





# def process_puzzles(puzzles_path, num_procs=12):
#     puzzles = load_puzzles(puzzles_path)
#     # puzzles = puzzles[:1000]
#     with multiprocessing.Pool(num_procs) as pool:
#         train_dp = pool.map(process_puzzle_a3, puzzles)
#     return train_dp

def process_puzzles(puzzles_path, num_procs=18):
    puzzles = load_puzzles(puzzles_path)
    # puzzles = puzzles[:1000]
    results = []
    with multiprocessing.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(process_puzzle_a3, puzzles), total=len(puzzles), desc="Processing puzzles"):
            results.append(result)
    return results


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
    # sample_weights = tf.convert_to_tensor(sample_weights, dtype=tf.int16)
    datapoint = [
        ' '.join(model_inputs),
        ' '.join(model_labels),
        ' '.join(cross_inputs),
        model_plays_white,
        sample_weights
    ]
    return datapoint




def process_puzzle_a3(puzzle):
    board = chess.Board()

    # Push all previous moves to the board
    moves = puzzle['moves']
    moves = moves.split(' ')
    for move in moves:
        board.push_uci(move)

    # Identify the color solving the puzzle
    white_turn = (board.turn == chess.WHITE)  # Which color is solving the puzzle?

    # Push puzzle line moves to the board
    line = puzzle['line']
    line_moves = line.split(' ')
    for idx, move in enumerate(line_moves):
            board.push_uci(move)


    # Initialize the model inputs
    if white_turn is True:
        self_attn = ['[start]']  # white moves from white perspective
        cross_attn = ['[black]'] # black moves from white perspective
    else:
        self_attn = ['[start]']  # black moves from black perspective
        cross_attn = []          # white moves from black perspective

    # Iterate over non-line moves
    self_attn_weights = [0]  # initial 0 to account for the [start] token
    for idx, move in enumerate(moves):
        if idx % 2 == 0:
            if white_turn is True:
                self_attn.append(move)
                self_attn_weights.append(0)
            else:
                cross_attn.append(move)
        else:
            if white_turn is True:
                cross_attn.append(move)
            else:
                self_attn.append(move)
                self_attn_weights.append(0)

    # iterate over the line moves
    # - the first line move will always be the model's move
    for idx, move in enumerate(line_moves):
        if idx % 2 == 0:
            self_attn.append(move)
            self_attn_weights.append(1)
        else:
            cross_attn.append(move)

    self_attn_inputs = self_attn[:-1]
    self_attn_labels = self_attn[1:]

    self_attn_weights = self_attn_weights[1:]  # remove the first 0
    while len(self_attn_weights) < config.seq_length:
        self_attn_weights.append(0)
    if len(self_attn_weights) > config.seq_length:
        self_attn_weights = self_attn_weights[:config.seq_length]

    input_sequence = ' '.join(self_attn_inputs)
    label_sequence = ' '.join(self_attn_labels)
    cross_attn_sequence = ' '.join(cross_attn)

    datapoint = [
        input_sequence,
        label_sequence,
        cross_attn_sequence,
        white_turn,
        self_attn_weights
    ]

    return datapoint






