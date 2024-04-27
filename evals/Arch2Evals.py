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

puzzles_dir = os.path.join(config.datasets_dir, 'puzzles')

""" Themes
- advancedPawn:
- advantage:
- anastasiaMate:
- arabianMate:
- attackingF2F7:
- attraction:
- backRankMate:
- bishopEndgame:
- bodenMate:
- capturingDefender:
- castling:
- clearance:
- crushing:
- defensiveMove:
- deflection:
- discoveredAttack:
- doubleBishopMate:
- doubleCheck:
- dovetailMate:
- enPassant:
- endgame:
- equality:
- exposedKing:
- fork:
- hangingPiece:
- hookMate:
- interference:
- intermezzo:
- kingsideAttack:
- knightEndgame:
- long:
- master:
- masterVsMaster:
- mate:
- mateIn1:
- mateIn2:
- mateIn3:
- mateIn4:
- mateIn5:
- middlegame:
- oneMove:
- opening:
- pawnEndgame:
- pin:
- promotion:
- queenEndgame:
- queenRookEndgame:
- queensideAttack:
- quietMove:
- rookEndgame:
- sacrifice:
- short:
- skewer:
- smotheredMate:
- superGM:
- trappedPiece:
- underPromotion:
- veryLong:
- xRayAttack:
- zugzwang:
"""



@tf.function
def encode_moves(model_input, model_label, cross_input, model_plays_white, sample_weights):
    model_input = config.encode_tf(model_input)
    model_label = config.encode_tf(model_label)
    cross_input = config.encode_tf(cross_input)
    return model_input, model_label, cross_input, model_plays_white, sample_weights

class Arch2Evals:
    def __init__(self, themes):
        self.evals_dir = os.path.join(config.datasets_dir, 'evals')
        if not os.path.exists(self.evals_dir):
            os.makedirs(self.evals_dir)
        self.puzzles = self.load_data()  # 1,982,791 total
        self.batch_size = 64

        # Get themes
        self.themes = themes
        self.themes_path = {}
        for theme in self.themes:
            self.themes_path[theme] = os.path.join(self.evals_dir, theme)

        # Get datasets
        # self.theme_datasets = self.preprocess_datasets()
        self.theme_datasets = self.load_theme_datasets()

        # Eval history
        self.eval_history = {}
        for theme in self.themes:
            self.eval_history[theme] = []

        # Eval cache
        self.eval_cache = {}
        self.batch_eval_cache = None

        # Results
        self.eval_results_dir = os.path.join(config.results_dir, 'evals')
        if not os.path.exists(self.eval_results_dir):
            os.makedirs(self.eval_results_dir)

    def load_data(self):
        f_1 = os.path.join(self.evals_dir, 'val', 'puzzles.pkl')
        f_2 = os.path.join(self.evals_dir, 'train', 'puzzles.pkl')
        # f_1 = os.path.join(puzzles_dir, 'lichess_db_puzzle_0_1mil.pkl')
        # f_2 = os.path.join(puzzles_dir, 'lichess_db_puzzle_1_2mil.pkl')

        with open(f_1, 'rb') as f:
            puzzles_1 = pickle.load(f)

        with open(f_2, 'rb') as f:
            puzzles_2 = pickle.load(f)

        all_puzzles = puzzles_1 + puzzles_2

        return all_puzzles

    def filter_by_theme(self, puzzles, theme, limit=None):
        if limit is None:
            limit = 100
        theme_puzzles = {
            '0-800': [],
            '800-1100': [],
            '1100-1400': [],
            '1400-1700': [],
            '1700-2000': [],
            '2000-3000': [],
        }
        num_found = 0
        start_idx = len(puzzles) - 1
        while start_idx >= 0:
            puzzle = puzzles[start_idx]
            if theme in puzzle['themes']:
                puzzle_rating = puzzle['rating']
                if puzzle_rating < 800 and len(theme_puzzles['0-800']) < limit:
                    theme_puzzles['0-800'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
                elif 800 < puzzle_rating < 1100 and len(theme_puzzles['800-1100']) < limit:
                    theme_puzzles['800-1100'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
                elif 1100 < puzzle_rating < 1400 and len(theme_puzzles['1100-1400']) < limit:
                    theme_puzzles['1100-1400'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
                elif 1400 < puzzle_rating < 1700 and len(theme_puzzles['1400-1700']) < limit:
                    theme_puzzles['1400-1700'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
                elif 1700 < puzzle_rating < 2000 and len(theme_puzzles['1700-2000']) < limit:
                    theme_puzzles['1700-2000'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
                elif 2000 < puzzle_rating < 3000 and len(theme_puzzles['2000-3000']) < limit:
                    theme_puzzles['2000-3000'].append(puzzle)
                    puzzles.pop(start_idx)
                    num_found += 1
            start_idx -= 1
            if limit is not None and num_found >= (limit * len(list(theme_puzzles.keys()))):
                break
        return theme_puzzles


    def load_theme_datasets(self):
        theme_dataset_map = {}
        for theme in self.themes:
            theme_path = self.themes_path[theme]
            dataset = tf.data.Dataset.load(theme_path)
            theme_dataset_map[theme] = dataset
        return theme_dataset_map

    def preprocess_datasets(self):
        puzzles = self.puzzles

        theme_puzzle_map = {}
        for idx, theme in enumerate(self.themes):
            theme_puzzles = self.filter_by_theme(puzzles, theme, limit=100)
            # print(len(theme_puzzles['0-800']), len(theme_puzzles['800-1100']), len(theme_puzzles['1100-1400']), len(theme_puzzles['1400-1700']), len(theme_puzzles['1700-2000']), len(theme_puzzles['2000-3000']))
            # print(len(puzzles))
            all_theme_puzzles = []
            for key, val in theme_puzzles.items():
                all_theme_puzzles += val
            theme_puzzle_map[theme] = all_theme_puzzles

        theme_dataset_map = {}
        for theme, pzs in theme_puzzle_map.items():
            dataset = self.get_puzzle_dataset(pzs)
            theme_path = self.themes_path[theme]
            dataset.save(theme_path)
            theme_dataset_map[theme] = dataset


        return theme_dataset_map

    def get_puzzle_dataset(self, puzzles):
        datapoints = []
        for puzzle in puzzles:
            datapoints.append(self.process_puzzle_a2(puzzle))
        p1 = [x[0] for x in datapoints]
        p2 = [x[1] for x in datapoints]
        p3 = [x[2] for x in datapoints]
        p4 = [x[3] for x in datapoints]
        p5 = [x[4] for x in datapoints]
        dataset = tf.data.Dataset.from_tensor_slices((p1, p2, p3, p4, p5))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(encode_moves, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def process_puzzle_a2(self, puzzle):
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

    # ---------------------------------
    # Validate model
    # ---------------------------------

    def run_eval(self, model, save_name=None):
        run_results = {}
        themes = self.themes
        for theme in themes:
            accuracy = self.run_eval_theme(model, theme)
            run_results[theme] = accuracy

        if save_name is not None:
            # save as json file in eval_results_dir
            file_name = save_name + '.json'
            full_save_dir = os.path.join(self.eval_results_dir, file_name)
            with open(full_save_dir, 'w') as f:
                json.dump(run_results, f, indent=4)
        return self.eval_history

    def run_eval_theme(self, model, theme):

        accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

        dataset = self.theme_datasets[theme]

        p_batches = tqdm(dataset, desc='Evaluating '+theme+'...')
        for model_input, model_label, cross_input, model_plays_white, sample_weights in p_batches:
            inputs = [model_input, cross_input, model_plays_white]
            predictions, val_predictions = self.call_model(model, inputs)
            accuracy_tracker.update_state(model_label, predictions, sample_weight=sample_weights)
            if len(self.eval_history[theme]) > 0:
                postfix_val = accuracy_tracker.result().numpy() - self.eval_history[theme][-1]
            else:
                postfix_val = accuracy_tracker.result().numpy()
            p_batches.set_postfix(result=postfix_val)

        accuracy = accuracy_tracker.result().numpy()
        self.eval_history[theme].append(np.float64(accuracy))
        return np.float64(accuracy)



    @tf.function
    def call_model(self, model, inputs):
        predictions, val_predictions = model(inputs, training=False)
        return predictions, val_predictions

    # ---------------------------------
    # Plotting
    # ---------------------------------

    def histogram_comparison(self, file_names, themes=None):
        if themes is None:
            themes = self.themes
        data_dicts = []
        for file_name in file_names:
            run_data = {}
            file_path = os.path.join(self.eval_results_dir, file_name+'.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
            for theme in themes:
                run_data[theme] = data[theme]
            data_dicts.append(run_data)

        # Convert the list of dictionaries to a DataFrame
        fig = plt.subplots(figsize=(10, 6))

        bar_width = 0.25
        all_model_values = []
        for data in data_dicts:
            model_values = []
            for key, val in data.items():
                model_values.append(val)
            all_model_values.append(model_values)

        # Set position of bar on X axis
        br_list = []
        br1 = np.arange(len(all_model_values[0]))
        br_list.append(br1)
        for i in range(1, len(all_model_values)):
            br = [x + bar_width for x in br_list[i-1]]
            br_list.append(br)

        idx = 0
        for model_values, br in zip(all_model_values, br_list):
            plt.bar(br, model_values, width=bar_width, edgecolor='grey', label=file_names[idx])
            idx += 1

        plt.xlabel('Eval', fontweight='bold', fontsize=15)
        plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
        plt.xticks([r + bar_width for r in range(len(all_model_values[0]))], themes)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(self.eval_results_dir, 'histogram.png'))




from model import get_pretrain_model_a2 as get_model
# from model import get_pretrain_model_v2 as get_model

if __name__ == '__main__':
    evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']

    # checkpoint_path = config.model_path

    model_name = 'chess-gpt-a5'
    checkpoint_path = os.path.join(config.weights_dir, model_name)
    model = get_model(checkpoint_path=checkpoint_path)

    ae = Arch2Evals(themes=evals)
    ae.run_eval(model)


    # results = ae.run_eval(model, themes=evals)
    # f_path = os.path.join(config.results_dir, 'evals', 'chess-gpt-v6.json')
    # with open(f_path, 'w') as f:
    #     json.dump(results, f, indent=4)


    # compare_files = ['chess-gpt-v4-1', 'chess-gpt-v4-2', 'chess-gpt-v3']
    # compare_themes = ['advantage', 'mate', 'fork', 'pin', 'equality', 'opening', 'middlegame', 'endgame']
    # ae.histogram_comparison(compare_files, themes=compare_themes)






