import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures
import random
import chess
import warnings
import pickle
import threading
import time
import sys
import math
from evals.AbstractEval import encode_moves, AbstractEval, cast_vals
from preprocess.strategies import language_modeling, color_masking
from evals.utils import process_puzzle_batch
from stockfish.utils import get_stockfish
from stockfish.rewards.reward_2 import calc_reward_slice
from preprocess.puzzle_processing import process_puzzles



import multiprocessing
multiprocessing.set_start_method('fork', force=True)


# ------------------------------
# Datasets
# ------------------------------
small_ds = False
# curr_dataset = config.pt_dataset
curr_dataset = os.path.join(config.datasets_dir, 'dataset-arch2-human-pz')
if not os.path.exists(curr_dataset):
    os.makedirs(curr_dataset)

# ------------------------------
# UCI Games
# ------------------------------
uci_dir = os.path.join(config.games_dir, 'combined')
lc0_dir = os.path.join(config.games_dir, 'lc0')
use_lc0 = False

# ------------------------------
# Puzzles
# ------------------------------
use_puzzles = True





class A2_DatasetGenerator:
    def __init__(self, dataset_dir):
        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir

        self.chunk_uci_dir = uci_dir
        self.lc0_dir = lc0_dir

        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')

        self.num_procs = 12

        # self.engine = get_stockfish()
        self.engine = None
        self.nodes = 50000

        self.puzzles_path = os.path.join(config.datasets_dir, 'evals', 'train', 'unused.pkl')

    # ------------------------------
    # Procure Dataset
    # ------------------------------

    def get_dataset(self, interleave=False, save=False, small=False):
        chunk_dir = self.chunk_uci_dir

        print('GETTING DATASETS')
        # 1. Get and split move files
        if not os.listdir(chunk_dir):
            print("No UCI files. Skipping dataset creation.")
            return

        move_files = self.load_uci_files()
        lc0_files = self.load_lc0_files()

        if small:
            move_files = move_files[:6]
            lc0_files = lc0_files[:6]

        all_files = move_files
        if use_lc0:
            all_files.extend(lc0_files)
        random.shuffle(all_files)

        train_files = all_files[:int(len(all_files) * 0.94)]
        val_files = all_files[int(len(all_files) * 0.94):]

        # ------------------------------
        # Preprocess Train
        # ------------------------------
        train_dp = []
        with multiprocessing.Pool(self.num_procs) as pool:
            train_dp = pool.map(A2_DatasetGenerator.preproc_datapoints, train_files)
        # for train_file in train_files:
        #     train_d = self.preproc_datapoints_eval(train_file)
        #     train_dp.append(train_d)
        all_train_dp = []
        for dp in train_dp:
            all_train_dp.extend(dp)
        if use_puzzles is True:
            print('Processing Puzzles')
            puzzle_dp = process_puzzles(self.puzzles_path)
            all_train_dp.extend(puzzle_dp)
        print('Shuffling train datapoints')
        random.shuffle(all_train_dp)
        print('Finished processing train files')

        # ------------------------------
        # Preprocess Val
        # ------------------------------
        val_dp = []
        with multiprocessing.Pool(self.num_procs) as pool:
            val_dp = pool.map(A2_DatasetGenerator.preproc_datapoints, val_files)
        # for val_file in val_files:
        #     val_d = self.preproc_datapoints_eval(val_file)
        #     val_dp.append(val_d)
        all_val_dp = []
        for dp in val_dp:
            all_val_dp.extend(dp)
        print('Finished processing val files')

        # ------------------------------
        # Parse Datasets
        # ------------------------------
        print('Parsing train dataset', len(all_train_dp))
        train_dataset = self.parse_dataset(all_train_dp)
        print('Parsing val dataset', len(all_val_dp))
        val_dataset = self.parse_dataset(all_val_dp)
        # train_dataset = self.parse_reward_dataset(all_train_dp)
        # val_dataset = self.parse_reward_dataset(all_val_dp)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset

    # ------------------------------
    # Eval Preprocessing
    # ------------------------------

    def preproc_datapoints_eval(self, move_file):
        all_datapoints = []

        # Iterate over lines of text in the file
        with open(move_file, 'r') as f:
            lines = f.readlines()
            progress_bar = tqdm(lines, desc='Parsing UCI file', total=len(lines))
            for line in lines:
                progress_bar.update(1)

                # Game Processing
                # board = chess.Board()
                white_moves = ['[white]']
                black_moves = ['[black]']

                moves = line.strip().split()
                if len(moves) < 12:
                    continue

                white_rewards = []
                black_rewards = []
                white_rewards_sample_weights = []
                black_rewards_sample_weights = []
                eval_sequence, eval_sample_weights = self.get_eval_components(moves, eval_len=4)
                # print(len(eval_sequence))
                for idx, eval_val in enumerate(eval_sequence):
                    # print('Move idx', idx)
                    if idx % 2 == 0:
                        white_rewards.append(eval_val)
                        white_rewards_sample_weights.append(eval_sample_weights[idx])
                    else:
                        black_rewards.append(eval_val)
                        black_rewards_sample_weights.append(eval_sample_weights[idx])
                while len(white_rewards) < config.seq_length:
                    white_rewards.append(0)
                    white_rewards_sample_weights.append(0)
                while len(black_rewards) < config.seq_length:
                    black_rewards.append(0)
                    black_rewards_sample_weights.append(0)

                # If last move is either [white] or [black], replace with [checkmate]
                if moves[-1] == '[white]' or moves[-1] == '[black]':
                    moves[-1] = '[checkmate]'

                for idx, move in enumerate(moves):
                    if idx % 2 == 0:
                        white_moves.append(move)
                    else:
                        black_moves.append(move)

                white_labels = white_moves[1:]
                white_inputs = white_moves[:-1]

                black_labels = black_moves[1:]
                black_inputs = black_moves[:-1]

                white_datapoint = [
                    ' '.join(white_inputs),
                    ' '.join(white_labels),
                    ' '.join(black_moves),
                    True,
                    white_rewards,
                    white_rewards_sample_weights
                ]
                black_datapoint = [
                    ' '.join(black_inputs),
                    ' '.join(black_labels),
                    ' '.join(white_moves),
                    False,
                    black_rewards,
                    black_rewards_sample_weights
                ]
                all_datapoints.append(white_datapoint)
                all_datapoints.append(black_datapoint)




                # break
            progress_bar.close()
        return all_datapoints

    def get_eval_components(self, moves, eval_len=2):
        num_moves = len(moves) - 3
        slice_start_idx = random.randint(5, num_moves)
        slice_end_idx = slice_start_idx + eval_len
        slice = [slice_start_idx, slice_end_idx]
        moves_str = ' '.join(moves)
        rewards, sample_weights = calc_reward_slice(self.engine, moves_str, slice, n=self.nodes)
        return rewards, sample_weights

    # ------------------------------
    # Preprocessing
    # ------------------------------

    @staticmethod
    def preproc_datapoints(move_file):
        all_datapoints = []

        # Iterate over lines of text in the file
        with open(move_file, 'r') as f:
            lines = f.readlines()
            # progress_bar = tqdm(lines, desc='Parsing UCI file')
            # print('Processing file...')
            for line in lines:
                # progress_bar.update(1)

                # Game Processing
                # board = chess.Board()
                white_moves = ['[white]']
                black_moves = ['[black]']

                moves = line.strip().split()
                if len(moves) < 6:
                    continue
                # If last move is either [white] or [black], replace with [checkmate]
                if moves[-1] == '[white]' or moves[-1] == '[black]':
                    moves[-1] = '[checkmate]'

                for idx, move in enumerate(moves):
                    if idx % 2 == 0:
                        white_moves.append(move)
                    else:
                        black_moves.append(move)

                white_labels = white_moves[1:]
                white_inputs = white_moves[:-1]

                black_labels = black_moves[1:]
                black_inputs = black_moves[:-1]

                white_sample_weights = [0 for _ in range(config.seq_length)]
                black_sample_weights = [0 for _ in range(config.seq_length)]
                for idx, wl in enumerate(white_labels):
                    if idx > len(white_sample_weights) - 1:
                        break
                    white_sample_weights[idx] = 1
                for idx, bl in enumerate(black_labels):
                    if idx > len(black_sample_weights) - 1:
                        break
                    black_sample_weights[idx] = 1

                # white_sample_weights = tf.convert_to_tensor(white_sample_weights, dtype=tf.int16)
                # black_sample_weights = tf.convert_to_tensor(black_sample_weights, dtype=tf.int16)

                white_datapoint = [
                    ' '.join(white_inputs),
                    ' '.join(white_labels),
                    ' '.join(black_moves),
                    True,
                    white_sample_weights
                ]
                black_datapoint = [
                    ' '.join(black_inputs),
                    ' '.join(black_labels),
                    ' '.join(white_moves),
                    False,
                    black_sample_weights
                ]
                all_datapoints.append(white_datapoint)
                all_datapoints.append(black_datapoint)

        return all_datapoints

    # ------------------------------
    # Dataset Parsing
    # ------------------------------

    def parse_dataset(self, datapoints):
        # p1, p2, p3, p4, p5 = zip(*datapoints)

        print('Unpacking dataset')
        p1 = [x[0] for x in datapoints]
        p2 = [x[1] for x in datapoints]
        p3 = [x[2] for x in datapoints]
        p4 = [x[3] for x in datapoints]
        p5 = [x[4] for x in datapoints]

        p1 = tf.convert_to_tensor(p1, dtype=tf.string)
        p2 = tf.convert_to_tensor(p2, dtype=tf.string)
        p3 = tf.convert_to_tensor(p3, dtype=tf.string)
        p4 = tf.convert_to_tensor(p4, dtype=tf.bool)
        p5 = tf.convert_to_tensor(p5, dtype=tf.int16)

        # print('Unzipped dataset')
        # print('First p1:', p1[0])
        # print('First p2:', p2[0])
        # print('First p3:', p3[0])
        # print('First p4:', p4[0])
        # print('First p5:', p5[0], len(p5[0]))

        print('Creating dataset')
        dataset = tf.data.Dataset.from_tensor_slices((p1, p2, p3, p4, p5))
        print('Created dataset')
        dataset = dataset.batch(config.global_batch_size)
        dataset = dataset.map(color_masking.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def parse_reward_dataset(self, datapoints):
        p1 = [x[0] for x in datapoints]
        p2 = [x[1] for x in datapoints]
        p3 = [x[2] for x in datapoints]
        p4 = [x[3] for x in datapoints]
        p5 = [x[4] for x in datapoints]
        p6 = [x[5] for x in datapoints]
        dataset = tf.data.Dataset.from_tensor_slices((p1, p2, p3, p4, p5, p6))
        dataset = dataset.batch(config.global_batch_size)
        dataset = dataset.map(color_masking.preprocess_reward_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    # ------------------------------
    # UCI Game Processing
    # ------------------------------

    def load_uci_files(self):
        move_files = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                move_files.append(full_path)
        return move_files

    def load_lc0_files(self):
        move_files = []
        for file in os.listdir(self.lc0_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.lc0_dir, file)
                move_files.append(full_path)
        return move_files

    def parse_piece_vectors(self, text_dataset):
        num_proc = 12
        with multiprocessing.Pool(processes=num_proc) as pool:
            # Map process_input function to the inputs
            results = pool.map(language_modeling.get_game_piece_encoding, iter(text_dataset))

        # results = []
        # for item in tqdm(text_dataset, desc='Parsing piece vectors', total=500000):
        #     result = language_modeling.get_game_piece_encoding(item)
        #     results.append(result)

        return results



    def save_datasets(self, train_dataset, val_dataset):
        print("Saving train dataset...")
        train_dataset.save(self.train_dataset_dir)
        print("Saving val dataset...")
        val_dataset.save(self.val_dataset_dir)



    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset


    def debug_datasets(self):
        train_dataset, val_dataset = self.load_datasets()

        # Print cardinality of datasets
        print('Train dataset cardinality:', train_dataset.cardinality().numpy())
        print('Val dataset cardinality:', val_dataset.cardinality().numpy())







if __name__ == '__main__':

    generator = A2_DatasetGenerator(curr_dataset)
    generator.get_dataset(save=True, small=small_ds)
    generator.debug_datasets()




