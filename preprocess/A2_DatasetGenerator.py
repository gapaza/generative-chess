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

import multiprocessing
multiprocessing.set_start_method('fork', force=True)


def worker(puzzle):
    # This is the function that will be executed in parallel for each puzzle
    return AbstractEval.process_puzzle(puzzle)


# ------------------------------
# Datasets
# ------------------------------

small_ds = False
# curr_dataset = config.pt_dataset
curr_dataset = os.path.join(config.datasets_dir, 'games-a2-human-128b')
if not os.path.exists(curr_dataset):
    os.makedirs(curr_dataset)

# ------------------------------
# UCI Games
# ------------------------------
uci_dir = os.path.join(config.games_dir, 'combined')
lc0_dir = os.path.join(config.games_dir, 'lc0')
use_lc0 = False



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

        # Have each file processed by a worker
        train_dp = []
        with multiprocessing.Pool(self.num_procs) as pool:
            train_dp = pool.map(A2_DatasetGenerator.preproc_datapoints, train_files)
        all_train_dp = []
        for dp in train_dp:
            all_train_dp.extend(dp)
        print('Finished processing train files')

        val_dp = []
        with multiprocessing.Pool(self.num_procs) as pool:
            val_dp = pool.map(A2_DatasetGenerator.preproc_datapoints, val_files)
        all_val_dp = []
        for dp in val_dp:
            all_val_dp.extend(dp)
        print('Finished processing val files')

        # Preprocess datapoints
        train_dataset = self.parse_dataset(all_train_dp)
        val_dataset = self.parse_dataset(all_val_dp)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset



    @staticmethod
    def preproc_datapoints(move_file):
        all_datapoints = []

        # Iterate over lines of text in the file
        with open(move_file, 'r') as f:
            lines = f.readlines()
            # progress_bar = tqdm(lines, desc='Parsing UCI file')
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

                white_datapoint = [
                    ' '.join(white_inputs),
                    ' '.join(white_labels),
                    ' '.join(black_moves),
                    True
                ]
                black_datapoint = [
                    ' '.join(black_inputs),
                    ' '.join(black_labels),
                    ' '.join(white_moves),
                    False
                ]
                all_datapoints.append(white_datapoint)
                all_datapoints.append(black_datapoint)

        return all_datapoints


    def parse_dataset(self, datapoints):
        p1 = [x[0] for x in datapoints]
        p2 = [x[1] for x in datapoints]
        p3 = [x[2] for x in datapoints]
        p4 = [x[3] for x in datapoints]
        dataset = tf.data.Dataset.from_tensor_slices((p1, p2, p3, p4))
        dataset = dataset.batch(config.global_batch_size)
        dataset = dataset.map(color_masking.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
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



    # datasets = [
    #     os.path.join(config.datasets_dir, 'games-puzzles-128b'),
    #     os.path.join(config.datasets_dir, 'lc0-128b')
    # ]
    # generator.integrate_datasets(datasets)


    # dataset = generator.get_dataset(save=True, small=small_ds)
    # generator.get_num_batches()
    # dataset_train, dataset_val = generator.load_datasets()

    # generator.debug_dataset()

