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
from preprocess.strategies import language_modeling
from evals.utils import process_puzzle_batch

import multiprocessing as mp
# multiprocessing.set_start_method('fork')
mp.set_start_method('spawn', force=True)

def worker(puzzle):
    # This is the function that will be executed in parallel for each puzzle
    return AbstractEval.process_puzzle(puzzle)


# ------------------------------
# Datasets
# - games-128b
# - puzzles-128b
# - games-puzzles-128b

small_ds = False
# curr_dataset = config.pt_dataset
curr_dataset = os.path.join(config.datasets_dir, 'games-puzzles-128b')
if not os.path.exists(curr_dataset):
    os.makedirs(curr_dataset)


# ------------------------------
# UCI Games
# ------------------------------
use_games = True
uci_dir = os.path.join(config.games_dir, 'combined')
uci_piece_dir = os.path.join(config.games_dir, 'combined_piece')


# ------------------------------
# Puzzles
# ------------------------------
use_puzzles = True
puzzles_dir = os.path.join(config.datasets_dir, 'evals', 'train', 'puzzles.pkl')


class PZ_DatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir

        # self.chunk_uci_dir = os.path.join(self.dataset_dir, 'chunks_uci')
        self.chunk_uci_dir = uci_dir
        self.chunk_san_dir = os.path.join(self.dataset_dir, 'chunks_san')
        self.chunk_size = 100000

        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')
        self.intermediate_dir = os.path.join(self.dataset_dir, 'intermediate')

    # ------------------------------
    # Procure Dataset
    # ------------------------------

    def get_dataset(self, interleave=False, save=False, small=False):
        chunk_dir = None
        if config.move_language == 'uci':
            chunk_dir = self.chunk_uci_dir
        elif config.move_language == 'san':
            chunk_dir = self.chunk_san_dir

        print('GETTING DATASETS')
        # 1. Get and split move files
        if not os.listdir(chunk_dir) or not os.path.exists(uci_piece_dir):
            print("No UCI files. Skipping dataset creation.")
            return

        move_files, piece_files = self.load_uci_files()

        if small:
            move_files = move_files[:6]
            piece_files = piece_files[:6]

        # zip files
        files = list(zip(move_files, piece_files))

        # shuffle files
        random.shuffle(files)

        train_files = files[:int(len(files) * 0.94)]
        val_files = files[int(len(files) * 0.94):]

        train_move_files, train_piece_files = zip(*train_files)
        val_move_files, val_piece_files = zip(*val_files)

        # Get puzzles
        puzzles = self.get_puzzles()
        train_puzzles = puzzles[:int(len(puzzles) * 0.94)]
        val_puzzles = puzzles[int(len(puzzles) * 0.94):]


        print("Parsing train dataset...")
        train_dataset = self.parse_dataset(train_move_files, train_piece_files, train_puzzles, train=True)
        print("Parsing val dataset...")
        val_dataset = self.parse_dataset(val_move_files, val_piece_files, val_puzzles, train=False)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset


    def parse_dataset(self, move_files, piece_files, puzzles, train=True):

        # 1. UCI Games
        if use_games is True:
            full_dataset = tf.data.TextLineDataset(move_files)

            piece_dataset = tf.data.TextLineDataset(piece_files)
            piece_dataset = piece_dataset.map(language_modeling.pad_piece_dataset, num_parallel_calls=tf.data.AUTOTUNE)

            uci_dataset = tf.data.Dataset.zip((full_dataset, piece_dataset))
            uci_dataset = uci_dataset.batch(config.global_batch_size, drop_remainder=True)
            uci_dataset = uci_dataset.map(language_modeling.preprocess_decoder_batch_piece_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)  # encoded_inputs, encoded_labels, pieces, sample_weights
        else:
            uci_dataset = None

        # 2. Process puzzles (if applicable)
        if use_puzzles is True:
            puzzle_dataset = self.process_puzzles(puzzles)
            print('Finished processing puzzles')

            # 3. Combine datasets, save to intermediate dataset
            if uci_dataset is not None:
                combined_dataset = puzzle_dataset.concatenate(uci_dataset)
            else:
                combined_dataset = puzzle_dataset

            if train is True and uci_dataset is not None:
                combined_dataset = combined_dataset.rebatch(1, drop_remainder=True)
                combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)
                print('Saving intermediate dataset')
                combined_dataset.save(self.intermediate_dir)
                combined_dataset = tf.data.Dataset.load(self.intermediate_dir)
                cardinality = tf.data.experimental.cardinality(combined_dataset).numpy()
                combined_dataset = combined_dataset.shuffle(cardinality)
                combined_dataset = combined_dataset.rebatch(config.global_batch_size, drop_remainder=True)
        else:
            combined_dataset = uci_dataset

        combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)
        return combined_dataset

    # ------------------------------
    # UCI Game Processing
    # ------------------------------

    def load_uci_files(self):
        move_files = []
        piece_files = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                piece_path = os.path.join(uci_piece_dir, file)
                move_files.append(full_path)
                piece_files.append(piece_path)
        return move_files, piece_files

    def parse_piece_vectors(self, text_dataset):
        num_proc = 12
        with mp.Pool(processes=num_proc) as pool:
            # Map process_input function to the inputs
            results = pool.map(language_modeling.get_game_piece_encoding, iter(text_dataset))

        # results = []
        # for item in tqdm(text_dataset, desc='Parsing piece vectors', total=500000):
        #     result = language_modeling.get_game_piece_encoding(item)
        #     results.append(result)

        return results


    # ------------------------------
    # Puzzle Processing
    # ------------------------------

    def get_puzzles(self):
        with open(puzzles_dir, 'rb') as f:
            puzzles = pickle.load(f)
        return puzzles

    def process_puzzles(self, puzzles):
        input_sequences = []
        label_sequences = []
        piece_encodings = []
        masks = []

        # Batch puzzles
        # puzzles = puzzles[:10000]
        num_batches = 12
        batch_size = math.ceil(len(puzzles) / num_batches)
        puzzle_batches = [puzzles[i:i + batch_size] for i in range(0, len(puzzles), batch_size)]

        # Create input parameters
        params = []
        for idx, batch in enumerate(puzzle_batches):
            params.append((batch, config.seq_length, idx))

        # Create a pool of workers, the number of which is equal to the number of available CPU cores
        num_proc = 12
        with mp.Pool(processes=num_proc) as pool:
            results = pool.map(process_puzzle_batch, params)

        for batch in tqdm(results, desc='Unpacking puzzles'):
            input_sequence, label_sequence, piece_encoding, mask = batch
            input_sequences.extend(input_sequence)
            label_sequences.extend(label_sequence)
            piece_encodings.extend(piece_encoding)
            masks.extend(mask)


        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, label_sequences, piece_encodings, masks)
        )
        # Cast dataset elements to int16
        dataset = dataset.batch(config.global_batch_size, drop_remainder=True)
        dataset = dataset.map(encode_moves, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(cast_vals, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset







    ###################
    ### Load / Save ###
    ###################

    def save_datasets(self, train_dataset, val_dataset):
        print("Saving train dataset...")
        train_dataset.save(self.train_dataset_dir)
        print("Saving val dataset...")
        val_dataset.save(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset

    def load_val_dataset(self):
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return val_dataset


    def get_num_batches(self):
        train_dataset, val_dataset = self.load_datasets()

        num_samples_train = tf.data.experimental.cardinality(train_dataset).numpy()
        print("Val Num samples: ", num_samples_train)

        num_samples_val = tf.data.experimental.cardinality(val_dataset).numpy()
        print("Val Num samples: ", num_samples_val)

    def debug_dataset(self):
        train_dataset, val_dataset = self.load_datasets()

        cardinality = tf.data.experimental.cardinality(train_dataset).numpy()
        print('Cardinality:', cardinality)

        for idx, item in enumerate(train_dataset):
            input_tensor, label_tensor, piece_tensor, mask = item
            # print('--> Input tensor', input_tensor)
            # print('--> Label tensor', label_tensor)

            input_list = input_tensor.numpy().tolist()

            input_list_game = input_list[0]
            input_game_tokens = [config.id2token[i] for i in input_list_game]
            print('--> Input game token ids:', input_list_game)
            print('-----> Input game tokens:', input_game_tokens)

            piece_tensor = piece_tensor.numpy().tolist()
            piece_tensor_game = piece_tensor[0]
            print('----------> Piece tensor:', piece_tensor_game)

            mask = mask.numpy().tolist()
            mask_game = mask[0]
            print('------------------> Mask:', mask_game)
            print('\n')

            if idx > 20:
                exit(0)

        return 0



if __name__ == '__main__':

    generator = PZ_DatasetGenerator(curr_dataset)
    dataset = generator.get_dataset(save=True, small=small_ds)
    # generator.get_num_batches()
    # dataset_train, dataset_val = generator.load_datasets()

    # generator.debug_dataset()











