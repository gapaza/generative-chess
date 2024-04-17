import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures
import random
import chess
import warnings

import threading
import multiprocessing
# multiprocessing.set_start_method('fork')
multiprocessing.set_start_method('spawn', force=True)
import time
import sys
import math



from preprocess.strategies import language_modeling



# curr_dataset = config.pt_dataset
curr_dataset = os.path.join(config.datasets_dir, 'combined-dataset-piece')
if not os.path.exists(curr_dataset):
    os.makedirs(curr_dataset)


small_ds = False
# uci_dir = os.path.join(config.games_dir, 'chesscom')
uci_dir = os.path.join(config.games_dir, 'combined')
uci_piece_dir = os.path.join(config.games_dir, 'combined_piece')






class PT_DatasetGenerator:
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

    ##########################
    ### 3. Procure Dataset ###
    ##########################

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

        # zip files
        files = list(zip(move_files, piece_files))

        # shuffle files
        random.shuffle(files)

        train_files = files[:int(len(files) * 0.94)]
        val_files = files[int(len(files) * 0.94):]

        train_move_files, train_piece_files = zip(*train_files)
        val_move_files, val_piece_files = zip(*val_files)


        print("Parsing train dataset...")
        train_dataset = self.parse_memory_dataset_piece(train_move_files, piece_files)
        print("Parsing val dataset...")
        val_dataset = self.parse_memory_dataset_piece(val_move_files, piece_files)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset

    def balance_val_files(self, val_files, kill=False):
        cc_count = 0
        mil_count = 0
        for vfile in val_files:
            print(vfile)
            if 'cc' in vfile:
                cc_count += 1
            elif 'mil' in vfile:
                mil_count += 1
        if cc_count < 2 or mil_count < 2:
            if kill is True:
                exit(0)




    def parse_memory_dataset(self, move_files):
        full_dataset = tf.data.TextLineDataset(move_files)
        full_dataset = full_dataset.batch(config.global_batch_size)
        full_dataset = full_dataset.map(language_modeling.preprocess_decoder_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return full_dataset.prefetch(tf.data.AUTOTUNE)


    def parse_memory_dataset_piece(self, move_files, piece_files):

        full_dataset = tf.data.TextLineDataset(move_files)

        # 1. Parse piece vectors
        piece_dataset = tf.data.TextLineDataset(piece_files)


        for pieces, moves in zip(piece_dataset, full_dataset):
            pieces_split = tf.strings.split(pieces, ' ')
            pieces_split = tf.strings.to_number(pieces_split, out_type=tf.int32)

            # convert moves from tensor () dtype=string to python string
            moves = moves.numpy().decode('utf-8')
            moves_split = moves.split(' ')


            # moves_encoded = config.encode_tf(moves)

            # moves_split = tf.strings.to_number(moves_split, out_type=tf.int32)
            print(pieces)
            print(pieces_split)
            print(moves)
            print(len(moves_split), moves_split)
            exit(0)



        # full_dataset = full_dataset.take(100)

        # 1. Parse piece vectors
        piece_vectors = self.parse_piece_vectors(full_dataset)
        piece_vectors_dataset = tf.data.Dataset.from_tensor_slices(piece_vectors)
        # print('Piece vectors:', piece_vectors_dataset)

        # 2. Combine datasets
        combined_dataset = tf.data.Dataset.zip((full_dataset, piece_vectors_dataset))

        # 3. Preprocess
        combined_dataset = combined_dataset.batch(config.global_batch_size)
        combined_dataset = combined_dataset.map(language_modeling.preprocess_decoder_batch_piece, num_parallel_calls=tf.data.AUTOTUNE)
        return combined_dataset.prefetch(tf.data.AUTOTUNE)

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

    @staticmethod
    def load_uci_files_static(chunk_uci_dir):
        move_files = []
        for file in os.listdir(chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(chunk_uci_dir, file)
                move_files.append(full_path)
        return move_files

    def load_san_files(self):
        move_files = []
        for file in os.listdir(self.chunk_san_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_san_dir, file)
                move_files.append(full_path)
        return move_files

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



        return 0


    def debug_dataset(self):
        train_dataset, val_dataset = self.load_datasets()

        for item in train_dataset:
            input_tensor, label_tensor, piece_tensor = item
            # print('--> Input tensor', input_tensor)
            # print('--> Label tensor', label_tensor)

            input_list = input_tensor.numpy().tolist()
            input_list_game = input_list[0]
            input_game_tokens = [config.id2token[i] for i in input_list_game]
            print('--> Input game token ids:', input_list_game)
            print('-----> Input game tokens:', input_game_tokens)

            piece_tensor = piece_tensor.numpy().tolist()
            piece_tensor_game = piece_tensor[0]
            print('--> Piece tensor:', piece_tensor_game)

            exit(0)

        return 0



if __name__ == '__main__':

    generator = PT_DatasetGenerator(curr_dataset)
    dataset = generator.get_dataset(save=True, small=small_ds)
    # generator.get_num_batches()
    # dataset_train, dataset_val = generator.load_datasets()

    # generator.debug_dataset()










