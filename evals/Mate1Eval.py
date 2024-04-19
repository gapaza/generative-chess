from evals.AbstractEval import AbstractEval, encode_moves
import os
import pickle
import config
from matplotlib import pyplot as plt
import chess
from tqdm import tqdm
import tensorflow as tf
import keras_nlp


from preprocess.strategies.language_modeling import get_padded_piece_encoding_from_str

puzzles_dir = os.path.join(config.root_dir, 'puzzles')







class Mate1Eval(AbstractEval):

    def __init__(self):
        super().__init__()
        self.mate_puzzles = self.filter_puzzles('mateIn1')  # 175,079 total
        self.mate_puzzles = sorted(self.mate_puzzles, key=lambda x: x['rating'])

        # print('Loaded', len(self.mate_puzzles), 'mate puzzles.')
        # print(self.mate_puzzles[0])
        # print(self.mate_puzzles[-1])
        # make a historgram of ratings
        # ratings = [p['rating'] for p in self.mate_puzzles]
        # plt.hist(ratings, bins=20)
        # plt.show()

        self.puzzles_600_800 = [p for p in self.mate_puzzles if 600 <= int(p['rating']) <= 800]      # 33954
        self.puzzles_800_1000 = [p for p in self.mate_puzzles if 800 <= int(p['rating']) <= 1000]    # 46891
        self.puzzles_1000_1200 = [p for p in self.mate_puzzles if 1000 <= int(p['rating']) <= 1200]  # 36446
        self.puzzles_1200_1400 = [p for p in self.mate_puzzles if 1200 <= int(p['rating']) <= 1400]  # 20759
        self.puzzles_1400_1600 = [p for p in self.mate_puzzles if 1400 <= int(p['rating']) <= 1600]  # 16233
        self.puzzles_1600_1800 = [p for p in self.mate_puzzles if 1600 <= int(p['rating']) <= 1800]  # 4357
        self.puzzles_1800_2000 = [p for p in self.mate_puzzles if 1800 <= int(p['rating']) <= 2000]  # 1036
        self.puzzles_2000_2200 = [p for p in self.mate_puzzles if 2000 <= int(p['rating']) <= 2200]  # 148

        # Dataset basename
        self.dataset_name = 'mate1eval'

    def process_puzzles(self, puzzles):
        input_sequences = []
        label_sequences = []
        piece_encodings = []
        masks = []

        for puzzle in tqdm(puzzles, desc='Processing Mate1Eval puzzles'):
            input_sequence, label_sequence, piece_encoding, mask = self.process_puzzle(puzzle)
            input_sequences.append(input_sequence)
            label_sequences.append(label_sequence)
            piece_encodings.append(piece_encoding)
            masks.append(mask)

        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, label_sequences, piece_encodings, masks)
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(encode_moves, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def process_puzzle(self, puzzle):
        board = chess.Board()

        moves = puzzle['moves']
        moves = moves.split(' ')
        for move in moves:
            board.push_uci(move)
        white_turn = (board.turn == chess.WHITE)

        line = puzzle['line']
        line_moves = line.split(' ')

        # Specific for mating puzzles
        if white_turn is True:
            end_token = '[white]'
        else:
            end_token = '[black]'

        input_sequence = ['[start]'] + moves + line_moves
        input_sequence = ' '.join(input_sequence)

        label_sequence = moves + line_moves + [end_token]
        label_sequence = ' '.join(label_sequence)

        mask = [0 for _ in range(len(moves))]
        mask += [1 for _ in range(len(line_moves))]
        mask += [1]  # for the end token
        while len(mask) < config.seq_length:
            mask.append(0)
        if len(mask) > config.seq_length:
            mask = mask[:config.seq_length]

        # Get game piece encoding
        all_moves = ' '.join(moves + line_moves)
        piece_encoding = get_padded_piece_encoding_from_str(all_moves)

        return input_sequence, label_sequence, piece_encoding, mask

    # ---------------------------------
    # Validate model
    # ---------------------------------

    def run_eval(self, model):

        accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

        puzzles = self.puzzles_2000_2200
        dataset = self.process_puzzles(puzzles)

        for input_sequences, label_sequences, piece_encodings, masks in tqdm(dataset, desc='Evaluating mates in one...'):
            predictions = model(input_sequences, training=False)
            accuracy_tracker.update_state(label_sequences, predictions, sample_weight=masks)

        print('Accuracy:', accuracy_tracker.result().numpy())


from model import get_pretrain_model as get_model

if __name__ == '__main__':

    checkpoint_path = config.model_path
    model = get_model(checkpoint_path=checkpoint_path)

    me = Mate1Eval()
    me.run_eval(model)


