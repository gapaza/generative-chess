import os
import pickle
import config
import chess
from preprocess.strategies.language_modeling import get_padded_piece_encoding_from_str
import tensorflow as tf
from tqdm import tqdm
import random
from copy import deepcopy

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
def encode_moves(input_seq, label_seq, piece_encoding, mask):
    input_seq = config.encode_tf(input_seq)
    label_seq = config.encode_tf(label_seq)
    # input_seq = tf.cast(input_seq, tf.int16)
    # label_seq = tf.cast(label_seq, tf.int16)
    return input_seq, label_seq, piece_encoding, mask

@tf.function
def cast_vals(input_seq, label_seq, piece_encoding, mask):
    input_seq = tf.cast(input_seq, tf.int16)
    label_seq = tf.cast(label_seq, tf.int16)
    piece_encoding = tf.cast(piece_encoding, tf.int16)
    mask = tf.cast(mask, tf.float16)
    return input_seq, label_seq, piece_encoding, mask



class AbstractEval:
    def __init__(self):
        self.evals_dir = os.path.join(config.datasets_dir, 'evals')
        if not os.path.exists(self.evals_dir):
            os.makedirs(self.evals_dir)
        self.puzzles = self.load_data()  # 1,982,791 total
        self.batch_size = 64

        # Get themes
        self.themes = []
        for puzzle in self.puzzles:
            for theme in puzzle['themes']:
                if theme not in self.themes:
                    self.themes.append(theme)
        self.themes = sorted(self.themes)
        print('Puzzle Themes:', self.themes)
        # self.sep_puzzles()

        # Eval history
        self.eval_history = {}
        for theme in self.themes:
            self.eval_history[theme] = []

        # print('Loaded', len(self.puzzles), 'puzzles.')
        # print(self.puzzles[0])
        # print(self.themes)

    def sep_puzzles(self):
        # shuffle puzzles
        puzzles = deepcopy(self.puzzles)
        random.shuffle(puzzles)

        print('NUM PUZZLES:', len(puzzles))

        # extract the first n themed puzzles for each theme as validation
        val_puzzles = []
        for theme in self.themes:
            filtered = self.filter_puzzles(theme, puzzles)
            val_puzzles += filtered[:100]
            puzzles = [p for p in puzzles if p not in filtered[:100]]


        # Save puzzles to dir
        train_puzzles_dir = os.path.join(self.evals_dir, 'train')
        if not os.path.exists(train_puzzles_dir):
            os.makedirs(train_puzzles_dir)
        val_puzzles_dir = os.path.join(self.evals_dir, 'val')
        if not os.path.exists(val_puzzles_dir):
            os.makedirs(val_puzzles_dir)

        # save entire puzzles object as pickle for train dataset
        f = os.path.join(train_puzzles_dir, 'puzzles.pkl')
        with open(f, 'wb') as f:
            pickle.dump(puzzles, f)

        # save entire puzzles object as pickle for val dataset
        f = os.path.join(val_puzzles_dir, 'puzzles.pkl')
        with open(f, 'wb') as f:
            pickle.dump(val_puzzles, f)

        exit(0)


    def load_data(self):
        f_1 = os.path.join(self.evals_dir, 'val', 'puzzles.pkl')
        # f_1 = os.path.join(puzzles_dir, 'lichess_db_puzzle_0_1mil.pkl')
        # f_2 = os.path.join(puzzles_dir, 'lichess_db_puzzle_1_2mil.pkl')

        with open(f_1, 'rb') as f:
            puzzles_1 = pickle.load(f)

        # with open(f_2, 'rb') as f:
        #     puzzles_2 = pickle.load(f)

        all_puzzles = puzzles_1  # + puzzles_2

        return all_puzzles

    def filter_puzzles(self, theme, puzzles=None):
        filtered = []
        if puzzles is None:
            puzzles = self.puzzles
        for puzzle in puzzles:
            if theme in puzzle['themes']:
                filtered.append(puzzle)
        return filtered

    def process_puzzles(self, puzzles):
        input_sequences = []
        label_sequences = []
        piece_encodings = []
        masks = []

        for puzzle in puzzles:
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

    @staticmethod
    def process_puzzle(puzzle):
        board = chess.Board()

        moves = puzzle['moves']
        moves = moves.split(' ')
        for move in moves:
            board.push_uci(move)

        line = puzzle['line']
        line_moves = line.split(' ')
        white_turn = (board.turn == chess.WHITE)
        for idx, move in enumerate(line_moves):
            white_turn = (board.turn == chess.WHITE)
            board.push_uci(move)

        # Create Input Sequence
        input_sequence = ['[start]'] + moves + line_moves
        input_sequence = ' '.join(input_sequence)

        # Create Label Sequence
        label_sequence = moves + line_moves
        if board.is_checkmate():
            if white_turn is True:  # White just mated black
                label_sequence += ['[white]']
            else:                   # Black just mated white
                label_sequence += ['[black]']
        label_sequence = ' '.join(label_sequence)

        # Create Mask
        mask = [0 for _ in range(len(moves))]
        mask += [1 for _ in range(len(line_moves))]
        if board.is_checkmate():
            mask += [1]  # Predict the color that just won
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
        for theme in self.themes:
            self.run_eval_theme(model, theme)
        return deepcopy(self.eval_history)

    def run_eval_theme(self, model, theme):

        accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

        puzzles = self.filter_puzzles(theme)
        dataset = self.process_puzzles(puzzles)

        p_batches = tqdm(dataset, desc='Evaluating '+theme+'...')
        for input_sequences, label_sequences, piece_encodings, masks in p_batches:
            predictions = model(input_sequences, training=False)
            accuracy_tracker.update_state(label_sequences, predictions, sample_weight=masks)
            p_batches.set_postfix(result=accuracy_tracker.result().numpy())

        accuracy = accuracy_tracker.result().numpy()
        self.eval_history[theme].append(accuracy)
        return accuracy







from model import get_pretrain_model as get_model

if __name__ == '__main__':
    checkpoint_path = config.model_path
    model = get_model(checkpoint_path=checkpoint_path)

    ae = AbstractEval()
    results = ae.run_eval(model)

    # Nicely format and print dictionary results
    for theme, accs in results.items():
        print(theme, accs[-1])






