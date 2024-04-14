import os
import pickle
from datetime import datetime
import tensorflow as tf
import platform


# Tensorflow Core
mixed_precision = False
if platform.system() != 'Darwin':
    if mixed_precision is True:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
else:
    tf.config.set_visible_devices([], 'GPU')


# Distributed Training
distributed = False
mirrored_strategy = tf.distribute.MirroredStrategy()
global_batch_size = 32  # 64, 128, 256, 512, 1024



#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'generative-chess')
tokens_dir = os.path.join(root_dir, 'tokens')

datasets_dir = os.path.join(root_dir, 'datasets')
weights_dir = os.path.join(root_dir, 'weights')
results_dir = os.path.join(root_dir, 'results')
games_dir = os.path.join(root_dir, 'games')

if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(games_dir):
    os.makedirs(games_dir)



stockfish_path = os.path.join(root_dir, 'stockfish', 'stockfish', 'stockfish-ubuntu-x86-64-avx2')



#
#       __  __             _        _
#      |  \/  |           | |      | |
#      | \  / |  ___    __| |  ___ | |
#      | |\/| | / _ \  / _` | / _ \| |
#      | |  | || (_) || (_| ||  __/| |
#      |_|  |_| \___/  \__,_| \___||_|
#

model_path = os.path.join(weights_dir, 'chess-gpt-v1')

dense_dim = 2048
heads = 8
seq_length = 128  # 128 nominal
embed_dim = 256   # 256 nominal
num_experts = 8

# --> Dropout
dropout = 0.1

# --> Training
pt_dataset = os.path.join(datasets_dir, 'test-dataset')
epochs = 200




#
#      __      __                 _             _
#      \ \    / /                | |           | |
#       \ \  / /___    ___  __ _ | |__   _   _ | |  __ _  _ __  _   _
#        \ \/ // _ \  / __|/ _` || '_ \ | | | || | / _` || '__|| | | |
#         \  /| (_) || (__| (_| || |_) || |_| || || (_| || |   | |_| |
#          \/  \___/  \___|\__,_||_.__/  \__,_||_| \__,_||_|    \__, |
#                                                                __/ |
#                                                               |___/
#

import tensorflow as tf
from keras.layers import TextVectorization
import re

def custom_standardization(input_data):
    # lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


move_language = 'uci'  # 'uci' | 'san'
if move_language == 'uci':
    vocab_file = os.path.join(tokens_dir, 'tokens_1969_merged.pkl')  # tokens_1966.pkl, tokens_1968_chesscom
elif move_language == 'san':
    vocab_file = os.path.join(tokens_dir, 'tokens_san_9940.pkl')

end_of_game_tokens = ["[white]", "[black]", "[draw]"]
special_tokens = ["[pos]", "[mask]", '[start]']

num_special_tokens = len(special_tokens) + 2
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    # remove empty string and [UNK]
    if '' in vocab:
        vocab.remove('')
    vocab.sort()
vocab = special_tokens + vocab + end_of_game_tokens
vocab_size = len(vocab)
tokenizer = TextVectorization(
    max_tokens=vocab_size + 2,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=seq_length,
)
tokenizer.set_vocabulary(vocab)
vocab = tokenizer.get_vocabulary()
vocab_size = len(vocab)
mask_token_id = tokenizer(["[mask]"]).numpy()[0][0]
padding_token_id = tokenizer(['']).numpy()[0][0]
pos_token_id = tokenizer(["[pos]"]).numpy()[0][0]
white_token_id = tokenizer(["[white]"]).numpy()[0][0]
black_token_id = tokenizer(["[black]"]).numpy()[0][0]
draw_token_id = tokenizer(["[draw]"]).numpy()[0][0]
start_token_id = tokenizer(["[start]"]).numpy()[0][0]
id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}

def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()

@tf.function
def encode_tf(input):
    encoded_input = tokenizer(input)
    # encoded_input = tf.reshape(encoded_input, (-1,))
    # if tf.rank(encoded_input) > 1:
    #     encoded_input = tf.squeeze(encoded_input, axis=0)
    return encoded_input
