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

if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)



stockfish_path = os.path.join(root_dir, 'stockfish', 'stockfish', 'stockfish-ubuntu-x86-64-avx2')













