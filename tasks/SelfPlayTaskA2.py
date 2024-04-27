import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import math
import chess.pgn
import json
import config
import matplotlib.pyplot as plt
import os
from tasks.AbstractTask import AbstractTask
import scipy.signal
from model import get_vs_models as get_model
from collections import OrderedDict
import tensorflow_addons as tfa
import chess
import chess.engine
from stockfish.rewards.reward_2 import calc_reward
from evals.Arch2Evals import Arch2Evals as Evals
from evals.plotting.training_comparison import plot_training_comparison
from stockfish.utils import combine_alternate


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



# Number of self-play games in a mini-batch
global_mini_batch_size = 64

run_evals = True
use_actor_warmup = True
critic_warmup_epochs = 0

pickup_epoch = 0

run_dir = 1004
run_dir_itr = 0

top_k = None

# set seeds
seed = 1
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


class SelfPlayTaskA2(AbstractTask):

    def __init__(
            self,
            run_num=0,
            problem=None,
            epochs=50,
            actor_load_path=None,
            actor_2_load_path=None,
            critic_load_path=None,
            debug=False,
            run_val=False,
            val_itr=0,
    ):
        super(SelfPlayTaskA2, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.run_val = run_val
        self.val_itr = val_itr
        self.actor_2_load_path = actor_2_load_path

        # Evals
        self.eval_themes = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
        self.eval = Evals(themes=self.eval_themes)
        self.eval2 = Evals(themes=self.eval_themes)

        # Algorithm parameters
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.max_steps_per_game = config.seq_length - 1  # 30 | 60








