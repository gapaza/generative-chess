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
from model import get_rl_models_a2 as get_model
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

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.001  # was 0.0001
        self.entropy_coef = 0.00  # was 0.02 originally
        self.counter = 0
        self.game_start_token_id = config.start_token_id
        self.num_actions = config.vocab_size
        self.curr_epoch = 0
        self.actor_updates = 0

        # Results
        self.plot_freq = 10

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')

        # Critic save
        self.critic_save_dir = os.path.join(self.run_dir, 'critic')
        if not os.path.exists(self.critic_save_dir):
            os.makedirs(self.critic_save_dir)
        self.critic_save_path = os.path.join(self.critic_save_dir, 'critic_weights')

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 32, "Hash": 4096 * 2})
        self.nodes = 50000
        self.lines = 1

        # Additional epoch info
        self.additional_info = {
            'model_1_checkmates': [],
            'model_2_checkmates': [],
            'model_1_illegal_moves': [],
            'model_2_illegal_moves': [],
            'model_1_win_predictions': [],
            'model_2_win_predictions': [],

            'model_1_loss': [],
            'model_2_loss': [],
            'model_1_entropy': [],
            'model_2_entropy': [],
            'model_1_kl': [],
            'model_2_kl': [],
            'model_1_return': [],
            'model_2_return': [],
            'game_len': [],
        }




    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.00001  # 0.0001
        self.critic_learning_rate = 0.00001  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40

        # Optimizers
        if self.actor_optimizer is None:
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
            self.actor_optimizer_2 = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_actor_2, self.c_critic = get_model(self.actor_load_path, self.actor_2_load_path, self.critic_load_path)



    def run(self):
        self.build()

        # self.run_evals()
        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()

            self.record(epoch_info)

            if self.curr_epoch % 50 == 0 and self.curr_epoch > 0:
                t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch + pickup_epoch))
                t_actor_save_path_2 = os.path.join(self.pretrain_save_dir, 'actor_weights_2_' + str(self.curr_epoch + pickup_epoch))
                t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch + pickup_epoch))
                self.c_actor.save_weights(t_actor_save_path)
                self.c_actor_2.save_weights(t_actor_save_path_2)
                self.c_critic.save_weights(t_critic_save_path)
                # self.c_critic.save_weights(self.critic_save_path)



        # Save the parameters of the current actor and critic
        self.c_actor.save_weights(self.actor_pretrain_save_path)
        self.c_actor_2.save_weights(self.actor_pretrain_save_path)
        self.c_critic.save_weights(self.critic_pretrain_save_path)




    def fast_mini_batch(self):
        total_eval_time = 0

        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards_post = [[] for _ in range(self.mini_batch_size)]
        all_logprobs = [[] for _ in range(self.mini_batch_size)]
        games = [[] for x in range(self.mini_batch_size)]
        epoch_games = []
        observation = [[self.game_start_token_id] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]
        ended_games = [False for x in range(self.mini_batch_size)]
        boards = [chess.Board() for x in range(self.mini_batch_size)]
        game_evals = [0.2 for x in range(self.mini_batch_size)] # Eval from white perspective
        action_mask = [[] for x in range(self.mini_batch_size)]

        # -------------------------------------
        # Sample Actors
        # -------------------------------------
        player_models = [
            1,  # self.c_actor
            2  # self.c_actor_2
        ]
        random.shuffle(player_models)
        white_model = player_models[0]
        black_model = player_models[1]
        self.additional_info['model_1_checkmates'].append(0)
        self.additional_info['model_2_checkmates'].append(0)
        self.additional_info['model_1_illegal_moves'].append(0)
        self.additional_info['model_2_illegal_moves'].append(0)
        self.additional_info['model_1_win_predictions'].append(0)
        self.additional_info['model_2_win_predictions'].append(0)


        for t in range(self.max_steps_per_game):
            if t % 2 == 0:  # even number
                actor = white_model
                white_turn = True
            else:  # odd number
                actor = black_model
                white_turn = False

                # 2. Sample actions
                action_log_prob, action, all_action_probs, batch_logprobs = self.sample_actor(observation, white_turn, actor=actor)  # returns shape: (batch,) and (batch,)

                # 3. Update the game state
                observation_new = deepcopy(observation)
                for idx, act in enumerate(action):
                    if ended_games[idx] is True:
                        all_actions[idx].append(0)
                        all_logprobs[idx].append(0)
                        m_action = int(0)
                        observation_new[idx].append(m_action)
                        action_mask[idx].append(0)
                    else:
                        all_actions[idx].append(deepcopy(act))
                        all_logprobs[idx].append(action_log_prob[idx])
                        m_action = int(deepcopy(act))
                        games[idx].append(m_action)
                        observation_new[idx].append(m_action)
                        action_mask[idx].append(1)


















    def sample_actor(self, observation, white_turn, actor=1):
        white_moves = [config.token2id['[white]']]
        black_moves = [config.token2id['[black]']]
        for idx, obs in enumerate(observation):
            if idx % 2 == 0:
                white_moves.append(obs)
            else:
                black_moves.append(obs)
        white_moves = tf.convert_to_tensor([white_moves], dtype=tf.int32)
        black_moves = tf.convert_to_tensor([black_moves], dtype=tf.int32)
        is_white = tf.convert_to_tensor([white_turn], dtype=tf.bool)

        if white_turn is True:
            inf_idx = len(white_moves) - 1  # all batch elements have the same length
            model_moves = white_moves
            cross_moves = black_moves
        else:
            inf_idx = len(black_moves) - 1  # all batch elements have the same length
            model_moves = black_moves
            cross_moves = white_moves

        if actor == 1:
            return self._sample_actor_1(model_moves, cross_moves, is_white, inf_idx)
        else:
            return self._sample_actor_2(model_moves, cross_moves, is_white, inf_idx)


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor_1(self, model_moves, cross_moves, is_white, inf_idx):
        observation_input = [model_moves, cross_moves, is_white]
        pred_probs, pred_values = self.c_actor(observation_input)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))
        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs, all_token_log_probs


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor_2(self, model_moves, cross_moves, is_white, inf_idx):
        observation_input = [model_moves, cross_moves, is_white]
        pred_probs, pred_values = self.c_actor_2(observation_input)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))
        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs, all_token_log_probs



