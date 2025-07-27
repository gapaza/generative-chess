import numpy as np
import time
import tensorflow as tf
import keras
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
from model import get_rl_models_a3 as get_model
from collections import OrderedDict
# import tensorflow_addons as tfa
import chess
import chess.engine
from stockfish.rewards.reward_3 import calc_reward
from evals.Arch3Evals import Arch3Evals as Evals
from evals.plotting.training_comparison import plot_training_comparison
from stockfish.utils import combine_alternate
from utils import save_game_pgn
from utils import get_inputs_from_game_a3 as get_inputs_from_game
from utils import get_engine_move



def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



# Number of self-play games in a mini-batch
global_mini_batch_size = 64

run_evals = False
use_actor_warmup = False
critic_warmup_epochs = 0

pickup_epoch = 0

run_dir = 22
run_dir_itr = 0

top_k = None

# set seeds
seed = 0
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)





class SelfPlayTaskA3(AbstractTask):

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
        super(SelfPlayTaskA3, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.run_val = run_val
        self.val_itr = val_itr
        self.actor_2_load_path = actor_2_load_path

        # Evals
        self.eval_themes = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
        # self.eval = Evals(themes=self.eval_themes)
        # self.eval2 = Evals(themes=self.eval_themes)

        # Algorithm parameters
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        # self.max_steps_per_game = config.seq_length - 1  # 30 | 60
        self.max_steps_per_game = 40  # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.002  # was 0.00015
        self.entropy_coef = 0.04  # was 0.02 originally, 0.01 before
        self.counter = 0
        self.game_start_token_id = config.start_token_id
        self.num_actions = config.vocab_size
        self.curr_epoch = 0
        self.actor_updates = 0

        # Results
        self.plot_freq = 50

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
        print('Using Stockfish engine at:', config.stockfish_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({
            'Threads': config.AVAILABLE_CPUS, 
            "Hash": 4096 * 2, 
            "UCI_LimitStrength": True, 
            "UCI_Elo": 1350
        })
        self.nodes = 1000000
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
        self.train_actor_iterations = 20  # was 250
        self.train_critic_iterations = 80  # was 40

        # Warmup Learning Rate
        self.actor_learning_rate_1 = tf.keras.optimizers.schedules.CosineDecay(
            0.0,
            100000,
            alpha=0.1,
            warmup_target=self.actor_learning_rate,
            warmup_steps=100
        )
        self.actor_learning_rate_2 = tf.keras.optimizers.schedules.CosineDecay(
            0.0,
            100000,
            alpha=0.1,
            warmup_target=self.actor_learning_rate,
            warmup_steps=100
        )
        self.critic_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            0.0,
            100000,
            alpha=0.1,
            warmup_target=self.critic_learning_rate,
            warmup_steps=100
        )


        # Optimizers
        if self.actor_optimizer is None:
            self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_learning_rate_1)
            self.actor_optimizer_2 = keras.optimizers.Adam(learning_rate=self.actor_learning_rate_2)
        if self.critic_optimizer is None:
            self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_learning_rate)


        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.c_actor, self.c_actor_2, self.c_critic = get_model(self.actor_load_path, self.actor_2_load_path, self.critic_load_path)


    def run(self):
        self.build()

        # self.run_evals()
        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()

            self.record()

            if self.curr_epoch % 200 == 0 and self.curr_epoch > 0:
                # t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch + pickup_epoch) + '.weights.h5')
                t_actor_save_path_2 = os.path.join(self.pretrain_save_dir, 'actor_weights_2_' + str(self.curr_epoch + pickup_epoch) + '.weights.h5')
                t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch + pickup_epoch) + '.weights.h5')
                # self.c_actor.save_weights(t_actor_save_path)
                self.c_actor_2.save_weights(t_actor_save_path_2)
                self.c_critic.save_weights(t_critic_save_path)
                # self.c_critic.save_weights(self.critic_save_path)

        # Save the parameters of the current actor and critic
        self.c_actor.save_weights(self.actor_pretrain_save_path)
        self.c_actor_2.save_weights(self.actor_pretrain_save_path)
        self.c_critic.save_weights(self.critic_pretrain_save_path)


    def pad_list(self, sequence, pad_token=0, pad_len=None):
        if pad_len is None:
            pad_len = config.seq_length
        while len(sequence) < pad_len:
            sequence.append(pad_token)
        if len(sequence) > pad_len:
            sequence = sequence[:pad_len]
        return sequence





    def fast_mini_batch(self):
        total_eval_time = 0
        # print('Starting mini-batch...')


        observation = [[] for x in range(self.mini_batch_size)]
        critic_observation_buffer = [[] for x in range(self.mini_batch_size)]
        ended_games = [False for x in range(self.mini_batch_size)] # Records if a game has ended
        nat_ended_games = [False for x in range(self.mini_batch_size)] # Records if a game has ended naturally (not by padding)
        boards = [chess.Board() for x in range(self.mini_batch_size)]
        game_evals = [0.2 for x in range(self.mini_batch_size)] # Eval from white perspective
        action_mask = [[] for x in range(self.mini_batch_size)]

        games = [[] for x in range(self.mini_batch_size)] # Each game is a list of token_ids
        epoch_games = []


        # --- Color Specific ---
        # - we can reconstruct the color-specific observations from the full observations with get_inputs_from_game
        white_actions = [[] for x in range(self.mini_batch_size)]
        black_actions = [[] for x in range(self.mini_batch_size)]
        white_action_mask = [[] for x in range(self.mini_batch_size)]
        black_action_mask = [[] for x in range(self.mini_batch_size)]
        white_logprobs = [[] for x in range(self.mini_batch_size)]
        black_logprobs = [[] for x in range(self.mini_batch_size)]

        # --- KL Divergence ---
        all_white_probs_full = [[] for _ in range(self.mini_batch_size)]
        all_black_probs_full = [[] for _ in range(self.mini_batch_size)]
        all_white_logprobs_full = [[] for _ in range(self.mini_batch_size)]
        all_black_logprobs_full = [[] for _ in range(self.mini_batch_size)]

        




        # -------------------------------------
        # Sample Actors
        # -------------------------------------
        curr_time = time.time()
        player_models = [
            1,  # self.c_actor
            2  # self.c_actor_2
        ]
        # random.shuffle(player_models)
        white_model = player_models[0]
        black_model = player_models[1]
        self.additional_info['model_1_checkmates'].append(0)
        self.additional_info['model_2_checkmates'].append(0)
        self.additional_info['model_1_illegal_moves'].append(0)
        self.additional_info['model_2_illegal_moves'].append(0)
        self.additional_info['model_1_win_predictions'].append(0)
        self.additional_info['model_2_win_predictions'].append(0)




        for t in range(self.max_steps_per_game):

            # 1. Select the actor to use based on the turn
            if t % 2 == 0:  # even number
                actor = white_model
                white_turn = True
            else:  # odd number
                actor = black_model
                white_turn = False


            # 2. Sample actions
            action_log_prob, action, all_action_probs, all_action_log_probs = self.sample_actor(observation, white_turn, actor=actor)  # returns shape: (batch,) and (batch,)

            # if actor == 1:
            #     action_log_prob, action, all_action_probs, all_action_log_probs = self.sample_actor(observation, white_turn, actor=actor)  # returns shape: (batch,) and (batch,)
            # else:
            #     action_log_prob, action, all_action_probs, all_action_log_probs = self.sample_engine(observation, white_turn, actor=actor)



            action_log_prob = action_log_prob.numpy().tolist()
            action = action.numpy().tolist()
            all_action_log_probs = all_action_log_probs.numpy().tolist()
            all_action_probs = all_action_probs.numpy().tolist()


            # 3. Update the game state
            observation_new = deepcopy(observation)
            for idx, act in enumerate(action):
                if ended_games[idx] is True:
                    # this move is not valid, the game has ended
                    m_action = int(0)  # If game has ended, set action to 0 for padding token
                    observation_new[idx].append(m_action)
                    action_mask[idx].append(0)
                    if white_turn is True:
                        white_action_mask[idx].append(0)
                        white_actions[idx].append(0)
                        white_logprobs[idx].append(0)
                    else:
                        black_action_mask[idx].append(0)
                        black_actions[idx].append(0)
                        black_logprobs[idx].append(0)
                else:
                    # we don't check if the move is valid or not, this is done next. plus we need to see invalid moves penalized
                    m_action = int(deepcopy(act))
                    games[idx].append(m_action)
                    observation_new[idx].append(m_action)
                    action_mask[idx].append(1)
                    if white_turn is True:
                        white_action_mask[idx].append(1)
                        white_actions[idx].append(m_action)
                        white_logprobs[idx].append(action_log_prob[idx])

                        all_white_probs_full[idx].append(all_action_probs[idx])
                        all_white_logprobs_full[idx].append(all_action_log_probs[idx])
                    else:
                        black_action_mask[idx].append(1)
                        black_actions[idx].append(m_action)
                        black_logprobs[idx].append(action_log_prob[idx])

                        all_black_probs_full[idx].append(all_action_probs[idx])
                        all_black_logprobs_full[idx].append(all_action_log_probs[idx])

            # 4. Check if on last epoch
            if t == (self.max_steps_per_game - 1):
                # print('on last epoch', t, self.max_steps_per_game)
                done = True
            else:
                # print('not on last epoch', t, self.max_steps_per_game)
                done = False
            
            # 5. Check if the games have ended
            for idx, game in enumerate(games):
                if ended_games[idx] is False:
                    ge = self.has_game_ended(game, actor)
                    if done is True:
                        game_ended = True # Force end of game if we are on the last epoch
                    else:
                        game_ended = ge
                    if game_ended is True:
                        epoch_games.append(' '.join([config.id2token[x] for x in game]))
                    ended_games[idx] = game_ended
                    nat_ended_games[idx] = ge

            # 6. Check if all games have ended
            if all(ended_games):
                done = True


            # 7. Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
                break
            else:
                observation = observation_new


        # Create action masks for critic
        # - the critic sees one additional position, as it predicts a value for all the actions AND the final state
        white_action_mask_critic = deepcopy(white_action_mask)
        for idx in range(len(white_action_mask_critic)):
            white_action_mask_critic[idx].append(1)  # Add padding token for critic
        for trajectory in white_action_mask_critic:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)

        black_action_mask_critic = deepcopy(black_action_mask)
        for idx in range(len(black_action_mask_critic)):
            black_action_mask_critic[idx].append(1)
        for trajectory in black_action_mask_critic:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)


        # Count the number of checkmates + finish trajectories
        for trajectory in observation:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length-1)
        for trajectory in action_mask:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length-1)
        for trajectory in critic_observation_buffer:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in white_action_mask:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in black_action_mask:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in white_logprobs:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in black_logprobs:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in white_actions:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in black_actions:
            trajectory = self.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)

        # Pad probs and logprobs for kl divergence calc
        for trajectory in all_white_logprobs_full:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])
        for trajectory in all_white_probs_full:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])
        for trajectory in all_black_logprobs_full:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])
        for trajectory in all_black_probs_full:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])


        # print('Saving game...')
        save_game_pgn(games[0], self.run_dir)
        # print('Game 1:', ' '.join([config.id2token[x] for x in games[0]]))
        # exit(0)

        # print('Time sample actors:', time.time() - curr_time)
        # -------------------------------------
        # Post process rewards
        # -------------------------------------
        curr_time = time.time()

        all_white_rewards = []
        all_black_rewards = []
        for idx, game in enumerate(games):
            uci_game = ' '.join([config.id2token[x] for x in game])
            white_rewards, black_rewards = calc_reward(self.engine, uci_game, n=self.nodes, pad=False)
            all_white_rewards.append(white_rewards)
            all_black_rewards.append(black_rewards)

        avg_white_reward = np.mean([np.sum(x) for x in all_white_rewards])
        avg_black_reward = np.mean([np.sum(x) for x in all_black_rewards])


        # print('\n\n-------------------------------------')
        # print('White rewards g1:', all_white_rewards[0])
        # print('Black rewards g1:', all_black_rewards[0])
        # exit(0)
    
        # print('Time post process rewards:', time.time() - curr_time)
        # -------------------------------------
        # Sample Critic
        # -------------------------------------
        curr_time = time.time()

        white_values = []
        black_values = []

        white_value_t = self.sample_critic(critic_observation_buffer, white_turn=True)
        white_value_t = white_value_t.numpy().tolist()  # (30, 31)
        for idx, values in zip(range(self.mini_batch_size), white_value_t):
            values_mask = white_action_mask[idx]
            w_values = []
            # for mask, v in zip(values_mask, values):
            #     if mask == 1:
            #         w_values.append(v)
            for idx2, mask in enumerate(values_mask):
                v = values[idx2]
                if mask == 1:
                    w_values.append(v)
                if mask == 0 and values_mask[idx2-1] == 1:  # also append value prediction after last action

                    # If the game ended naturally, we do not use the extra value predition
                    if nat_ended_games[idx] is True:
                        all_white_rewards[idx].append(0)
                        w_values.append(0)
                    else:
                        all_white_rewards[idx].append(v)
                        w_values.append(v)


            white_values.append(w_values)


        black_value_t = self.sample_critic(critic_observation_buffer, white_turn=False)
        black_value_t = black_value_t.numpy().tolist()  # (30, 31)
        for idx, values in zip(range(self.mini_batch_size), black_value_t):
            values_mask = black_action_mask[idx]
            b_values = []

            for idx2, mask in enumerate(values_mask):
                v = values[idx2]
                if mask == 1:
                    b_values.append(v)
                if mask == 0 and values_mask[idx2-1] == 1:  # also append value prediction after last action

                    if nat_ended_games[idx] is True:
                        all_black_rewards[idx].append(0)
                        b_values.append(0)
                    else:
                        all_black_rewards[idx].append(v)
                        b_values.append(v)




            black_values.append(b_values)

        # print('\n\n-------------------------------------')
        # print('White rewards g1:', all_white_rewards[0])
        # print('White values g1:', white_values[0])

        # print('Black rewards g1:', all_black_rewards[0])
        # print('Black values g1:', black_values[0])
        # exit(0)

        # print('Time sample critic:', time.time() - curr_time)
        # -------------------------------------
        # Calculate Advantage and Returns
        # -------------------------------------
        curr_time = time.time()

        all_white_advantages = []
        all_white_returns = []
        cum_white_adv = []
        for idx in range(len(all_white_rewards)):
            # print('White rewards:', len(all_white_rewards[idx]), all_white_rewards[idx])
            # print('White values:', len(white_values[idx]), white_values[idx])
            w_rewards = np.array(all_white_rewards[idx])
            w_values = np.array(white_values[idx])
            w_deltas = w_rewards[:-1] + self.gamma * w_values[1:] - w_values[:-1]
            # w_deltas = w_rewards[:-1] + self.gamma * w_rewards[1:] - w_rewards[:-1]
            w_advantages = discounted_cumulative_sums(w_deltas, self.gamma * self.lam)
            all_white_advantages.append(w_advantages)
            w_returns = discounted_cumulative_sums(w_rewards, self.gamma).tolist()
            w_returns = self.pad_list(w_returns, pad_token=0, pad_len=config.seq_length)
            all_white_returns.append(w_returns)
            cum_white_adv.extend(w_advantages)
        avg_white_adv = np.mean(cum_white_adv)
        std_white_adv = np.std(cum_white_adv)
        for idx in range(len(all_white_advantages)):
            all_white_advantages[idx] = ((all_white_advantages[idx] - avg_white_adv) / (std_white_adv + 1e-10)).tolist()
            all_white_advantages[idx] = self.pad_list(all_white_advantages[idx], pad_token=0, pad_len=config.seq_length)
        all_white_advantages = tf.convert_to_tensor(all_white_advantages, dtype=tf.float32)

        all_black_advantages = []
        all_black_returns = []
        cum_black_adv = []
        for idx in range(len(all_black_rewards)):
            b_rewards = np.array(all_black_rewards[idx])
            b_values = np.array(black_values[idx])
            b_deltas = b_rewards[:-1] + self.gamma * b_values[1:] - b_values[:-1]
            # b_deltas = b_rewards[:-1] + self.gamma * b_rewards[1:] - b_rewards[:-1]
            b_advantages = discounted_cumulative_sums(b_deltas, self.gamma * self.lam)
            all_black_advantages.append(b_advantages)
            b_returns = discounted_cumulative_sums(b_rewards, self.gamma).tolist()
            b_returns = self.pad_list(b_returns, pad_token=0, pad_len=config.seq_length)
            all_black_returns.append(b_returns)
            cum_black_adv.extend(b_advantages)
        avg_black_adv = np.mean(cum_black_adv)
        std_black_adv = np.std(cum_black_adv)
        for idx in range(len(all_black_advantages)):
            all_black_advantages[idx] = ((all_black_advantages[idx] - avg_black_adv) / (std_black_adv + 1e-10)).tolist()
            all_black_advantages[idx] = self.pad_list(all_black_advantages[idx], pad_token=0, pad_len=config.seq_length)
        all_black_advantages = tf.convert_to_tensor(all_black_advantages, dtype=tf.float32)

        # print('Time calculate advantages and returns:', time.time() - curr_time)
        # -------------------------------------
        # Construct model inputs
        # -------------------------------------
        curr_time = time.time()

        white_model_inputs = []
        white_cross_inputs = []
        white_color_inputs = []
        black_model_inputs = []
        black_cross_inputs = []
        black_color_inputs = []
        for idx in range(len(observation)):
            wi, wci, w_inf_idx = get_inputs_from_game(observation[idx], white_turn=True)
            wi = self.pad_list(wi, pad_token=0, pad_len=config.seq_length)
            wci = self.pad_list(wci, pad_token=0, pad_len=config.seq_length)
            white_model_inputs.append(wi)
            white_cross_inputs.append(wci)
            white_color_inputs.append(True)

            bi, bci, b_inf_idx = get_inputs_from_game(observation[idx], white_turn=False)
            bi = self.pad_list(bi, pad_token=0, pad_len=config.seq_length)
            bci = self.pad_list(bci, pad_token=0, pad_len=config.seq_length)
            black_model_inputs.append(bi)
            black_cross_inputs.append(bci)
            black_color_inputs.append(False)

        # print('Time loop over inputs:', time.time() - curr_time)
        curr_time = time.time()

        white_model_inputs = tf.convert_to_tensor(np.array(white_model_inputs, dtype=np.float32))
        white_cross_inputs = tf.convert_to_tensor(np.array(white_cross_inputs, dtype=np.float32))
        white_color_inputs = tf.convert_to_tensor(np.array(white_color_inputs, dtype=bool))

        black_model_inputs = tf.convert_to_tensor(np.array(black_model_inputs, dtype=np.float32))
        black_cross_inputs = tf.convert_to_tensor(np.array(black_cross_inputs, dtype=np.float32))
        black_color_inputs = tf.convert_to_tensor(np.array(black_color_inputs, dtype=bool))

        white_actions = tf.convert_to_tensor(np.array(white_actions, dtype=np.int32))
        black_actions = tf.convert_to_tensor(np.array(black_actions, dtype=np.int32))

        white_logprobs = tf.convert_to_tensor(np.array(white_logprobs, dtype=np.float32))
        black_logprobs = tf.convert_to_tensor(np.array(black_logprobs, dtype=np.float32))

        white_action_mask = tf.convert_to_tensor(np.array(white_action_mask, dtype=np.float32))
        black_action_mask = tf.convert_to_tensor(np.array(black_action_mask, dtype=np.float32))

        white_action_mask_critic = tf.convert_to_tensor(np.array(white_action_mask_critic, dtype=np.float32))
        black_action_mask_critic = tf.convert_to_tensor(np.array(black_action_mask_critic, dtype=np.float32))

        all_white_probs_full = tf.convert_to_tensor(np.array(all_white_probs_full, dtype=np.float32))
        all_black_probs_full = tf.convert_to_tensor(np.array(all_black_probs_full, dtype=np.float32))

        all_white_logprobs_full = tf.convert_to_tensor(np.array(all_white_logprobs_full, dtype=np.float32))
        all_black_logprobs_full = tf.convert_to_tensor(np.array(all_black_logprobs_full, dtype=np.float32))

        # print('Time convert full logprobs to tensor:', time.time() - curr_time)
        # -------------------------------------
        # Train Actor 1
        # -------------------------------------
        # print('Training Actor 1...')

        if player_models.index(1) == 0:
            model_1_inputs = white_model_inputs
            model_1_cross = white_cross_inputs
            model_1_color = white_color_inputs
            model_1_action_buffer = white_actions
            model_1_logprobs = white_logprobs
            model_1_action_mask = white_action_mask
            model_1_advantages = all_white_advantages
            model_1_logprobs_full = all_white_logprobs_full
            model_1_probs_full = all_white_probs_full
        else:
            model_1_inputs = black_model_inputs
            model_1_cross = black_cross_inputs
            model_1_color = black_color_inputs
            model_1_action_buffer = black_actions
            model_1_logprobs = black_logprobs
            model_1_action_mask = black_action_mask
            model_1_advantages = all_black_advantages
            model_1_logprobs_full = all_black_logprobs_full
            model_1_probs_full = all_black_probs_full


        policy_update_itr = 0
        kl_1, entr_1, loss_1, policy_loss_1 = 0, 0, 0, 0
        if self.curr_epoch >= critic_warmup_epochs:
            for i in range(self.train_actor_iterations):
                policy_update_itr += 1
                kl_1, entr_1, loss_1, policy_loss_1 = self.train_actor_1(
                    model_1_inputs, model_1_cross, model_1_color,
                    model_1_action_buffer,
                    model_1_logprobs,
                    model_1_action_mask,
                    model_1_advantages,

                    model_1_logprobs_full,
                    model_1_probs_full
                )
                if abs(kl_1) > self.target_kl:
                    # Early Stopping
                    break
            kl_1 = kl_1.numpy()
            entr_1 = entr_1.numpy()
            loss_1 = loss_1.numpy()
            policy_loss_1 = policy_loss_1.numpy()
        self.actor_updates += policy_update_itr


        # # -------------------------------------
        # # Train Actor 2
        # # -------------------------------------
        # curr_time = time.time()

        # if player_models.index(2) == 0:
        #     model_2_inputs = white_model_inputs
        #     model_2_cross = white_cross_inputs
        #     model_2_color = white_color_inputs
        #     model_2_action_buffer = white_actions
        #     model_2_logprobs = white_logprobs
        #     model_2_action_mask = white_action_mask
        #     model_2_advantages = all_white_advantages
        #     model_2_logprobs_full = all_white_logprobs_full
        #     model_2_probs_full = all_white_probs_full
        # else:
        #     model_2_inputs = black_model_inputs
        #     model_2_cross = black_cross_inputs
        #     model_2_color = black_color_inputs
        #     model_2_action_buffer = black_actions
        #     model_2_logprobs = black_logprobs
        #     model_2_action_mask = black_action_mask
        #     model_2_advantages = all_black_advantages
        #     model_2_logprobs_full = all_black_logprobs_full
        #     model_2_probs_full = all_black_probs_full

        # policy_update_itr = 0
        kl_2, entr_2, loss_2, policy_loss_2 = 0, 0, 0, 0
        # if self.curr_epoch >= critic_warmup_epochs:
        #     for i in range(self.train_actor_iterations):
        #         policy_update_itr += 1
        #         kl_2, entr_2, loss_2, policy_loss_2 = self.train_actor_2(
        #             model_2_inputs, model_2_cross, model_2_color,
        #             model_2_action_buffer,
        #             model_2_logprobs,
        #             model_2_action_mask,
        #             model_2_advantages,

        #             model_2_logprobs_full,
        #             model_2_probs_full
        #         )
        #         if abs(kl_2) > self.target_kl:
        #             # Early Stopping
        #             break
        #     kl_2 = kl_2.numpy()
        #     entr_2 = entr_2.numpy()
        #     loss_2 = loss_2.numpy()
        #     policy_loss_2 = policy_loss_2.numpy()
        # self.actor_updates += policy_update_itr


        # print('Time train actor 2:', time.time() - curr_time)
        # -------------------------------------
        # Train Critic
        # -------------------------------------
        # print('Training Critic...')
        curr_time = time.time()

        combined_model_inputs = tf.concat([white_model_inputs, black_model_inputs], axis=0)
        combined_cross_inputs = tf.concat([white_cross_inputs, black_cross_inputs], axis=0)
        combined_color_inputs = tf.concat([white_color_inputs, black_color_inputs], axis=0)
        # combined_mask_inputs = tf.concat([white_action_mask, black_action_mask], axis=0)
        combined_mask_inputs = tf.concat([white_action_mask_critic, black_action_mask_critic], axis=0)

        all_white_returns_tensor = tf.convert_to_tensor(all_white_returns, dtype=tf.float32)
        all_black_returns_tensor = tf.convert_to_tensor(all_black_returns, dtype=tf.float32)
        combined_returns_tensor = tf.concat([all_white_returns_tensor, all_black_returns_tensor], axis=0)
        combined_returns_tensor = tf.expand_dims(combined_returns_tensor, axis=-1)


        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                combined_model_inputs, combined_cross_inputs, combined_color_inputs,
                combined_returns_tensor,  combined_mask_inputs
            )

        self.c_loss.append(value_loss)

        # print('Time train critic:', time.time() - curr_time)
        # -------------------------------------
        # Run Statistics
        # -------------------------------------

        avg_game_len = np.mean([len(x) for x in games])

        # Update additional info
        self.additional_info['model_1_loss'].append(loss_1)
        self.additional_info['model_2_loss'].append(loss_2)
        self.additional_info['model_1_entropy'].append(entr_1)
        self.additional_info['model_2_entropy'].append(entr_2)
        self.additional_info['model_1_kl'].append(kl_1)
        self.additional_info['model_2_kl'].append(kl_2)

        avg_game_len = np.mean([len(x) for x in games])
        self.additional_info['game_len'].append(avg_game_len)


        if player_models[0] == 1:
            self.additional_info['model_1_return'].append(avg_white_reward)
            self.additional_info['model_2_return'].append(avg_black_reward)
        else:
            self.additional_info['model_1_return'].append(avg_black_reward)
            self.additional_info['model_2_return'].append(avg_white_reward)



    

    
    
    
    
    
    
    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None,), dtype=tf.bool),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    # ])

    @tf.function(jit_compile=True)
    def train_critic(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            return_buffer, mask_tensor
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            preds, pred_values = self.c_critic(inputs)  # (batch, seq_len, 2)
            value_loss = self.mse_loss(return_buffer, pred_values, sample_weight=mask_tensor)

            if config.mixed_precision is True:
                value_loss = self.critic_optimizer.get_scaled_loss(value_loss)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        if config.mixed_precision is True:
            critic_grads = self.critic_optimizer.get_unscaled_gradients(critic_grads)

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))
        return value_loss

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None,), dtype=tf.bool),

    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),

    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    # ])
    @tf.function(jit_compile=True)
    def train_actor_1(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            action_buffer,
            logprobability_buffer,
            color_action_mask_tensor,
            color_advantage_buffer,

            color_logprobs_full,
            color_probs_full
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:
            # pred_probs, pred_vals = self.c_actor(inputs)  # shape: (batch, seq_len, 2)
            # pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)

            # # --- Find true advantage
            # pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape: (batch, seq_len, 2)
            # pred_log_probs = tf.cast(pred_log_probs, dtype=tf.float32)
            # one_hot_actions = tf.one_hot(action_buffer, self.num_actions)
            # one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
            # logprobability = tf.reduce_sum(
            #     one_hot_actions * pred_log_probs, axis=-1
            # )  # shape (batch, seq_len)
            # color_logprobs = logprobability * color_action_mask_tensor
            # color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            # white_ratio = tf.exp(
            #     color_logprobs - color_logprobs_buffer
            # )
            # white_true_advantage = white_ratio * color_advantage_buffer

            # # --- Find min advantage
            # white_min_advantage = tf.where(
            #     color_advantage_buffer > 0,
            #     (1 + self.clip_ratio) * color_advantage_buffer,
            #     (1 - self.clip_ratio) * color_advantage_buffer,
            # )

            # policy_loss = -tf.reduce_mean(
            #     tf.minimum(white_true_advantage, white_min_advantage)
            # )
            # loss = 0
            # loss += policy_loss

            # entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            # entr = tf.reduce_mean(entr * color_action_mask_tensor)  # Higher positive value means more exploration - shape (batch,)
            # loss = loss - (self.entropy_coef * entr)

            # if config.mixed_precision is True:
            #     loss = self.actor_optimizer.get_scaled_loss(loss)


            logits, _ = self.c_actor(inputs)
            probs     = tf.nn.softmax(logits, axis=-1)
            logp      = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.))

            onehot    = tf.one_hot(action_buffer, self.num_actions, dtype=tf.float32)
            new_logp  = tf.reduce_sum(onehot * logp, axis=-1)

            old_logp  = tf.stop_gradient(logprobability_buffer)
            A_t       = tf.stop_gradient(color_advantage_buffer)

            ratio     = tf.exp(new_logp - old_logp)
            clipped   = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate = tf.minimum(ratio * A_t, clipped * A_t)

            surrogate = surrogate * color_action_mask_tensor
            policy_loss = -tf.reduce_mean(surrogate)

            entr = -tf.reduce_sum(probs * logp, axis=-1)
            entr = tf.reduce_mean(entr * color_action_mask_tensor)

            loss = policy_loss - self.entropy_coef * entr
            if config.mixed_precision:
                loss = self.actor_optimizer_2.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.c_actor.trainable_variables)
        if config.mixed_precision is True:
            gradients = self.actor_optimizer.get_unscaled_gradients(gradients)


        self.actor_optimizer.apply_gradients(zip(gradients, self.c_actor.trainable_variables))

        # True KL Divergence
        mask_tensor = tf.expand_dims(color_action_mask_tensor, axis=-1)
        pred_probs, pred_vals = self.c_actor(inputs)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        pred_log_probs = pred_log_probs * mask_tensor
        color_probs_full = color_probs_full * mask_tensor
        color_logprobs_full = color_logprobs_full * mask_tensor
        true_kl = tf.reduce_sum(
            color_probs_full * (color_logprobs_full - pred_log_probs),
            axis=-1
        )  # shape (9, 280)
        kl = tf.reduce_mean(true_kl)  # shape (1,)


        return kl, entr, loss, policy_loss

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None,), dtype=tf.bool),

    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),

    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    # ])
    @tf.function(jit_compile=True)
    def train_actor_2(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            action_buffer,
            logprobability_buffer,
            color_action_mask_tensor,
            color_advantage_buffer,

            color_logprobs_full,
            color_probs_full
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:
            # pred_probs, pred_vals = self.c_actor_2(inputs)  
            # pred_probs = tf.nn.softmax(pred_probs, axis=-1)  

            # # --- Find true advantage
            # pred_log_probs = tf.math.log(pred_probs)  
            # pred_log_probs = tf.cast(pred_log_probs, dtype=tf.float32)
            # one_hot_actions = tf.one_hot(action_buffer, self.num_actions)
            # one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
            # logprobability = tf.reduce_sum(
            #     one_hot_actions * pred_log_probs, axis=-1
            # )  
            # white_logprobs = logprobability * color_action_mask_tensor
            # white_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            # white_ratio = tf.exp(
            #     white_logprobs - white_logprobs_buffer
            # )
            # white_true_advantage = white_ratio * color_advantage_buffer

            # # --- Find min advantage
            # white_min_advantage = tf.where(
            #     color_advantage_buffer > 0,
            #     (1 + self.clip_ratio) * color_advantage_buffer,
            #     (1 - self.clip_ratio) * color_advantage_buffer,
            # )

            # policy_loss = -tf.reduce_mean(
            #     tf.minimum(white_true_advantage, white_min_advantage)
            # )
            # loss = 0
            # loss += policy_loss

            # entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  
            # entr = tf.reduce_mean(entr * color_action_mask_tensor)
            # loss = loss - (self.entropy_coef * entr)

            # if config.mixed_precision is True:
            #     loss = self.actor_optimizer_2.get_scaled_loss(loss)


            logits, _ = self.c_actor_2(inputs)
            probs     = tf.nn.softmax(logits, axis=-1)
            logp      = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.))

            onehot    = tf.one_hot(action_buffer, self.num_actions, dtype=tf.float32)
            new_logp  = tf.reduce_sum(onehot * logp, axis=-1)

            old_logp  = tf.stop_gradient(logprobability_buffer)
            A_t       = tf.stop_gradient(color_advantage_buffer)

            ratio     = tf.exp(new_logp - old_logp)
            clipped   = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate = tf.minimum(ratio * A_t, clipped * A_t)

            surrogate = surrogate * color_action_mask_tensor
            policy_loss = -tf.reduce_mean(surrogate)

            entr = -tf.reduce_sum(probs * logp, axis=-1)
            entr = tf.reduce_mean(entr * color_action_mask_tensor)

            loss = policy_loss - self.entropy_coef * entr
            if config.mixed_precision:
                loss = self.actor_optimizer_2.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.c_actor_2.trainable_variables)
        if config.mixed_precision is True:
            gradients = self.actor_optimizer_2.get_unscaled_gradients(gradients)

        self.actor_optimizer_2.apply_gradients(zip(gradients, self.c_actor_2.trainable_variables))

        # True KL Divergence
        mask_tensor = tf.expand_dims(color_action_mask_tensor, axis=-1)
        pred_probs, pred_vals = self.c_actor_2(inputs)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        pred_log_probs = pred_log_probs * mask_tensor
        color_probs_full = color_probs_full * mask_tensor
        color_logprobs_full = color_logprobs_full * mask_tensor
        true_kl = tf.reduce_sum(
            color_probs_full * (color_logprobs_full - pred_log_probs),
            axis=-1
        )  
        kl = tf.reduce_mean(true_kl) 
        return kl, entr, loss, policy_loss









    
    
    def has_game_ended(self, game, actor):
        uci_moves = [config.id2token[x] for x in game]
        last_move = uci_moves[-1]
        prev_moves = uci_moves[:-1]  # Remove the last move, which is the current move

        if actor == 1:
            prefix = 'model_1'
        else:
            prefix = 'model_2'

        board = chess.Board()
        for move in prev_moves:
            try:
                board.push(chess.Move.from_uci(move))
            except ValueError:
                raise ValueError(f"Invalid previous move in game: {move} in {uci_moves}")

        # --- check illegal token / move ---
        if last_move in config.non_move_tokens:
            print('Illegal token detected in game:', uci_moves)
            return True
        uci_move = chess.Move.from_uci(last_move)
        if uci_move not in board.legal_moves:
            self.additional_info[prefix + '_illegal_moves'][-1] += 1
            # print('Illegal move detected in game:', uci_moves)
            return True
        board.push(uci_move)

        # --- check if the game has ended ---
        if board.is_checkmate():
            self.additional_info[prefix + '_checkmates'][-1] += 1
            # print('Checkmate detected in game:', uci_moves)
            return True
        if board.is_stalemate():
            print('Stalemate detected in game:', uci_moves)
            return True

        return False
    
    
    def sample_actor(self, observation, white_turn, actor=1):
        all_model_moves = []
        all_cross_moves = []
        all_white_turns = []
        max_inf_idx = 0
        all_inf_idx = []
        for obs in observation:
            mm, cm, inf_idx = get_inputs_from_game(obs, white_turn)
            mm = self.pad_list(mm, pad_len=config.seq_length)
            cm = self.pad_list(cm, pad_len=config.seq_length)
            all_model_moves.append(mm)
            all_cross_moves.append(cm)
            all_white_turns.append(white_turn)
            all_inf_idx.append(inf_idx)
            if inf_idx > max_inf_idx:
                max_inf_idx = inf_idx
        inf_idx = max_inf_idx
        model_moves = tf.convert_to_tensor(all_model_moves, dtype=tf.int32)
        cross_moves = tf.convert_to_tensor(all_cross_moves, dtype=tf.int32)
        is_white = tf.convert_to_tensor(all_white_turns, dtype=tf.bool)
        # print('All inference indices:', all_inf_idx, inf_idx)
        if actor == 1:
            # print('Sampling model 1')
            return self._sample_actor_1(model_moves, cross_moves, is_white, inf_idx)
        else:
            # print('Sampling model 2')
            # return self._sample_actor_2(model_moves, cross_moves, is_white, inf_idx)
            return self._sample_actor_2_new(model_moves, cross_moves, is_white, inf_idx)
        

    def sample_engine(self, observation, white_turn, actor=1):
        actions_log_prob = []
        actions = []
        all_token_probs = []
        all_token_log_probs = []

        for obs in observation:
            uci_move = get_engine_move(obs, self.engine)
            if uci_move is None:
                uci_move_id = 0
            else:
                uci_move_id = int(config.token2id[uci_move])

            actions_log_prob.append(0.0)  # Engine does not provide log probability
            actions.append(uci_move_id)  # Engine always returns a single move
            all_token_probs.append([0.0 for x in range(config.vocab_size)])  # Engine always
            all_token_log_probs.append([0.0 for x in range(config.vocab_size)])  # returns a single move

        actions_log_prob = tf.convert_to_tensor(actions_log_prob, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        all_token_probs = tf.convert_to_tensor(all_token_probs, dtype=tf.float32)
        all_token_log_probs = tf.convert_to_tensor(all_token_log_probs, dtype=tf.float32)
        return actions_log_prob, actions, all_token_probs, all_token_log_probs









    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    # @tf.function(jit_compile=True, reduce_retracing=True)
    def _sample_actor_1(self, model_moves, cross_moves, is_white, inf_idx):
        observation_input = [model_moves, cross_moves, is_white]
        pred_probs, pred_values = self.c_actor.call_tf(observation_input)
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
    # @tf.function(jit_compile=True, reduce_retracing=True)
    def _sample_actor_2(self, model_moves, cross_moves, is_white, inf_idx):
        observation_input = [model_moves, cross_moves, is_white]
        pred_probs, pred_values = self.c_actor_2.call_tf(observation_input)
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
    def _sample_actor_1_new(self,
                    model_moves,
                    cross_moves,
                    is_white,
                    inf_idx):
        top_k = 5
        """Generate one action per batch element using top-k sampling."""
        # ──────────────────────────────
        # 1. Forward pass
        # ──────────────────────────────
        observation_input = [model_moves, cross_moves, is_white]
        pred_logits, pred_values = self.c_actor.call_tf(observation_input)
        pred_probs  = tf.nn.softmax(pred_logits, axis=-1)          # (B, L, V)
        all_probs   = pred_probs[:, inf_idx, :]                    # (B, V)
        log_probs   = tf.math.log(all_probs + 1e-10)               # (B, V)

        # ──────────────────────────────
        # 2. Top-k filtering
        # ──────────────────────────────
        k           = tf.minimum(top_k, tf.shape(all_probs)[-1])   # guard against k>V
        topk_vals, topk_ids = tf.math.top_k(all_probs, k=k)        # each (B, k)
        topk_log    = tf.math.log(topk_vals + 1e-10)               # (B, k)

        # ──────────────────────────────
        # 3. Sample inside the top-k set
        # ──────────────────────────────
        sampled_pos = tf.squeeze(tf.random.categorical(topk_log, 1), axis=-1)  # (B,)
        next_ids    = tf.gather(topk_ids, sampled_pos, batch_dims=1)           # (B,)

        # ──────────────────────────────
        # 4. Retrieve log-prob *under the full policy*
        #    (needed for advantage–based or off-policy losses)
        # ──────────────────────────────
        batch_idx   = tf.range(tf.shape(log_probs)[0], dtype=tf.int32)         # (B,)
        act_logprob = tf.gather_nd(log_probs, tf.stack([batch_idx, next_ids], axis=-1))  # (B,)

        # ──────────────────────────────
        # 5. Return same tuple shape as before
        # ──────────────────────────────
        return act_logprob, next_ids, all_probs, log_probs


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor_2_new(self,
                    model_moves,
                    cross_moves,
                    is_white,
                    inf_idx):
        top_k = 2
        """Generate one action per batch element using top-k sampling."""
        # ──────────────────────────────
        # 1. Forward pass
        # ──────────────────────────────
        observation_input = [model_moves, cross_moves, is_white]
        pred_logits, pred_values = self.c_actor_2.call_tf(observation_input)
        pred_probs  = tf.nn.softmax(pred_logits, axis=-1)          # (B, L, V)
        all_probs   = pred_probs[:, inf_idx, :]                    # (B, V)
        log_probs   = tf.math.log(all_probs + 1e-10)               # (B, V)

        # ──────────────────────────────
        # 2. Top-k filtering
        # ──────────────────────────────
        k           = tf.minimum(top_k, tf.shape(all_probs)[-1])   # guard against k>V
        topk_vals, topk_ids = tf.math.top_k(all_probs, k=k)        # each (B, k)
        topk_log    = tf.math.log(topk_vals + 1e-10)               # (B, k)

        # ──────────────────────────────
        # 3. Sample inside the top-k set
        # ──────────────────────────────
        sampled_pos = tf.squeeze(tf.random.categorical(topk_log, 1), axis=-1)  # (B,)
        next_ids    = tf.gather(topk_ids, sampled_pos, batch_dims=1)           # (B,)

        # ──────────────────────────────
        # 4. Retrieve log-prob *under the full policy*
        #    (needed for advantage–based or off-policy losses)
        # ──────────────────────────────
        batch_idx   = tf.range(tf.shape(log_probs)[0], dtype=tf.int32)         # (B,)
        act_logprob = tf.gather_nd(log_probs, tf.stack([batch_idx, next_ids], axis=-1))  # (B,)

        # ──────────────────────────────
        # 5. Return same tuple shape as before
        # ──────────────────────────────
        return act_logprob, next_ids, all_probs, log_probs






    def sample_critic(self, observation, white_turn=True):
        all_model_moves = []
        all_cross_moves = []
        all_white_turns = []
        max_inf_idx = 0
        for obs in observation:
            mm, cm, inf_idx = get_inputs_from_game(obs, white_turn)
            mm = self.pad_list(mm, pad_len=config.seq_length)
            cm = self.pad_list(cm, pad_len=config.seq_length)
            all_model_moves.append(mm)
            all_cross_moves.append(cm)
            all_white_turns.append(white_turn)
            if inf_idx > max_inf_idx:
                max_inf_idx = inf_idx
        inf_idx = max_inf_idx
        model_moves = tf.convert_to_tensor(all_model_moves, dtype=tf.int32)
        cross_moves = tf.convert_to_tensor(all_cross_moves, dtype=tf.int32)
        is_white = tf.convert_to_tensor(all_white_turns, dtype=tf.bool)
        return self._sample_critic(model_moves, cross_moves, is_white, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_critic(self, model_moves, cross_moves, is_white, inf_idx):
        inputs = [model_moves, cross_moves, is_white]
        pred, t_value = self.c_critic(inputs, training=False)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value





    def record(self):

        # This maps additional info keys to shorthand labels for printing
        display_keys = {
            'model_1_return': 'ret1',
            'model_2_return': 'ret2',
            'game_len': 'gamelen',
            'model_1_checkmates': 'chkm1',
            'model_2_checkmates': 'chkm2',
            'model_1_illegal_moves': 'ill1',
            'model_2_illegal_moves': 'ill2',
            'model_1_entropy': 'ent1',
            'model_2_entropy': 'ent2',
            'model_1_kl': 'kl1',
            'model_2_kl': 'kl2',
            # 'model_1_loss': 'l1',
            # 'model_2_loss': 'l2',

        }
        display_key_keys = list(display_keys.keys())
        # Record new epoch / print
        if self.debug is True:
            print(f"SelfPlay {self.run_num} - {self.curr_epoch} ", end=' ')
            for key in display_key_keys:
                disp_key = display_keys[key]
                value = self.additional_info[key]
                print(f"{disp_key}: {value[-1]:.4f}", end=' | ')
            # print('c_loss', self.c_loss[-1].numpy(), '|', 'model updates', self.actor_updates)
            print('c_loss', self.c_loss[-1].numpy())


        if len(self.additional_info['model_1_return']) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
        else:
            return



    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.additional_info['model_1_return']))]
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(12, 6.5))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        # plt.plot(epochs, self.returns)
        plt.plot(epochs, self.additional_info['model_1_return'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_return'], label='Model 2')
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.title('PPO Return Plot')
        plt.legend()

        # Critic loss plot
        plt.subplot(gs[0, 1])
        c_loss = self.c_loss
        c_epochs = epochs
        plt.plot(c_epochs, c_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Critic loss')
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[1, 0])
        # plt.plot(epochs, self.entropy)
        plt.plot(epochs, self.additional_info['model_1_entropy'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_entropy'], label='Model 2')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy Plot')
        plt.legend()

        # KL divergence plot
        plt.subplot(gs[1, 1])
        # plt.plot(epochs, self.kl)
        plt.plot(epochs, self.additional_info['model_1_kl'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_kl'], label='Model 2')
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.title('KL Divergence Plot')
        plt.legend()

        # Save and close
        plt.tight_layout()
        # save_path = os.path.join(self.run_dir, 'plots.png')
        save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')







if __name__ == '__main__':
    actor_path = os.path.join(config.weights_dir, 'chess-gpt-a3-v10.weights.h5')
    actor_2_path = os.path.join(config.weights_dir, 'chess-gpt-a3-v10.weights.h5')
    critic_path = os.path.join(config.weights_dir, 'chess-gpt-a3-v10.weights.h5')

    task = SelfPlayTaskA3(
        run_num=run_dir,
        problem=None,
        epochs=100000,
        actor_load_path=actor_path,
        actor_2_load_path=actor_2_path,
        critic_load_path=critic_path,
        debug=True,
        run_val=False,
        val_itr=run_dir_itr,
    )
    task.run()













