import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor
#set multiprocessing type to fork
import multiprocessing as mp
# mp.set_start_method('fork')
# mp.set_start_method('spawn', force=True)
from stockfish.utils import combine_alternate


import random
import math
import chess.pgn
import json
import config
import matplotlib.pyplot as plt
import os
from tasks.AbstractTask import AbstractTask
import scipy.signal
from model import get_rl_models as get_model
from collections import OrderedDict
import tensorflow_addons as tfa
import chess
import chess.engine

# from stockfish.rewards.reward_1 import calc_reward
from stockfish.rewards.reward_2 import calc_reward

from evals.AbstractEval import AbstractEval
from evals.plotting.training_comparison import plot_training_comparison
from stockfish.rewards.RewardProcess import StockfishProcess

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def id_seq_to_uci(seq):
    return [config.id2token[x] for x in seq]


# Number of self-play games in a mini-batch
global_mini_batch_size = 32

run_evals = True
use_actor_warmup = True
critic_warmup_epochs = 0

pickup_epoch = 0

run_dir = 102
run_dir_itr = 2

top_k = None

# set seeds
seed = 1
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


class SelfPlaySepGrad(AbstractTask):

    def __init__(
            self,
            run_num=0,
            problem=None,
            epochs=50,
            actor_load_path=None,
            critic_load_path=None,
            debug=False,
            run_val=False,
            val_itr=0,
    ):
        super(SelfPlaySepGrad, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.run_val = run_val
        self.val_itr = val_itr

        # Evals
        self.eval = AbstractEval()

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

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 32, "Hash": 4096 * 2})
        self.nodes = 50000
        self.lines = 1

        # self.num_engines = 4
        # self.engine_workers = []
        # self.engine_queues = []
        # for _ in range(self.num_engines):
        #     to_queue = mp.Queue()
        #     from_queue = mp.Queue()
        #     engine = StockfishProcess(to_queue, from_queue)
        #     self.engine_workers.append(engine)
        #     self.engine_queues.append((to_queue, from_queue))
        #     engine.start()


    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.000001  # 0.0001
        self.critic_learning_rate = 0.00001  # 0.0001
        self.train_actor_iterations = 100  # was 250
        self.train_critic_iterations = 40  # was 40


        # if use_actor_warmup is True:
        #     self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        #         0.0,  # initial learning rate
        #         1000,  # decay_steps
        #         alpha=1.0,
        #         warmup_target=self.actor_learning_rate,
        #         warmup_steps=1000
        #     )
        # self.critic_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        #     0.0,  # initial learning rate
        #     1000,  # decay_steps
        #     alpha=1.0,
        #     warmup_target=self.critic_learning_rate,
        #     warmup_steps=int(critic_warmup_epochs * (global_mini_batch_size * .75))
        # )



        # Optimizers
        if self.actor_optimizer is None:
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        # Color specific optimizers
        self.white_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.000002)
        self.black_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.000002)




        self.c_actor.summary()


    def run(self):
        self.build()

        self.run_evals()
        for x in range(self.epochs):
            self.curr_epoch = x
            epoch_info = self.fast_mini_batch()

            self.record(epoch_info)

            if self.curr_epoch % 50 == 0:
                t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch + pickup_epoch))
                t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch + pickup_epoch))
                self.c_actor.save_weights(t_actor_save_path)
                self.c_critic.save_weights(t_critic_save_path)


        # Save the parameters of the current actor and critic
        self.c_actor.save_weights(self.actor_pretrain_save_path)
        self.c_critic.save_weights(self.critic_pretrain_save_path)



    def sample_actions(self, board, actions, actions_log_prob):
        tokens = id_seq_to_uci(actions)
        # Filter out tokens that are special tokens
        filtered_tokens = []
        filtered_tokens_idx = []
        for idx, token in enumerate(tokens):
            if token in config.special_tokens or token in config.end_of_game_tokens or token is '':
                continue
            move = chess.Move.from_uci(token)
            if move in board.legal_moves:
                filtered_tokens.append(token)
                filtered_tokens_idx.append(idx)
        if len(filtered_tokens) == 0:  # Default to top action
            return actions[0], actions_log_prob[0]
        else:
            # randomly select based on the log probabilities of the filtered actions
            filtered_action_log_probs = [actions_log_prob[x] for x in filtered_tokens_idx]
            filtered_action_probs = [math.exp(lp) for lp in filtered_action_log_probs]
            total = sum(filtered_action_probs)  # Calculate the sum of all probabilities
            filtered_action_probs = np.array([p / total for p in filtered_action_probs])  # Divide each probability by the total sum
            action_idx = np.random.choice(filtered_tokens_idx, p=filtered_action_probs)
            return actions[action_idx], actions_log_prob[action_idx]


    def fast_mini_batch(self):
        total_eval_time = 0

        all_total_rewards = []
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

        all_logprobs_full = [[] for _ in range(self.mini_batch_size)]  # Logprobs for all actions
        all_probs_full = [[] for _ in range(self.mini_batch_size)]

        # -------------------------------------
        # Sample Actor
        # -------------------------------------
        curr_time = time.time()

        for t in range(self.max_steps_per_game):


            # Sample the actor
            action_log_prob, action, all_action_probs, batch_logprobs = self.sample_actor(observation)  # returns shape: (batch,) and (batch,)

            # Update the game state
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
                    # if idx == 0:
                    #     # print(config.id2token[m_action], round(math.exp(action_log_prob[idx]), 3), end=' ')
                    #     print(config.id2token[m_action], end=' ')

            # Determine reward for each batch element
            if len(games[0]) == self.max_steps_per_game:
                done = True
                for idx, game in enumerate(games):
                    if ended_games[idx] is False:
                        reward, game_ended, new_eval = self.calc_reward(
                            game,
                            game_evals[idx],
                            boards[idx]
                        )
                        if game_ended is False:
                            game_evals[idx] = new_eval
                        if game_ended != ended_games[idx]:
                            epoch_games.append(' '.join([config.id2token[x] for x in game]))
                        ended_games[idx] = game_ended
            else:
                done = False
                for idx, game in enumerate(games):
                    if ended_games[idx] is False:
                        reward, game_ended, new_eval = self.calc_reward(
                            game,
                            game_evals[idx],
                            boards[idx]
                        )
                        if game_ended is False:
                            game_evals[idx] = new_eval
                        if game_ended != ended_games[idx]:
                            epoch_games.append(' '.join([config.id2token[x] for x in game]))
                        ended_games[idx] = game_ended

            if all(ended_games):
                done = True


            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new


        # Count the number of checkmates
        num_checkmates = 0
        for idx, board in enumerate(boards):
            if board.is_checkmate() is True:
                # checkmating_moves = [config.id2token[x] for x in games[idx]]
                # print('Checkmating moves:', ' '.join(checkmating_moves))
                # print('Rewards:', all_rewards[idx])
                num_checkmates += 1


        for trajectory in observation:
            if len(trajectory) < config.seq_length - 1:
                trajectory += [0] * ((config.seq_length-1) - len(trajectory))
            trajectory = trajectory[:config.seq_length - 1]

        for trajectory in action_mask:
            if len(trajectory) < config.seq_length - 1:
                trajectory += [0] * ((config.seq_length-1) - len(trajectory))

        for trajectory in critic_observation_buffer:
            if len(trajectory) < config.seq_length:
                trajectory += [0] * (config.seq_length - len(trajectory))

        # print('\nGen time:', time.time() - curr_time)
        self.save_game_pgn(games[0])
        # -------------------------------------
        # Post process rewards
        # -------------------------------------
        curr_time_2 = time.time()

        # --- Batch Eval
        # num_batches = self.num_engines - 1
        # batch_size = global_mini_batch_size // num_batches
        # batches = [games[i:i + batch_size] for i in range(0, len(games), batch_size)]
        #
        # # Put the batches into the engine queues
        # # print('--> PUTTING BATCHES INTO QUEUES')
        # for idx, batch in enumerate(batches):
        #     to_queue = self.engine_queues[idx][0]
        #     to_queue.put(batch)
        #
        # # Get the results from the engine queues
        # all_rewards_post = []
        # for to_queue, from_queue in self.engine_queues:
        #     # print('Waiting on queue')
        #     results = from_queue.get(timeout=60)
        #     all_rewards_post.extend(results)

        all_even_masks = []  # Black moves (0 - white, 1 - black)
        all_odd_masks = []   # White moves (1 - white, 0 - black)
        all_even_masks_clip = []
        all_odd_masks_clip = []
        all_white_rewards = []
        all_black_rewards = []

        # --- Linear Eval
        for idx, game in enumerate(games):
            uci_game = ' '.join([config.id2token[x] for x in game])
            reward = calc_reward(self.engine, uci_game, n=self.nodes)
            # Scale rewards down
            reward = [x * 0.1 for x in reward]
            all_rewards_post[idx] = reward


            even_mask = [x % 2 for x in range(len(game))]  # [0, 1, 0, 1, ...]
            odd_mask = [1 - x for x in even_mask]          # [1, 0, 1, 0, ...]
            even_mask += [0] * (config.seq_length - len(even_mask))
            odd_mask += [0] * (config.seq_length - len(odd_mask))
            even_mask_clip = even_mask[:config.seq_length-1]
            odd_mask_clip = odd_mask[:config.seq_length-1]
            all_even_masks.append(even_mask)
            all_odd_masks.append(odd_mask)
            all_even_masks_clip.append(even_mask_clip)
            all_odd_masks_clip.append(odd_mask_clip)

            white_rewards = []
            black_rewards = []
            for idx2, r in enumerate(reward):
                if even_mask[idx2] == 1:
                    black_rewards.append(r)
                elif odd_mask[idx2] == 1:
                    white_rewards.append(r)
            all_white_rewards.append(white_rewards)
            all_black_rewards.append(black_rewards)

        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        white_value_t = []
        black_value_t = []

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards_post[idx].append(last_reward)


            white_v_t = []
            black_v_t = []
            for idx2, v in enumerate(value):
                if all_even_masks[idx][idx2] == 1:
                    black_v_t.append(v)
                elif all_odd_masks[idx][idx2] == 1:
                    white_v_t.append(v)

                # see if this is last iteration
                if idx2 == len(value) - 1:
                    if all_even_masks[idx][idx2] == 1:
                        all_black_rewards[idx].append(v)
                    elif all_odd_masks[idx][idx2] == 1:
                        all_white_rewards[idx].append(v)
            white_value_t.append(white_v_t)
            black_value_t.append(black_v_t)


        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------

        proc_time = time.time()
        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]

        all_white_advantages = [[] for _ in range(self.mini_batch_size)]
        all_black_advantages = [[] for _ in range(self.mini_batch_size)]

        all_white_returns = [[] for _ in range(self.mini_batch_size)]
        all_black_returns = [[] for _ in range(self.mini_batch_size)]

        all_combined_returns = [[] for _ in range(self.mini_batch_size)]

        # print('All rewards:', all_rewards_post[0])
        # print('White rewards:', all_white_rewards[0])
        # print('Black rewards:', all_black_rewards[0])
        # print('\n')
        # print('White returns:', white_value_t[0])
        # print('Black returns:', black_value_t[0])


        for idx in range(len(all_rewards_post)):
            w_rewards = np.array(all_white_rewards[idx])
            b_rewards = np.array(all_black_rewards[idx])

            w_values = np.array(white_value_t[idx])
            b_values = np.array(black_value_t[idx])

            w_deltas = w_rewards[:-1] + self.gamma * w_values[1:] - w_values[:-1]
            b_deltas = b_rewards[:-1] + self.gamma * b_values[1:] - b_values[:-1]

            w_adv_tensor = discounted_cumulative_sums(
                w_deltas, self.gamma * self.lam
            )
            b_adv_tensor = discounted_cumulative_sums(
                b_deltas, self.gamma * self.lam
            )
            all_white_advantages[idx] = w_adv_tensor
            all_black_advantages[idx] = b_adv_tensor



            w_ret_tensor = discounted_cumulative_sums(
                w_rewards, self.gamma
            )
            b_ret_tensor = discounted_cumulative_sums(
                b_rewards, self.gamma
            )

            combined_returns = combine_alternate(w_ret_tensor, b_ret_tensor)
            # print('White return tensor:', w_ret_tensor)
            # print('Black return tensor:', b_ret_tensor)
            # exit(0)


            while len(combined_returns) < config.seq_length:
                combined_returns.append(0)
            if len(combined_returns) > config.seq_length:
                combined_returns = combined_returns[:config.seq_length]
            all_combined_returns[idx] = np.array(combined_returns)

            w_ret_tensor = np.array(w_ret_tensor, dtype=np.float32)
            b_ret_tensor = np.array(b_ret_tensor, dtype=np.float32)
            all_white_returns[idx] = w_ret_tensor
            all_black_returns[idx] = b_ret_tensor









            rewards = np.array(all_rewards_post[idx])
            values = np.array(value_t[idx])
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            adv_tensor = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            all_advantages[idx] = adv_tensor

            ret_tensor = discounted_cumulative_sums(
                rewards, self.gamma
            )  # [:-1]
            ret_tensor = np.array(ret_tensor, dtype=np.float32)
            all_returns[idx] = ret_tensor

        advantage_mean, advantage_std = (
            np.mean(all_advantages),
            np.std(all_advantages),
        )
        all_advantages = (all_advantages - advantage_mean) / advantage_std

        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(all_actions, dtype=tf.int32)
        logprob_tensor = tf.convert_to_tensor(all_logprobs, dtype=tf.float32)
        advantage_tensor = tf.convert_to_tensor(all_advantages, dtype=tf.float32)
        critic_observation_tensor = tf.convert_to_tensor(critic_observation_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        return_tensor = tf.expand_dims(return_tensor, axis=-1)
        action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.float32)


        # print('All white advantages:', all_white_advantages[0])
        # print('All Advantages:', all_advantages[0])



        white_adv_means = [np.mean(x) for x in all_white_advantages if len(x) > 0]
        black_adv_means = [np.mean(x) for x in all_black_advantages if len(x) > 0]
        white_adv_mean = np.mean(white_adv_means)
        black_adv_mean = np.mean(black_adv_means)
        white_adv_stds = [np.std(x) for x in all_white_advantages if len(x) > 0]
        black_adv_stds = [np.std(x) for x in all_black_advantages if len(x) > 0]
        white_adv_std = np.mean(white_adv_stds)
        black_adv_std = np.mean(black_adv_stds)


        norm_white_advantages = []
        for white_adv in all_white_advantages:
            norm_white_adv = (white_adv - white_adv_mean) / white_adv_std
            norm_white_advantages.append(norm_white_adv)
        norm_black_advantages = []
        for black_adv in all_black_advantages:
            norm_black_adv = (black_adv - black_adv_mean) / black_adv_std
            norm_black_advantages.append(norm_black_adv)
        all_white_advantages = norm_white_advantages
        all_black_advantages = norm_black_advantages



        # white_advantage_mean, white_advantage_std = (
        #     np.mean(all_white_advantages),
        #     np.std(all_white_advantages),
        # )
        # all_white_advantages = (all_white_advantages - white_advantage_mean) / white_advantage_std
        #
        # black_advantage_mean, black_advantage_std = (
        #     np.mean(all_black_advantages),
        #     np.std(all_black_advantages),
        # )
        # all_black_advantages = (all_black_advantages - black_advantage_mean) / black_advantage_std


        all_white_advantages_padded = []
        all_black_advantages_padded = []
        for idx in range(len(all_white_advantages)):
            w_adv = all_white_advantages[idx]
            b_adv = all_black_advantages[idx]

            white_adv_pad = []
            for w in w_adv:
                white_adv_pad.append(w)
                white_adv_pad.append(0)
            white_adv_pad = white_adv_pad[:-1]
            while len(white_adv_pad) < config.seq_length-1:
                white_adv_pad.append(0)
            if len(white_adv_pad) > config.seq_length-1:
                white_adv_pad = white_adv_pad[:config.seq_length-1]

            black_adv_pad = []
            for b in b_adv:
                black_adv_pad.append(0)
                black_adv_pad.append(b)
            while len(black_adv_pad) < config.seq_length-1:
                black_adv_pad.append(0)
            if len(black_adv_pad) > config.seq_length-1:
                black_adv_pad = black_adv_pad[:config.seq_length-1]

            all_white_advantages_padded.append(white_adv_pad)
            all_black_advantages_padded.append(black_adv_pad)


        all_combined_advantages = []
        for idx in range(len(all_white_advantages)):
            w_adv = all_white_advantages[idx]
            b_adv = all_black_advantages[idx]
            combined_adv = combine_alternate(w_adv, b_adv)
            while len(combined_adv) < config.seq_length-1:
                combined_adv.append(0)
            if len(combined_adv) > config.seq_length-1:
                combined_adv = combined_adv[:config.seq_length-1]
            all_combined_advantages.append(combined_adv)

        # print('\n')
        # print('Combined advantages:', all_combined_advantages[0])
        # print('White advantages:', all_white_advantages[0])
        # print('Black advantages:', all_black_advantages[0])



        combined_advantage_tensor = tf.convert_to_tensor(all_combined_advantages, dtype=tf.float32)



        white_advantage_tensor = tf.convert_to_tensor(all_white_advantages_padded, dtype=tf.float32)
        black_advantage_tensor = tf.convert_to_tensor(all_black_advantages_padded, dtype=tf.float32)

        # white_return_tensor = tf.convert_to_tensor(all_white_returns, dtype=tf.float32)
        # white_return_tensor = tf.expand_dims(white_return_tensor, axis=-1)
        #
        # black_return_tensor = tf.convert_to_tensor(all_black_returns, dtype=tf.float32)
        # black_return_tensor = tf.expand_dims(black_return_tensor, axis=-1)

        white_action_mask_tensor = tf.convert_to_tensor(all_even_masks_clip, dtype=tf.float32)  # 0 - white, 1 - black
        black_action_mask_tensor = tf.convert_to_tensor(all_odd_masks_clip, dtype=tf.float32)   # 1 - white, 0 - black


        combined_returns_tensor = tf.convert_to_tensor(all_combined_returns, dtype=tf.float32)
        combined_returns_tensor = tf.expand_dims(combined_returns_tensor, axis=-1)




        # white_action_mask_tensor,  # 0 - white, 1 - black
        # black_action_mask_tensor,  # 1 - white, 0 - black
        # white_advantage_buffer,
        # black_advantage_buffer,

        # print('----> Log probability buffer:', logprob_tensor.shape)    # (batch, 127)
        # print('----------> Advantage buffer:', advantage_tensor.shape)  # (batch, 127)
        #
        # print('----> White advantage tensor:', white_advantage_tensor.shape)    # (batch, 128)
        # print('-----> Mask out black values:', black_action_mask_tensor.shape)  # (batch, 127)
        #
        # print('----> Black advantage tensor:', black_advantage_tensor.shape)
        # print('-----> Mask out white values:', white_action_mask_tensor.shape)


        # -------------------------------------
        # Train Actor
        # -------------------------------------
        observation_tensor = observation_tensor[:, :config.seq_length-1]
        # print('SHAPES:', observation_tensor.shape, action_tensor.shape, logprob_tensor.shape, advantage_tensor.shape, action_mask_tensor.shape)

        curr_time = time.time()
        policy_update_itr = 0
        kl, entr, white_policy_loss, black_policy_loss = 0, 0, 0, 0
        if self.curr_epoch >= critic_warmup_epochs:
            for i in range(self.train_actor_iterations):
                policy_update_itr += 1
                kl, entr, white_policy_loss, black_policy_loss = self.train_actor(
                    observation_tensor,
                    action_tensor,
                    logprob_tensor,
                    advantage_tensor,
                    action_mask_tensor,
                    white_action_mask_tensor,
                    black_action_mask_tensor,
                    white_advantage_tensor,
                    black_advantage_tensor,
                    combined_advantage_tensor
                )
                if abs(kl) > 1.5 * self.target_kl:
                    # Early Stopping
                    break
            kl = kl.numpy()
            entr = entr.numpy()
            white_policy_loss = white_policy_loss.numpy()
            black_policy_loss = black_policy_loss.numpy()

        self.actor_updates += policy_update_itr
        # print('Actor time:', time.time() - curr_time, 'seconds')

        # -------------------------------------
        # Train Critic
        # -------------------------------------

        curr_time = time.time()
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                combined_returns_tensor,  # return_tensor,
            )

        # print('Critic time:', time.time() - curr_time, 'seconds')


        avg_game_len = np.mean([len(x) for x in games])

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_rewards_post),
            'c_loss': value_loss.numpy(),
            'w_loss': white_policy_loss,
            'b_loss': black_policy_loss,
            'p_iter': policy_update_itr,
            'entropy': entr,
            'kl': kl,
            'game_len': avg_game_len,
            'checkmates': num_checkmates,
        }



        # print('GAME:', epoch_games[0])

        return epoch_info


    def save_game_pgn(self, game_moves):
        uci_moves = [config.id2token[x] for x in game_moves]
        uci_moves_clipped = [x for x in uci_moves if x not in config.end_of_game_tokens and x not in config.special_tokens and x is not '']

        game = chess.pgn.Game()
        node = game
        board = chess.Board()
        for idx, move in enumerate(uci_moves_clipped):
            # print('GAME MOVE:', move)
            move = chess.Move.from_uci(move)
            if move in board.legal_moves:
                board.push(move)
                node = node.add_variation(move)
            else:
                break

        game.headers["Event"] = "Example Game"
        game.headers["Site"] = "Internet"
        game.headers["Date"] = "2024.04.13"
        game.headers["Round"] = "1"
        game.headers["White"] = "Player1"
        game.headers["Black"] = "Player2"
        game.headers["Result"] = "*"  # Or use board.result() if the game is over

        pgn_path = os.path.join(self.run_dir, 'game.pgn')
        with open(pgn_path, "w") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)

        # Append UCI move sequence to the end of the PGN file
        with open(pgn_path, 'a') as pgn_file:
            pgn_file.write('\n\n')
            pgn_file.write(' '.join(uci_moves))
            pgn_file.write('\n\n')

    # This is only to be called if the game has not finished
    # All position evals are from the perspective of the white player
    def calc_reward(self, game, prev_eval, board):
        uci_moves = [config.id2token[x] for x in game]
        last_move = uci_moves[-1]

        # 0. Determine the turn color
        white_turn = (board.turn == chess.WHITE)

        # check if it is checkmate
        if board.is_checkmate():
            # this means the current player is checkmated. win for other side
            if white_turn is True:
                winning_token = '[black]'
            else:
                winning_token = '[white]'
            # print('\nCheckmate prediction token:', last_move, 'Winning token:', winning_token)
            # print('Move sequence string:', ' '.join(uci_moves))
            if last_move == winning_token:
                return 1.0, True, prev_eval
            else:
                return -1.0, True, prev_eval
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            if last_move == '[draw]':
                return 1.0, True, prev_eval
            else:
                return -1.0, True, prev_eval


        # 1. Check if a legal move has been made
        if last_move not in config.non_move_tokens:
            move = chess.Move.from_uci(last_move)
            if move not in board.legal_moves:
                return -1, True, prev_eval  # Illegal UCI
        else:
            return -1, True, prev_eval  # Illegal special token move


        # -------------------------------------
        # Push move
        # -------------------------------------
        board.push(move)

        # 2. Check if checkmate
        if board.is_checkmate():
            # print('\nCheckmate played\n')
            return 1.0, False, prev_eval

        # 3. Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return -0.1, False, prev_eval


        return 0.0, False, prev_eval


    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.int32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        if top_k is not None:
            return self._sample_actor_top_k(observation_input, inf_idx)
        else:
            return self._sample_actor(observation_input, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs, pred_values = self.c_actor(observation_input)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)

        # Batch sampling
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
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor_top_k(self, observation_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs, pred_vals = self.c_actor(observation_input)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)

        # Get top k moves
        k = 5
        if top_k is not None:
            k = top_k
        pred_probs_inf = pred_probs[:, inf_idx, :]  # shape (batch, vocab_size)
        top_probs, top_indices = tf.math.top_k(pred_probs_inf, k=k) # shape (batch, k), shape (batch, k)
        top_probs_log = tf.math.log(top_probs + 1e-10)
        return top_probs_log, top_indices, pred_probs

    def sample_critic(self, observation):
        for trajectory in observation:
            if len(trajectory) < config.seq_length:
                trajectory += [0] * (config.seq_length - len(trajectory))
        inf_idx = len(observation[0]) - 1
        observation_input = tf.convert_to_tensor(observation, dtype=tf.float32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_critic(observation_input)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def _sample_critic(self, observation_input):
        pred, t_value = self.c_critic(observation_input)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value

    @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_actor(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer,
            action_mask_tensor,
            white_action_mask_tensor,  # 0 - white, 1 - black
            black_action_mask_tensor,  # 1 - white, 0 - black
            white_advantage_buffer,
            black_advantage_buffer,
            combined_advantage_buffer,
    ):

        # White loss calc
        # print('--> White actor update')
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.c_actor(observation_buffer)  # shape: (batch, seq_len, 2)
            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)

            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            white_logprobs = logprobability * black_action_mask_tensor
            white_logprobs_buffer = logprobability_buffer * black_action_mask_tensor
            white_ratio = tf.exp(
                white_logprobs - white_logprobs_buffer
            )
            white_min_advantage = tf.where(
                white_advantage_buffer > 0,
                (1 + self.clip_ratio) * white_advantage_buffer,
                (1 - self.clip_ratio) * white_advantage_buffer,
            )
            white_policy_loss = -tf.reduce_mean(
                tf.minimum(white_ratio * white_advantage_buffer, white_min_advantage)
            )

        white_grads = tape.gradient(white_policy_loss, self.c_actor.trainable_variables)
        self.white_optimizer.apply_gradients(zip(white_grads, self.c_actor.trainable_variables))

        # Black loss calc
        # print('--> Black actor update')
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.c_actor(observation_buffer)  # shape: (batch, seq_len, 2)
            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)

            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            black_logprobs = logprobability * white_action_mask_tensor
            black_logprobs_buffer = logprobability_buffer * white_action_mask_tensor
            black_ratio = tf.exp(
                black_logprobs - black_logprobs_buffer
            )
            black_min_advantage = tf.where(
                black_advantage_buffer > 0,
                (1 + self.clip_ratio) * black_advantage_buffer,
                (1 - self.clip_ratio) * black_advantage_buffer,
            )
            black_policy_loss = -tf.reduce_mean(
                tf.minimum(black_ratio * black_advantage_buffer, black_min_advantage)
            )

            # Entropy Term
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)

        black_grads = tape.gradient(black_policy_loss, self.c_actor.trainable_variables)
        self.black_optimizer.apply_gradients(zip(black_grads, self.c_actor.trainable_variables))


        # Combined loss calc
        # with tf.GradientTape() as tape:
        #     pred_probs, pred_vals = self.c_actor(observation_buffer)  # shape: (batch, seq_len, 2)
        #     pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
        #     pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)
        #
        #     logprobability = tf.reduce_sum(
        #         tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        #     )  # shape (batch, seq_len)
        #     logprobability *= action_mask_tensor
        #
        #     # Total loss
        #     loss = 0
        #
        #     # PPO Surrogate Loss
        #     ratio = tf.exp(
        #         logprobability - logprobability_buffer
        #     )
        #     min_advantage = tf.where(
        #         combined_advantage_buffer > 0,
        #         (1 + self.clip_ratio) * combined_advantage_buffer,
        #         (1 - self.clip_ratio) * combined_advantage_buffer,
        #     )
        #     policy_loss = -tf.reduce_mean(
        #         tf.minimum(ratio * combined_advantage_buffer, min_advantage)
        #     )
        #
        #     loss += policy_loss
        #
        #     # Entropy Term
        #     entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
        #     entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
        #     loss = loss - (self.entropy_coef * entr)
        #
        # policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        # self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        #  KL Divergence
        pred_probs, pred_vals = self.c_actor(observation_buffer)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        logprobability *= action_mask_tensor
        kl = tf.reduce_mean(
            logprobability_buffer - logprobability
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, white_policy_loss, black_policy_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            observation_buffer,
            return_buffer,
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            preds, pred_values = self.c_critic(observation_buffer)  # (batch, seq_len, 2)

            # Value Loss (mse)
            value_loss = tf.reduce_mean((return_buffer - pred_values) ** 2)

        critic_grads = tape.gradient(value_loss, self.c_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.c_critic.trainable_variables))

        return value_loss



    def record(self, epoch_info):
        if epoch_info is None:
            return

        # Record new epoch / print
        if self.debug is True:
            print(f"SelfPlay {self.run_num} - {self.curr_epoch} ", end=' ')
            for key, value in epoch_info.items():
                if isinstance(value, list):
                    print(f"{key}: {value}", end=' | ')
                else:
                    print("%s: %.5f" % (key, value), end=' | ')
            print('actor_updates:', self.actor_updates)


        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['w_loss'] + epoch_info['b_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.run_evals()
            self.plot_ppo()
        else:
            return

    def run_evals(self):
        if run_evals is False:
            return
        evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
        eval_history = self.eval.run_eval(self.c_actor, themes=evals)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'evals_val_' + str(self.val_itr) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)


    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(16, 8))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.returns)
        plt.xlabel('Epoch')
        plt.ylabel('Mini-batch Return')
        plt.title('PPO Return Plot')

        # Critic loss plot
        plt.subplot(gs[0, 1])
        if len(self.c_loss) < 100:
            c_loss = self.c_loss
            c_epochs = epochs
        else:
            c_loss = self.c_loss[50:]
            c_epochs = epochs[50:]
        plt.plot(c_epochs, c_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Critic loss')
        plt.title('Critic Loss Plot')

        # Policy entropy plot
        plt.subplot(gs[1, 0])
        plt.plot(epochs, self.entropy)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy Plot')

        # KL divergence plot
        plt.subplot(gs[1, 1])
        plt.plot(epochs, self.kl)
        plt.xlabel('Epoch')
        plt.ylabel('KL')
        plt.title('KL Divergence Plot')

        # Save and close
        plt.tight_layout()
        # save_path = os.path.join(self.run_dir, 'plots.png')
        save_path = os.path.join(self.run_dir, 'plots_val_' + str(self.val_itr) + '.png')
        plt.savefig(save_path)
        plt.close('all')


if __name__ == '__main__':
    actor_path = config.model_path
    critic_path = config.model_path

    # actor_path = os.path.join(config.results_dir, 'run_100', 'pretrained', 'actor_weights_2000')
    # critic_path = os.path.join(config.results_dir, 'run_100', 'pretrained', 'critic_weights_2000')

    # actor_path = None
    # critic_path = None

    task = SelfPlaySepGrad(
        run_num=run_dir,
        problem=None,
        epochs=10000,
        actor_load_path=actor_path,
        critic_load_path=critic_path,
        debug=True,
        run_val=False,
        val_itr=run_dir_itr,
    )
    # task.get_even_mask([x for x in range(10)])
    task.run()