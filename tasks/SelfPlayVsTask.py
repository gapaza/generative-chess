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
from evals.AbstractEval import AbstractEval
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

sample_temp = 0.9

top_k = None

# set seeds
seed = 1
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


class SelfPlayVsTask(AbstractTask):

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
        super(SelfPlayVsTask, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.run_val = run_val
        self.val_itr = val_itr
        self.actor_2_load_path = actor_2_load_path


        # Evals
        self.eval = AbstractEval()
        self.eval2 = AbstractEval()

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
            2   # self.c_actor_2
        ]
        random.shuffle(player_models)
        white_model = player_models[0]
        black_model = player_models[1]
        # print('-----------------------------------')
        # print('White player:', 'Actor ' + str(white_model), '| Black player:', 'Actor ' + str(black_model))

        self.additional_info['model_1_checkmates'].append(0)
        self.additional_info['model_2_checkmates'].append(0)
        self.additional_info['model_1_illegal_moves'].append(0)
        self.additional_info['model_2_illegal_moves'].append(0)
        self.additional_info['model_1_win_predictions'].append(0)
        self.additional_info['model_2_win_predictions'].append(0)

        for t in range(self.max_steps_per_game):

            # 1. Determine which model to use
            if t % 2 == 0:  # even number
                actor = white_model
            else:  # odd number
                actor = black_model

            # 2. Sample actions
            action_log_prob, action, all_action_probs, batch_logprobs = self.sample_actor(observation, actor=actor)  # returns shape: (batch,) and (batch,)

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

            # 4. Determine if game has ended
            if len(games[0]) == self.max_steps_per_game:
                done = True
                for idx, game in enumerate(games):
                    if ended_games[idx] is False:
                        reward, game_ended, new_eval = self.calc_reward(
                            game,
                            game_evals[idx],
                            boards[idx],
                            actor
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
                            boards[idx],
                            actor
                        )
                        if game_ended is False:
                            game_evals[idx] = new_eval
                        if game_ended != ended_games[idx]:
                            epoch_games.append(' '.join([config.id2token[x] for x in game]))
                        ended_games[idx] = game_ended
            if all(ended_games):
                done = True

            # 5. Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new

        # Count the number of checkmates + finish trajectories
        num_checkmates = 0
        for idx, board in enumerate(boards):
            if board.is_checkmate() is True:
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


        # print('game -', epoch_games[0])
        self.save_game_pgn(games[0])
        # -------------------------------------
        # Post process rewards
        # -------------------------------------

        all_even_masks = []  # Black moves (0 - white, 1 - black)
        all_odd_masks = []  # White moves (1 - white, 0 - black)
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
            odd_mask = [1 - x for x in even_mask]  # [1, 0, 1, 0, ...]
            even_mask += [0] * (config.seq_length - len(even_mask))
            odd_mask += [0] * (config.seq_length - len(odd_mask))
            even_mask_clip = even_mask[:config.seq_length - 1]
            odd_mask_clip = odd_mask[:config.seq_length - 1]
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

        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        all_white_advantages = [[] for _ in range(self.mini_batch_size)]
        all_black_advantages = [[] for _ in range(self.mini_batch_size)]
        all_white_returns = [[] for _ in range(self.mini_batch_size)]
        all_black_returns = [[] for _ in range(self.mini_batch_size)]
        all_combined_returns = [[] for _ in range(self.mini_batch_size)]

        cum_white_advantages = []
        cum_black_advantages = []

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

            cum_white_advantages += deepcopy(w_adv_tensor.tolist())
            cum_black_advantages += deepcopy(b_adv_tensor.tolist())




            w_ret_tensor = discounted_cumulative_sums(
                w_rewards, self.gamma
            )
            b_ret_tensor = discounted_cumulative_sums(
                b_rewards, self.gamma
            )
            combined_returns = combine_alternate(w_ret_tensor, b_ret_tensor)
            while len(combined_returns) < config.seq_length:
                combined_returns.append(0)
            if len(combined_returns) > config.seq_length:
                combined_returns = combined_returns[:config.seq_length]
            all_combined_returns[idx] = np.array(combined_returns)
            w_ret_tensor = np.array(w_ret_tensor, dtype=np.float32)
            b_ret_tensor = np.array(b_ret_tensor, dtype=np.float32)
            all_white_returns[idx] = w_ret_tensor
            all_black_returns[idx] = b_ret_tensor

            # Color agnostic advantage
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

        # Color specific tensors
        # print('Advantage Means')
        # white_adv_means = [np.mean(x) for x in all_white_advantages if len(x) > 0]
        # black_adv_means = [np.mean(x) for x in all_black_advantages if len(x) > 0]
        # white_adv_mean = np.mean(white_adv_means)
        # black_adv_mean = np.mean(black_adv_means)
        # white_adv_stds = [np.std(x) for x in all_white_advantages if len(x) > 0]
        # black_adv_stds = [np.std(x) for x in all_black_advantages if len(x) > 0]
        # white_adv_std = np.mean(white_adv_stds)
        # black_adv_std = np.mean(black_adv_stds)

        white_adv_mean = np.mean(cum_white_advantages)
        black_adv_mean = np.mean(cum_black_advantages)
        white_adv_std = np.std(cum_white_advantages)
        black_adv_std = np.std(cum_black_advantages)

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
            while len(white_adv_pad) < config.seq_length - 1:
                white_adv_pad.append(0)
            if len(white_adv_pad) > config.seq_length - 1:
                white_adv_pad = white_adv_pad[:config.seq_length - 1]

            black_adv_pad = []
            for b in b_adv:
                black_adv_pad.append(0)
                black_adv_pad.append(b)
            while len(black_adv_pad) < config.seq_length - 1:
                black_adv_pad.append(0)
            if len(black_adv_pad) > config.seq_length - 1:
                black_adv_pad = black_adv_pad[:config.seq_length - 1]

            all_white_advantages_padded.append(white_adv_pad)
            all_black_advantages_padded.append(black_adv_pad)

        all_combined_advantages = []
        for idx in range(len(all_white_advantages)):
            w_adv = all_white_advantages[idx]
            b_adv = all_black_advantages[idx]
            combined_adv = combine_alternate(w_adv, b_adv)
            while len(combined_adv) < config.seq_length - 1:
                combined_adv.append(0)
            if len(combined_adv) > config.seq_length - 1:
                combined_adv = combined_adv[:config.seq_length - 1]
            all_combined_advantages.append(combined_adv)

        white_advantage_tensor = tf.convert_to_tensor(all_white_advantages_padded, dtype=tf.float32)
        black_advantage_tensor = tf.convert_to_tensor(all_black_advantages_padded, dtype=tf.float32)
        white_action_mask_tensor = tf.convert_to_tensor(all_odd_masks_clip, dtype=tf.float32)   # 1 - white, 0 - black
        black_action_mask_tensor = tf.convert_to_tensor(all_even_masks_clip, dtype=tf.float32)  # 0 - white, 1 - black

        combined_returns_tensor = tf.convert_to_tensor(all_combined_returns, dtype=tf.float32)
        combined_returns_tensor = tf.expand_dims(combined_returns_tensor, axis=-1)

        observation_tensor = observation_tensor[:, :config.seq_length - 1]

        a1_idx = player_models.index(1)
        if a1_idx == 0:  # actor 1 is playing white
            a1_advantage_tensor = white_advantage_tensor
            a1_action_mask_tensor = white_action_mask_tensor
            a2_advantage_tensor = black_advantage_tensor
            a2_action_mask_tensor = black_action_mask_tensor
        else:
            a1_advantage_tensor = black_advantage_tensor
            a1_action_mask_tensor = black_action_mask_tensor
            a2_advantage_tensor = white_advantage_tensor
            a2_action_mask_tensor = white_action_mask_tensor

        # -------------------------------------
        # Printing Models
        # -------------------------------------
        # a1_color = 'white' if player_models[0] == 1 else 'black'
        # a2_color = 'white' if player_models[1] == 1 else 'black'
        # print('\n---------------- Actor 1:', a1_color)
        # first_obs = observation_tensor.numpy().tolist()[0]
        # first_obs_tokens = [config.id2token[x] for x in first_obs]
        # print('------ First Observation:', ' '.join(first_obs_tokens))
        # print('----- Observation Tensor:', observation_tensor.numpy().tolist()[0])
        # print('---------- Action Tensor:', action_tensor.numpy().tolist()[0])
        # print('--------- Logprob Tensor:', logprob_tensor.numpy().tolist()[0])
        # print('----- Action Mask Tensor:', action_mask_tensor.numpy().tolist()[0])
        # print('-- Action Mask Tensor A1:', a1_action_mask_tensor.numpy().tolist()[0])
        # print('---- Advantage Tensor A1:', a1_advantage_tensor.numpy().tolist()[0])


        # -------------------------------------
        # Train Actor 1
        # -------------------------------------
        # print('--> Actor 1 update')

        policy_update_itr = 0
        kl_1, entr_1, loss_1, policy_loss_1 = 0, 0, 0, 0
        if self.curr_epoch >= critic_warmup_epochs:
            for i in range(self.train_actor_iterations):
                policy_update_itr += 1
                kl_1, entr_1, loss_1, policy_loss_1 = self.train_actor_1(
                    observation_tensor,
                    action_tensor,
                    logprob_tensor,
                    action_mask_tensor,
                    a1_action_mask_tensor,
                    a1_advantage_tensor,
                )
                if abs(kl_1) > 1.5 * self.target_kl:
                    # Early Stopping
                    break
            kl_1 = kl_1.numpy()
            entr_1 = entr_1.numpy()
            loss_1 = loss_1.numpy()
            policy_loss_1 = policy_loss_1.numpy()
        self.actor_updates += policy_update_itr

        # -------------------------------------
        # Train Actor 2
        # -------------------------------------
        # print('--> Actor 2 update')

        kl_2, entr_2, loss_2, policy_loss_2 = 0, 0, 0, 0
        if self.curr_epoch >= critic_warmup_epochs:
            for i in range(self.train_actor_iterations):
                policy_update_itr += 1
                kl_2, entr_2, loss_2, policy_loss_2 = self.train_actor_2(
                    observation_tensor,
                    action_tensor,
                    logprob_tensor,
                    action_mask_tensor,
                    a2_action_mask_tensor,
                    a2_advantage_tensor,
                )
                if abs(kl_2) > 1.5 * self.target_kl:
                    # Early Stopping
                    break
            kl_2 = kl_2.numpy()
            entr_2 = entr_2.numpy()
            loss_2 = loss_2.numpy()
            policy_loss_2 = policy_loss_2.numpy()
        self.actor_updates += policy_update_itr

        # -------------------------------------
        # Train Critic
        # -------------------------------------
        # print('--> Critic update')

        curr_time = time.time()
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                combined_returns_tensor,  # return_tensor,
            )

        avg_game_len = np.mean([len(x) for x in games])

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_rewards_post),
            'c_loss': value_loss.numpy(),
            'a1_loss': loss_1,
            'a2_loss': loss_2,
            'p_iter': policy_update_itr,
            'a1_entropy': entr_1,
            'a2_entropy': entr_2,
            'a1_kl': kl_1,
            'a2_kl': kl_2,
            'game_len': avg_game_len,
            'checkmates': num_checkmates,
        }

        # Update additional info
        self.additional_info['model_1_loss'].append(loss_1)
        self.additional_info['model_2_loss'].append(loss_2)
        self.additional_info['model_1_entropy'].append(entr_1)
        self.additional_info['model_2_entropy'].append(entr_2)
        self.additional_info['model_1_kl'].append(kl_1)
        self.additional_info['model_2_kl'].append(kl_2)

        avg_game_len = np.mean([len(x) for x in games])
        self.additional_info['game_len'].append(avg_game_len)

        avg_white_reward = np.mean([np.sum(x) for x in all_white_rewards])
        avg_black_reward = np.mean([np.sum(x) for x in all_black_rewards])
        if player_models[0] == 1:
            self.additional_info['model_1_return'].append(avg_white_reward)
            self.additional_info['model_2_return'].append(avg_black_reward)
        else:
            self.additional_info['model_1_return'].append(avg_black_reward)
            self.additional_info['model_2_return'].append(avg_white_reward)


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



    def calc_reward(self, game, prev_eval, board, actor):
        uci_moves = [config.id2token[x] for x in game]
        last_move = uci_moves[-1]

        if actor == 1:
            prefix = 'model_1'
        else:
            prefix = 'model_2'


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
                self.additional_info[prefix + '_win_predictions'][-1] += 1
                return 1.0, True, prev_eval
            else:
                return -1.0, True, prev_eval
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            if last_move == '[draw]':
                self.additional_info[prefix + '_win_predictions'][-1] += 1
                return 1.0, True, prev_eval
            else:
                return -1.0, True, prev_eval


        # 1. Check if a legal move has been made
        if last_move not in config.non_move_tokens:
            move = chess.Move.from_uci(last_move)
            if move not in board.legal_moves:
                self.additional_info[prefix + '_illegal_moves'][-1] += 1
                return -1, True, prev_eval  # Illegal UCI
        else:
            self.additional_info[prefix + '_illegal_moves'][-1] += 1
            return -1, True, prev_eval  # Illegal special token move


        # -------------------------------------
        # Push move
        # -------------------------------------
        board.push(move)

        # 2. Check if checkmate
        if board.is_checkmate():
            # print('\nCheckmate played\n')
            self.additional_info[prefix + '_checkmates'][-1] += 1
            return 1.0, False, prev_eval

        # 3. Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return -0.1, False, prev_eval

        return 0.0, False, prev_eval



    def sample_actor(self, observation, actor=1):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.int32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        if actor == 1:
            return self._sample_actor_1(observation_input, inf_idx)
        else:
            return self._sample_actor_2(observation_input, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor_1(self, observation_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs, pred_values = self.c_actor(observation_input)
        # pred_probs_t = pred_probs / sample_temp
        
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)


        # pred_probs_t = tf.nn.softmax(pred_probs_t, axis=-1)  # shape (batch, seq_len, vocab_size)
        # all_token_probs_t = pred_probs_t[:, inf_idx, :]  # shape (batch, 2)
        # all_token_log_probs_t = tf.math.log(all_token_probs_t + 1e-10)


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
    def _sample_actor_2(self, observation_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs, pred_values = self.c_actor_2(observation_input)
        # pred_probs_t = pred_probs / sample_temp

        
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)

        # pred_probs_t = tf.nn.softmax(pred_probs_t, axis=-1)  # shape (batch, seq_len, vocab_size)
        # all_token_probs_t = pred_probs_t[:, inf_idx, :]  # shape (batch, 2)
        # all_token_log_probs_t = tf.math.log(all_token_probs_t + 1e-10)


        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)

        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))
        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs, all_token_log_probs

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
    ])
    def train_actor_1(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            action_mask_tensor,
            color_action_mask_tensor,
            color_advantage_buffer,
    ):

        # White loss calc
        # print('--> White actor update')
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.c_actor(observation_buffer)  # shape: (batch, seq_len, 2)

            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)

            loss = 0
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            white_logprobs = logprobability * color_action_mask_tensor
            white_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            white_ratio = tf.exp(
                white_logprobs - white_logprobs_buffer
            )
            white_min_advantage = tf.where(
                color_advantage_buffer > 0,
                (1 + self.clip_ratio) * color_advantage_buffer,
                (1 - self.clip_ratio) * color_advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(white_ratio * color_advantage_buffer, white_min_advantage)
            )
            loss += policy_loss

            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)


        gradients = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.c_actor.trainable_variables))

        #  KL Divergence
        pred_probs, pred_vals = self.c_actor_2(observation_buffer)

        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        color_logprobs = logprobability * color_action_mask_tensor
        color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
        kl = tf.reduce_mean(
            color_logprobs_buffer - color_logprobs
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, loss, policy_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_actor_2(
            self,
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            action_mask_tensor,
            color_action_mask_tensor,
            color_advantage_buffer,
    ):

        # White loss calc
        # print('--> White actor update')
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.c_actor_2(observation_buffer)  # shape: (batch, seq_len, 2)

            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)

            loss = 0
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            white_logprobs = logprobability * color_action_mask_tensor
            white_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            white_ratio = tf.exp(
                white_logprobs - white_logprobs_buffer
            )
            white_min_advantage = tf.where(
                color_advantage_buffer > 0,
                (1 + self.clip_ratio) * color_advantage_buffer,
                (1 - self.clip_ratio) * color_advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(white_ratio * color_advantage_buffer, white_min_advantage)
            )
            loss += policy_loss

            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        gradients = tape.gradient(loss, self.c_actor_2.trainable_variables)
        self.actor_optimizer_2.apply_gradients(zip(gradients, self.c_actor_2.trainable_variables))

        #  KL Divergence
        pred_probs, pred_vals = self.c_actor_2(observation_buffer)

        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs)
        logprobability = tf.reduce_sum(
            tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
        )  # shape (batch, seq_len)
        color_logprobs = logprobability * color_action_mask_tensor
        color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
        kl = tf.reduce_mean(
            color_logprobs_buffer - color_logprobs
        )
        kl = tf.reduce_sum(kl)

        return kl, entr, loss, policy_loss

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
            'model_1_loss': 'l1',
            'model_2_loss': 'l2',

        }
        display_key_keys = list(display_keys.keys())
        # Record new epoch / print
        if self.debug is True:
            print(f"SelfPlay {self.run_num} - {self.curr_epoch} ", end=' ')
            for key in display_key_keys:
                disp_key = display_keys[key]
                value = self.additional_info[key]
                print(f"{disp_key}: {value[-1]:.4f}", end=' | ')
            print(self.actor_updates)

        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['a1_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['a1_entropy'])
        self.kl.append(epoch_info['a1_kl'])

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
            self.plot_models()
            self.run_evals()
        else:
            return

    def run_evals(self):
        if run_evals is False:
            return
        evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']

        # Evals for model 1
        eval_history = self.eval.run_eval(self.c_actor, themes=evals)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'a1_evals_val_' + str(self.val_itr) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)

        # Evals for model 2
        eval_history = self.eval2.run_eval(self.c_actor_2, themes=evals)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'a2_evals_val_' + str(self.val_itr) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)

    def plot_ppo(self):

        # --- Plotting ---
        epochs = [x for x in range(len(self.returns))]
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

    def plot_models(self):

        game_epochs = [x * global_mini_batch_size for x in range(len(self.additional_info['model_1_win_predictions']))]

        # --- Plotting ---
        epochs = [x for x in range(len(self.additional_info['model_1_win_predictions']))]
        gs = gridspec.GridSpec(1, 3)
        fig = plt.figure(figsize=(12, 4))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Results', fontsize=16)


        # Model illegal moves
        plt.subplot(gs[0, 0])
        plt.plot(epochs, self.additional_info['model_1_illegal_moves'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_illegal_moves'], label='Model 2')
        plt.xlabel('Epochs')
        plt.ylabel('Illegal Moves')
        plt.legend()
        plt.title('Illegal Moves')

        # Model checkmates
        plt.subplot(gs[0, 1])
        plt.plot(epochs, self.additional_info['model_1_checkmates'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_checkmates'], label='Model 2')
        plt.xlabel('Epochs')
        plt.ylabel('Checkmates')
        plt.legend()
        plt.title('Checkmates')

        # Model win predictions
        plt.subplot(gs[0, 2])
        plt.plot(epochs, self.additional_info['model_1_win_predictions'], label='Model 1')
        plt.plot(epochs, self.additional_info['model_2_win_predictions'], label='Model 2')
        plt.xlabel('Epochs')
        plt.ylabel('Win Predictions')
        plt.legend()
        plt.title('Win Predictions')

        # Save and close
        plt.tight_layout()
        save_path = os.path.join(self.run_dir, 'model_plots_'+str(self.val_itr)+'.png')
        plt.savefig(save_path)











if __name__ == '__main__':
    actor_path = config.model_path
    actor_2_path = config.model_path
    critic_path = config.model_path

    actor_path = os.path.join(config.results_dir, 'run_1003', 'pretrained', 'actor_weights_50')
    actor_2_path = os.path.join(config.results_dir, 'run_1003', 'pretrained', 'actor_weights_2_50')
    critic_path = os.path.join(config.results_dir, 'run_1003', 'pretrained', 'critic_weights_50')

    task = SelfPlayVsTask(
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








