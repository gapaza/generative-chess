import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import random
import math
import json
import config
import matplotlib.pyplot as plt
import os
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tasks.AbstractTask import AbstractTask
import scipy.signal
from model import get_rl_models as get_model
from collections import OrderedDict
import tensorflow_addons as tfa
import chess
import chess.engine

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def id_seq_to_uci(seq):
    return [config.id2token[x] for x in seq]


# Number of self-play games in a mini-batch
global_mini_batch_size = 32


use_actor_warmup = True
use_critic_warmup = False

pickup_epoch = 0

run_dir = 0
run_dir_itr = 0


class SelfPlayTask(AbstractTask):

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
        super(SelfPlayTask, self).__init__(run_num, None, problem, epochs, actor_load_path, critic_load_path)
        self.debug = debug
        self.run_val = run_val
        self.val_itr = val_itr

        # Algorithm parameters
        self.mini_batch_size = global_mini_batch_size
        self.nfe = 0
        self.epochs = epochs
        self.max_steps_per_game = config.seq_length - 1 # 30 | 60

        # PPO alg parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.001  # was 0.01
        self.entropy_coef = 0.00  # was 0.02 originally
        self.counter = 0
        self.game_start_token_id = config.start_token_id
        self.num_actions = config.vocab_size
        self.curr_epoch = 0

        # Results
        self.plot_freq = 5

        # Pretrain save dir
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.actor_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights')
        self.critic_pretrain_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights')

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 12, "Hash": 1024})
        self.nodes = 100000
        self.lines = 1




    def build(self):

        # Optimizer parameters
        self.actor_learning_rate = 0.0001  # 0.0001
        self.critic_learning_rate = 0.0005  # 0.0001
        self.train_actor_iterations = 250  # was 250
        self.train_critic_iterations = 40  # was 40
        self.beta_1 = 0.9
        if self.run_val is False:
            self.beta_1 = 0.0

        if use_actor_warmup is True:
            self.actor_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                0.0,  # initial learning rate
                1000,  # decay_steps
                alpha=1.0,
                warmup_target=self.actor_learning_rate,
                warmup_steps=500
            )

        # Optimizers
        if self.actor_optimizer is None:
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate)
            # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.actor_learning_rate, beta_1=self.beta_1)
            self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.actor_learning_rate)
        if self.critic_optimizer is None:
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate)
            # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.critic_learning_rate, beta_1=self.beta_1)
            # self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.critic_learning_rate)

        self.c_actor, self.c_critic = get_model(self.actor_load_path, self.critic_load_path)

        self.c_actor.summary()


    def run(self):
        self.build()

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
        all_total_rewards = []
        all_actions = [[] for _ in range(self.mini_batch_size)]
        all_rewards = [[] for _ in range(self.mini_batch_size)]
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
        # Sample Actor
        # -------------------------------------

        for t in range(self.max_steps_per_game):
            # topk_action_log_prob, topk_action, topk_action_probs = self.sample_actor(observation)  # returns shape: (batch,) and (batch,)
            # topk_action_log_prob, topk_action = topk_action_log_prob.numpy().tolist(), topk_action.numpy().tolist()
            # action_log_prob, action = [], []
            # for idx in range(self.mini_batch_size):
            #     # Get top actions for batch element
            #     # print('Top Actions:', topk_action[idx], id_seq_to_uci(topk_action[idx]))
            #     sample_action, sample_action_prob = self.sample_actions(boards[idx], topk_action[idx], topk_action_log_prob[idx])
            #     action.append(sample_action)
            #     action_log_prob.append(sample_action_prob)


            action_log_prob, action, all_action_probs = self.sample_actor(observation)  # returns shape: (batch,) and (batch,)



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
                    if idx == 0:
                        print(config.id2token[m_action], end=' ')

            # Determine reward for each batch element
            if len(games[0]) == self.max_steps_per_game:
                done = True
                for idx, game in enumerate(games):
                    if ended_games[idx] is False:

                        # Evaluate latest move
                        reward, game_ended, new_eval = self.calc_reward(
                            game,
                            game_evals[idx],
                            boards[idx]
                        )
                        all_rewards[idx].append(reward)
                        if game_ended is False:
                            game_evals[idx] = new_eval

                        if game_ended != ended_games[idx]:
                            epoch_games.append(' '.join([config.id2token[x] for x in game]))

                        ended_games[idx] = game_ended
                    else:
                        all_rewards[idx].append(0)
            else:
                done = False
                for idx, game in enumerate(games):
                    if ended_games[idx] is False:
                        # Evaluate latest move
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
                        all_rewards[idx].append(reward)
                    else:
                        all_rewards[idx].append(0)

            if all(ended_games):
                done = True


            # Update the observation
            if done is True:
                critic_observation_buffer = deepcopy(observation_new)
            else:
                observation = observation_new


        for trajectory in observation:
            if len(trajectory) < config.seq_length:
                trajectory += [0] * ((config.seq_length-1) - len(trajectory))
        for trajectory in action_mask:
            if len(trajectory) < config.seq_length:
                trajectory += [0] * ((config.seq_length-1) - len(trajectory))

        for trajectory in critic_observation_buffer:
            if len(trajectory) < config.seq_length:
                trajectory += [0] * (config.seq_length - len(trajectory))



        print('')
        # -------------------------------------
        # Sample Critic
        # -------------------------------------

        # --- SINGLE CRITIC PREDICTION --- #
        value_t = self.sample_critic(critic_observation_buffer)
        value_t = value_t.numpy().tolist()  # (30, 31)
        for idx, value in zip(range(self.mini_batch_size), value_t):
            last_reward = value[-1]
            all_rewards[idx].append(last_reward)

        # -------------------------------------
        # Calculate Advantage and Return
        # -------------------------------------

        proc_time = time.time()
        all_advantages = [[] for _ in range(self.mini_batch_size)]
        all_returns = [[] for _ in range(self.mini_batch_size)]
        for idx in range(len(all_rewards)):
            rewards = np.array(all_rewards[idx])
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

        # -------------------------------------
        # Train Actor
        # -------------------------------------

        curr_time = time.time()
        policy_update_itr = 0
        for i in range(self.train_actor_iterations):
            policy_update_itr += 1
            kl, entr, policy_loss, actor_loss = self.train_actor(
                observation_tensor,
                action_tensor,
                logprob_tensor,
                advantage_tensor,
                action_mask_tensor
            )
            if abs(kl) > 1.5 * self.target_kl:
                # Early Stopping
                break

        # -------------------------------------
        # Train Critic
        # -------------------------------------

        curr_time = time.time()
        for i in range(self.train_critic_iterations):
            value_loss = self.train_critic(
                critic_observation_tensor,
                return_tensor,
            )

        # Update results tracker
        epoch_info = {
            'mb_return': np.mean(all_rewards),
            'c_loss': value_loss.numpy(),
            'p_loss': policy_loss.numpy(),
            'p_iter': policy_update_itr,
            'entropy': entr.numpy(),
            'kl': kl.numpy(),
        }

        # print('GAME:', epoch_games[0])

        return epoch_info



    # This is only to be called if the game has not finished
    # All position evals are from the perspective of the white player
    def calc_reward(self, game, prev_eval, board):
        uci_moves = [config.id2token[x] for x in game]
        last_move = uci_moves[-1]

        # 0. Determine the turn color
        white_turn = (board.turn == chess.WHITE)

        # 1. Check if a legal move has been made
        if last_move not in config.end_of_game_tokens and last_move not in config.special_tokens:
            move = chess.Move.from_uci(last_move)
            if move not in board.legal_moves:
                return -1, True, prev_eval
        else:
            return -1, True, prev_eval

        # 2. Check if checkmating move
        board.push(move)
        if board.is_checkmate():
            return 1, True, prev_eval

        # 3. Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return -0.1, True, prev_eval


        # 4. Calculate reward for non-checkmating legal move
        analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)
        new_eval = self.parse_analysis(analysis)
        if new_eval is None:
            new_eval = prev_eval
        else:
            new_eval = new_eval / 100.0  # Convert centipawns to pawns
        # print('New eval:', new_eval)

        eval_norm = 10.0
        reward = abs(new_eval - prev_eval) / eval_norm
        if white_turn and new_eval < prev_eval:
            reward *= -1
        elif not white_turn and new_eval > prev_eval:
            reward *= -1

        # print('Move:', last_move, 'Prev eval:', prev_eval, 'New eval:', new_eval, 'Reward:', reward)


        return reward, False, new_eval

    def parse_analysis(self, analysis):
        # print('Analysis', analysis)
        for idx, line in enumerate(analysis):
            # line_top_move = line["pv"][0].uci()
            line_top_move_score = line["score"].white().score()
            return line_top_move_score
        return 0.0

    # -------------------------------------
    # Actor-Critic Functions
    # -------------------------------------

    def sample_actor(self, observation):
        inf_idx = len(observation[0]) - 1  # all batch elements have the same length
        observation_input = deepcopy(observation)
        observation_input = tf.convert_to_tensor(observation_input, dtype=tf.int32)
        inf_idx = tf.convert_to_tensor(inf_idx, dtype=tf.int32)
        return self._sample_actor(observation_input, inf_idx)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ])
    def _sample_actor(self, observation_input, inf_idx):
        # print('sampling actor', inf_idx)
        pred_probs = self.c_actor(observation_input)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape (batch, seq_len, vocab_size)

        # Get top k moves
        # k = 5
        # pred_probs_inf = pred_probs[:, inf_idx, :]  # shape (batch, vocab_size)
        # top_probs, top_indices = tf.math.top_k(pred_probs_inf, k=k) # shape (batch, k), shape (batch, k)
        # top_probs_log = tf.math.log(top_probs + 1e-10)
        # return top_probs_log, top_indices, pred_probs


        # Batch sampling
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)
        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))

        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs

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
        t_value = self.c_critic.vcall(observation_input)  # (batch, seq_len, 2)
        t_value = t_value[:, :, 0]
        return t_value




    @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
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
            action_mask_tensor
    ):
        with tf.GradientTape() as tape:
            pred_probs = self.c_actor(observation_buffer)  # shape: (batch, seq_len, 2)
            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.math.log(pred_probs)  # shape: (batch, seq_len, 2)
            logprobability = tf.reduce_sum(
                tf.one_hot(action_buffer, self.num_actions) * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            logprobability *= action_mask_tensor

            # Total loss
            loss = 0

            # PPO Surrogate Loss
            ratio = tf.exp(
                logprobability - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
            loss += policy_loss

            # Entropy Term
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        policy_grads = tape.gradient(loss, self.c_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.c_actor.trainable_variables))

        #  KL Divergence
        pred_probs = self.c_actor(observation_buffer)
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

        return kl, entr, policy_loss, loss

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
            pred_values = self.c_critic.vcall(observation_buffer)  # (batch, seq_len, 2)

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
            print('')


        # Update metrics
        self.returns.append(epoch_info['mb_return'])
        self.c_loss.append(epoch_info['c_loss'])
        self.p_loss.append(epoch_info['p_loss'])
        self.p_iter.append(epoch_info['p_iter'])
        self.entropy.append(epoch_info['entropy'])
        self.kl.append(epoch_info['kl'])

        if len(self.entropy) % self.plot_freq == 0:
            print('--> PLOTTING')
            self.plot_ppo()
        else:
            return



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

    # actor_path = None
    # critic_path = None

    task = SelfPlayTask(
        run_num=run_dir,
        problem=None,
        epochs=500,
        actor_load_path=actor_path,
        critic_load_path=critic_path,
        debug=True,
        run_val=False,
        val_itr=run_dir_itr,
    )
    task.run()