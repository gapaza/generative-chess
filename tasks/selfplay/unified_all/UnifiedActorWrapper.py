import config
import tasks.selfplay.utils as sp_utils
import tensorflow as tf
import chess
import chess.engine
from stockfish.rewards.reward_2 import calc_reward_move, calc_reward_move_simple
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


critic_loss_coef_warmup = 0.1
critic_loss_coef_final = 0.1
critic_warm_up_epochs = 55



illegal_move_reward = -0.0
checkmate_reward = 10.0
draw_reward = 0
move_bonus = 0.0


use_critic = True


class ActorWrapper:


    def __init__(self, actor, optimizer, critic, batch_size, task, model_num=1):
        self.actor = actor
        self.optimizer = optimizer
        self.critic = critic
        self.batch_size = batch_size
        self.task = task
        self.engine = task.engine
        self.num_actions = task.num_actions
        self.model_num = model_num
        self.model_dir = os.path.join(self.task.run_dir, 'model_' + str(self.model_num))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_info = {
            'checkmates': [],
            'loss': [],
            'critic_loss': [],
            'entropy': [],
            'kl': [],
            'return': [],
        }


        self.gamma = self.task.gamma
        self.lam = self.task.lam
        self.clip_ratio = self.task.clip_ratio
        self.target_kl = self.task.target_kl
        self.entropy_coef = self.task.entropy_coef

        # Update variables
        self.action_log_probs = [[] for _ in range(batch_size)]
        self.rewards = [[] for _ in range(batch_size)]
        self.values = [[] for _ in range(batch_size)]
        self.values_t = [[] for _ in range(batch_size)]
        self.actions = [[] for _ in range(batch_size)]
        self.actions_mask = [[] for _ in range(batch_size)]
        self.critic_actions_mask = [[] for _ in range(batch_size)]
        self.advantages = [[] for _ in range(self.batch_size)]
        self.returns = [[] for _ in range(self.batch_size)]

        self.all_log_probs = [[] for _ in range(batch_size)]
        self.all_probs = [[] for _ in range(batch_size)]

        self.checkmates = 0
        self.illegal_moves = 0
        self.draws = 0
        self.avg_reward = 0
        self.flat_rewards = []

    def reset_batch(self):
        self.action_log_probs = [[] for _ in range(self.batch_size)]
        self.actions = [[] for _ in range(self.batch_size)]
        self.actions_mask = [[] for _ in range(self.batch_size)]
        self.critic_actions_mask = [[] for _ in range(self.batch_size)]

        self.all_log_probs = [[] for _ in range(self.batch_size)]
        self.all_probs = [[] for _ in range(self.batch_size)]

        self.rewards = [[] for _ in range(self.batch_size)]
        self.values = [[] for _ in range(self.batch_size)]
        self.values_t = [[] for _ in range(self.batch_size)]
        self.advantages = [[] for _ in range(self.batch_size)]
        self.returns = [[] for _ in range(self.batch_size)]

        self.checkmates = 0
        self.illegal_moves = 0
        self.draws = 0
        self.avg_reward = 0
        self.flat_rewards = []

    def calc_reward(self, board, uci_move, prev_score, engine, prev_analysis):
        white_turn = (board.turn == chess.WHITE)

        # Check if illegal move
        if uci_move in config.non_move_tokens:
            self.illegal_moves += 1
            return True, illegal_move_reward, prev_score, prev_analysis
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            self.illegal_moves += 1
            return True, illegal_move_reward, prev_score, prev_analysis

        # Make move
        board.push(move)

        # Check if checkmateing move
        if board.is_checkmate():
            self.checkmates += 1
            return True, checkmate_reward, prev_score, prev_analysis
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            self.draws += 1
            return True, draw_reward, prev_score, prev_analysis

        # Analyze position
        analysis = engine.analyse(board, chess.engine.Limit(nodes=self.task.nodes), multipv=1)
        top_line = analysis[0]
        next_score = top_line["score"]
        color_turn = 'white' if white_turn is True else 'black'
        reward = calc_reward_move(color_turn, prev_score, next_score)
        # reward = calc_reward_move_simple(color_turn, prev_score, next_score)
        # reward += move_bonus
        return False, reward, next_score, analysis

    def get_actions(self, games, white_turn, games_done, scores, boards, analysis):

        # Get all legal uci moves for each board
        legal_moves_mask = []
        for idx, board in enumerate(boards):
            prev_analysis = analysis[idx]
            prev_score = scores[idx]

            mask = [0 for _ in range(config.vocab_size)]
            if prev_score.is_mate():
                top_line = prev_analysis[0]
                moves = top_line['pv']
                moves = [move.uci() for move in moves]
                best_move = moves[0]
                best_move_id = config.token2id[best_move]
                mask[best_move_id] = 1
            else:
                legal_uci_moves = [move.uci() for move in board.legal_moves]
                legal_uci_moves_ids = sp_utils.token2id(legal_uci_moves)
                for move_id in legal_uci_moves_ids:
                    mask[move_id] = 1

            legal_moves_mask.append(mask)
        legal_moves_mask = tf.convert_to_tensor(legal_moves_mask, dtype=tf.float32)


        actions_log_prob, actions, all_token_probs, all_token_log_probs = self.sample_actor(games, white_turn, legal_moves_mask)
        # print('action log prob', actions_log_prob)
        # print('actions', actions)
        # exit(0)


        actions_log_prob = actions_log_prob.numpy().tolist()
        actions = actions.numpy().tolist()
        all_token_probs = all_token_probs.numpy().tolist()
        all_token_log_probs = all_token_log_probs.numpy().tolist()

        gen_uci_moves = []
        all_rewards = []
        for idx in range(len(games)):
            if games_done[idx] is True:
                gen_uci_moves.append(None)
                continue

            # 1. Extract and record action
            action_id = int(actions[idx])
            action_token = config.id2token[action_id]
            gen_uci_moves.append(action_token)

            self.action_log_probs[idx].append(actions_log_prob[idx])
            self.actions[idx].append(action_id)
            self.actions_mask[idx].append(1)

            self.all_probs[idx].append(all_token_probs[idx])
            self.all_log_probs[idx].append(all_token_log_probs[idx])

            # 2. Calculate action reward
            board = boards[idx]
            prev_score = scores[idx]
            prev_analysis = analysis[idx]
            game_done, reward, next_score, next_analysis = self.calc_reward(board, action_token, prev_score, self.engine, prev_analysis)
            self.flat_rewards.append(deepcopy(reward))
            reward *= 0.01  # Scale reward for RL
            self.rewards[idx].append(reward)
            all_rewards.append(reward)
            games_done[idx] = game_done
            scores[idx] = next_score
            analysis[idx] = next_analysis

        # Set critic actions mask
        self.critic_actions_mask = deepcopy(self.actions_mask)
        for a_mask in self.critic_actions_mask:
            a_mask.append(1)

        return gen_uci_moves, games_done, scores, boards

    def pad_batch(self):
        for trajectory in self.action_log_probs:
            trajectory = sp_utils.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in self.actions_mask:
            trajectory = sp_utils.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in self.actions:
            trajectory = sp_utils.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in self.critic_actions_mask:
            trajectory = sp_utils.pad_list(trajectory, pad_token=0, pad_len=config.seq_length)
        for trajectory in self.all_log_probs:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])
        for trajectory in self.all_probs:
            while len(trajectory) < config.seq_length:
                trajectory.append([0 for _ in range(config.vocab_size)])

    def calc_advantages(self, games, white_turn):
        self.pad_batch()

        # 1. Sample critic
        self.values_t = self.sample_critic(games, white_turn)
        for idx in range(len(self.rewards)):
            self.rewards[idx].append(self.values_t[idx][-1])

        # 2. Calculate advantages
        flattened_advantages = []
        for idx in range(len(self.rewards)):
            trajectory_rewards = np.array(self.rewards[idx])
            trajectory_values = np.array(self.values_t[idx])
            deltas = trajectory_rewards[:-1] + self.gamma * trajectory_values[1:] - trajectory_values[:-1]
            adv = sp_utils.discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            flattened_advantages.extend(adv.tolist())
            self.advantages[idx] = adv.tolist()

            ret = sp_utils.discounted_cumulative_sums(
                trajectory_rewards, self.gamma
            )
            ret = sp_utils.pad_list(ret.tolist(), pad_token=0, pad_len=config.seq_length)
            self.returns[idx] = ret
        avg_advantage = np.mean(flattened_advantages)
        std_advantage = np.std(flattened_advantages)
        for idx in range(len(self.advantages)):
            adv = self.advantages[idx]
            adv_norm =[(a - avg_advantage) / (std_advantage + 1e-10) for a in adv]
            adv_norm = sp_utils.pad_list(adv_norm, pad_token=0, pad_len=config.seq_length)
            self.advantages[idx] = adv_norm


    def train_ppo(self, games, white_turn, other_wrapper):

        # Get model 1 inputs
        model_moves, cross_moves, model_turns = sp_utils.get_model_inputs_from_uci_games(games, white_turn)
        action_buffer = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        logprobability_buffer = tf.convert_to_tensor(self.action_log_probs, dtype=tf.float32)
        action_mask_tensor = tf.convert_to_tensor(self.actions_mask, dtype=tf.float32)
        advantage_buffer = tf.convert_to_tensor(self.advantages, dtype=tf.float32)
        logprobs_full = tf.convert_to_tensor(self.all_log_probs, dtype=tf.float32)
        probs_full = tf.convert_to_tensor(self.all_probs, dtype=tf.float32)
        returns, mask_tensor = self.get_critic_info()
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns = tf.expand_dims(returns, axis=-1)
        mask_tensor = tf.convert_to_tensor(mask_tensor, dtype=tf.float32)

        # Get other model inputs
        other_turn = not white_turn
        other_model_moves, other_cross_moves, other_model_turns = sp_utils.get_model_inputs_from_uci_games(games,
                                                                                                           other_turn)
        other_action_buffer = tf.convert_to_tensor(other_wrapper.actions, dtype=tf.int32)
        other_logprobability_buffer = tf.convert_to_tensor(other_wrapper.action_log_probs, dtype=tf.float32)
        other_action_mask_tensor = tf.convert_to_tensor(other_wrapper.actions_mask, dtype=tf.float32)
        other_advantage_buffer = tf.convert_to_tensor(other_wrapper.advantages, dtype=tf.float32)
        other_logprobs_full = tf.convert_to_tensor(other_wrapper.all_log_probs, dtype=tf.float32)
        other_probs_full = tf.convert_to_tensor(other_wrapper.all_probs, dtype=tf.float32)
        other_returns, other_mask_tensor = other_wrapper.get_critic_info()
        other_returns = tf.convert_to_tensor(other_returns, dtype=tf.float32)
        other_returns = tf.expand_dims(other_returns, axis=-1)
        other_mask_tensor = tf.convert_to_tensor(other_mask_tensor, dtype=tf.float32)

        # Concatenate model inputs
        model_moves = tf.concat([model_moves, other_model_moves], axis=0)
        cross_moves = tf.concat([cross_moves, other_cross_moves], axis=0)
        model_turns = tf.concat([model_turns, other_model_turns], axis=0)
        action_buffer = tf.concat([action_buffer, other_action_buffer], axis=0)
        logprobability_buffer = tf.concat([logprobability_buffer, other_logprobability_buffer], axis=0)
        action_mask_tensor = tf.concat([action_mask_tensor, other_action_mask_tensor], axis=0)
        advantage_buffer = tf.concat([advantage_buffer, other_advantage_buffer], axis=0)
        logprobs_full = tf.concat([logprobs_full, other_logprobs_full], axis=0)
        probs_full = tf.concat([probs_full, other_probs_full], axis=0)
        returns = tf.concat([returns, other_returns], axis=0)
        mask_tensor = tf.concat([mask_tensor, other_mask_tensor], axis=0)

        if self.task.curr_epoch < critic_warm_up_epochs:
            critic_coef = critic_loss_coef_warmup
        else:
            critic_coef = critic_loss_coef_final
        critic_coef = tf.constant(critic_coef, dtype=tf.float32)

        # Train actor
        policy_update_itr = 0
        value_loss = 0
        for i in range(self.task.train_actor_itr):
            policy_update_itr += 1
            # ---> TODO: change back to critic
            kl, entr, loss, policy_loss = self.train_actor(
                model_moves, cross_moves, model_turns,
                action_buffer, logprobability_buffer, action_mask_tensor, advantage_buffer, logprobs_full, probs_full
            )
            if abs(kl) > 1.5 * self.task.target_kl:
                # Early Stopping
                break

        kl = kl.numpy()
        entr = entr.numpy()
        loss = loss.numpy()
        policy_loss = policy_loss.numpy()

        model_color = 'Unified'
        model_color += ' (' + str(self.model_num) + '):'
        combined_rewards = self.flat_rewards + other_wrapper.flat_rewards
        combined_checkmates = self.checkmates + other_wrapper.checkmates

        print(model_color, 'avg reward', np.mean(combined_rewards), 'critic loss', value_loss, 'critic coef', critic_coef.numpy(), 'kl', kl, 'entr', entr, 'loss', loss, 'policy_loss', policy_loss, 'checkmates', combined_checkmates, 'illegal_moves', self.illegal_moves, 'policy_update_itr', policy_update_itr)

        self.model_info['checkmates'].append(combined_checkmates)
        self.model_info['loss'].append(loss)
        self.model_info['entropy'].append(entr)
        self.model_info['kl'].append(kl)
        self.model_info['return'].append(np.mean(combined_rewards))
        self.model_info['critic_loss'].append(value_loss)

        return kl, entr, loss, policy_loss







    def train_ppo_unified(self, games, white_turn, other_wrapper):

        # Get model 1 inputs
        model_moves, cross_moves, model_turns = sp_utils.get_model_inputs_from_uci_games(games, white_turn)
        action_buffer = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        logprobability_buffer = tf.convert_to_tensor(self.action_log_probs, dtype=tf.float32)
        action_mask_tensor = tf.convert_to_tensor(self.actions_mask, dtype=tf.float32)
        advantage_buffer = tf.convert_to_tensor(self.advantages, dtype=tf.float32)
        logprobs_full = tf.convert_to_tensor(self.all_log_probs, dtype=tf.float32)
        probs_full = tf.convert_to_tensor(self.all_probs, dtype=tf.float32)
        returns, mask_tensor = self.get_critic_info()
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns = tf.expand_dims(returns, axis=-1)
        mask_tensor = tf.convert_to_tensor(mask_tensor, dtype=tf.float32)

        # Get other model inputs
        other_turn = not white_turn
        other_model_moves, other_cross_moves, other_model_turns = sp_utils.get_model_inputs_from_uci_games(games, other_turn)
        other_action_buffer = tf.convert_to_tensor(other_wrapper.actions, dtype=tf.int32)
        other_logprobability_buffer = tf.convert_to_tensor(other_wrapper.action_log_probs, dtype=tf.float32)
        other_action_mask_tensor = tf.convert_to_tensor(other_wrapper.actions_mask, dtype=tf.float32)
        other_advantage_buffer = tf.convert_to_tensor(other_wrapper.advantages, dtype=tf.float32)
        other_logprobs_full = tf.convert_to_tensor(other_wrapper.all_log_probs, dtype=tf.float32)
        other_probs_full = tf.convert_to_tensor(other_wrapper.all_probs, dtype=tf.float32)
        other_returns, other_mask_tensor = other_wrapper.get_critic_info()
        other_returns = tf.convert_to_tensor(other_returns, dtype=tf.float32)
        other_returns = tf.expand_dims(other_returns, axis=-1)
        other_mask_tensor = tf.convert_to_tensor(other_mask_tensor, dtype=tf.float32)

        # Concatenate model inputs
        model_moves = tf.concat([model_moves, other_model_moves], axis=0)
        cross_moves = tf.concat([cross_moves, other_cross_moves], axis=0)
        model_turns = tf.concat([model_turns, other_model_turns], axis=0)
        action_buffer = tf.concat([action_buffer, other_action_buffer], axis=0)
        logprobability_buffer = tf.concat([logprobability_buffer, other_logprobability_buffer], axis=0)
        action_mask_tensor = tf.concat([action_mask_tensor, other_action_mask_tensor], axis=0)
        advantage_buffer = tf.concat([advantage_buffer, other_advantage_buffer], axis=0)
        logprobs_full = tf.concat([logprobs_full, other_logprobs_full], axis=0)
        probs_full = tf.concat([probs_full, other_probs_full], axis=0)
        returns = tf.concat([returns, other_returns], axis=0)
        mask_tensor = tf.concat([mask_tensor, other_mask_tensor], axis=0)

        if self.task.curr_epoch < critic_warm_up_epochs:
            critic_coef = critic_loss_coef_warmup
        else:
            critic_coef = critic_loss_coef_final
        critic_coef = tf.constant(critic_coef, dtype=tf.float32)

        # Train actor
        policy_update_itr = 0
        value_loss = 0
        for i in range(self.task.train_actor_itr):
            policy_update_itr += 1
            # ---> TODO: change back to critic
            # kl, entr, loss, policy_loss = self.train_actor(
            #     model_moves, cross_moves, model_turns,
            #     action_buffer, logprobability_buffer, action_mask_tensor, advantage_buffer, logprobs_full, probs_full
            # )
            kl, entr, loss, policy_loss, value_loss = self.train_actor_critic(
                model_moves, cross_moves, model_turns,
                action_buffer, logprobability_buffer, action_mask_tensor, advantage_buffer, logprobs_full, probs_full,
                returns, mask_tensor,
                critic_coef
            )
            # kl_1, entr_1, loss_1, policy_loss_1, value_loss_1 = self.train_actor_critic(
            #     model_moves, cross_moves, model_turns,
            #     action_buffer, logprobability_buffer, action_mask_tensor, advantage_buffer, logprobs_full, probs_full,
            #     returns, mask_tensor
            # )
            # kl_2, entr_2, loss_2, policy_loss_2, value_loss_2 = self.train_actor_critic(
            #     other_model_moves, other_cross_moves, other_model_turns,
            #     other_action_buffer, other_logprobability_buffer, other_action_mask_tensor, other_advantage_buffer, other_logprobs_full, other_probs_full,
            #     other_returns, other_mask_tensor
            # )
            # kl = (kl_1 + kl_2) / 2
            # entr = (entr_1 + entr_2) / 2
            # loss = (loss_1 + loss_2) / 2
            # policy_loss = (policy_loss_1 + policy_loss_2) / 2
            # value_loss = (value_loss_1 + value_loss_2) / 2
            if abs(kl) > 1.5 * self.task.target_kl:
                # Early Stopping
                break

        kl = kl.numpy()
        entr = entr.numpy()
        loss = loss.numpy()
        policy_loss = policy_loss.numpy()
        value_loss = value_loss.numpy()



        model_color = 'Unified'
        model_color += ' (' + str(self.model_num) + '):'
        combined_rewards = self.flat_rewards + other_wrapper.flat_rewards
        combined_checkmates = self.checkmates + other_wrapper.checkmates

        print(model_color, 'avg reward', np.mean(combined_rewards), 'critic loss', value_loss, 'critic coef', critic_coef.numpy(), 'kl', kl, 'entr', entr, 'loss', loss, 'policy_loss', policy_loss, 'checkmates', combined_checkmates, 'illegal_moves', self.illegal_moves, 'policy_update_itr', policy_update_itr)

        self.model_info['checkmates'].append(combined_checkmates)
        self.model_info['loss'].append(loss)
        self.model_info['entropy'].append(entr)
        self.model_info['kl'].append(kl)
        self.model_info['return'].append(np.mean(combined_rewards))
        self.model_info['critic_loss'].append(value_loss)

        return kl, entr, loss, policy_loss

    def get_critic_info(self):
        return self.returns, self.critic_actions_mask

    def sample_critic(self, games, white_turn):
        model_moves, cross_moves, model_turns = sp_utils.get_model_inputs_from_uci_games(games, white_turn)
        values_t = self._sample_critic(model_moves, cross_moves, model_turns)
        values_t = values_t.numpy().tolist()
        values_t_list = []
        for idx, values in enumerate(values_t):
            values_mask = self.critic_actions_mask[idx]
            subvalues_t = []
            for mask, v in zip(values_mask, values):
                if mask == 1:
                    subvalues_t.append(v)
            values_t_list.append(subvalues_t)
        return values_t_list

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
    ])
    def _sample_critic(self, model_moves, cross_moves, is_white):
        # ---> TODO: change back to critic
        inputs = [model_moves, cross_moves, is_white]
        pred, t_value = self.critic(inputs, training=False)  # (batch, seq_len, 2)
        # pred, t_value = self.actor(inputs, training=False)
        t_value = t_value[:, :, 0]
        return t_value

    def sample_actor(self, games, white_turn, legal_moves_mask):
        all_model_moves = []
        all_cross_moves = []
        all_white_turns = []
        max_inf_idx = 0
        for game in games:
            game_ids = sp_utils.token2id(game)
            model_moves, cross_moves, inf_idx = sp_utils.get_inputs_from_game(game_ids, white_turn)
            model_moves = sp_utils.pad_list(model_moves, pad_len=config.seq_length)
            cross_moves = sp_utils.pad_list(cross_moves, pad_len=config.seq_length)
            all_model_moves.append(model_moves)
            all_cross_moves.append(cross_moves)
            all_white_turns.append(white_turn)
            if inf_idx > max_inf_idx:
                max_inf_idx = inf_idx
        model_moves = tf.convert_to_tensor(all_model_moves, dtype=tf.int32)
        cross_moves = tf.convert_to_tensor(all_cross_moves, dtype=tf.int32)
        white_turns = tf.convert_to_tensor(all_white_turns, dtype=tf.bool)
        return self._sample_actor(model_moves, cross_moves, white_turns, max_inf_idx, legal_moves_mask)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # shape=(global_mini_batch_size, None)
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def _sample_actor(self, model_moves, cross_moves, is_white, inf_idx, legal_moves_mask):
        # legal_moves_mask = tf.expand_dims(legal_moves_mask, axis=-1)  # shape (batch, 1, vocab_size)

        observation_input = [model_moves, cross_moves, is_white]
        pred_logits, pred_values = self.actor(observation_input)

        pred_probs = tf.nn.softmax(pred_logits, axis=-1)  # shape (batch, seq_len, vocab_size)
        all_token_probs = pred_probs[:, inf_idx, :]  # shape (batch, 2)
        all_token_log_probs = tf.math.log(all_token_probs + 1e-10)
        # samples = tf.random.categorical(all_token_log_probs, 1)  # shape (batch, 1)

        # Masked sampling
        inf_logits = pred_logits[:, inf_idx, :]  # shape (batch, vocab_size)
        masked_logits = tf.where(legal_moves_mask == 1, inf_logits, tf.fill(tf.shape(inf_logits), -1e9))
        masked_probs = tf.nn.softmax(masked_logits, axis=-1)
        masked_log_probs = tf.math.log(masked_probs + 1e-10)
        samples = tf.random.categorical(masked_log_probs, 1)  # shape (batch, 1)

        next_bit_ids = tf.squeeze(samples, axis=-1)  # shape (batch,)
        batch_indices = tf.range(0, tf.shape(all_token_log_probs)[0], dtype=tf.int64)  # shape (batch,)
        next_bit_probs = tf.gather_nd(all_token_log_probs, tf.stack([batch_indices, next_bit_ids], axis=-1))
        actions = next_bit_ids  # (batch,)
        actions_log_prob = next_bit_probs  # (batch,)
        return actions_log_prob, actions, all_token_probs, all_token_log_probs

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),

        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),

        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ])
    def train_actor_critic(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            action_buffer,
            logprobability_buffer,
            color_action_mask_tensor,
            color_advantage_buffer,
            logprobs_full,
            probs_full,
            return_buffer, mask_tensor,
            critic_coef
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.actor(inputs)  # shape: (batch, seq_len, 2)
            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)

            # --- Find true advantage
            pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.cast(pred_log_probs, dtype=tf.float32)
            one_hot_actions = tf.one_hot(action_buffer, self.num_actions)
            one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
            logprobability = tf.reduce_sum(
                one_hot_actions * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            color_logprobs = logprobability * color_action_mask_tensor
            color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            white_ratio = tf.exp(
                color_logprobs - color_logprobs_buffer
            )
            white_true_advantage = white_ratio * color_advantage_buffer

            # --- Find min advantage
            white_min_advantage = tf.where(
                color_advantage_buffer > 0,
                (1 + self.clip_ratio) * color_advantage_buffer,
                (1 - self.clip_ratio) * color_advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(white_true_advantage, white_min_advantage)
            )
            loss = 0
            loss += policy_loss

            # Model Entropy
            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

            # Critic Loss
            value_loss = self.task.critic_loss_fn(return_buffer, pred_vals, sample_weight=mask_tensor)
            loss += critic_coef * value_loss


        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        #  KL Divergence
        # pred_probs, pred_vals = self.c_actor(inputs)
        # pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        # pred_log_probs = tf.math.log(pred_probs)
        # one_hot_actions = tf.one_hot(action_buffer, self.num_actions)
        # one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
        # logprobability = tf.reduce_sum(
        #     one_hot_actions * pred_log_probs, axis=-1
        # )  # shape (batch, seq_len)
        # color_logprobs = logprobability * color_action_mask_tensor
        # color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
        # kl = tf.reduce_mean(
        #     color_logprobs_buffer - color_logprobs
        # )
        # kl_old = tf.reduce_sum(kl)

        # True KL Divergence
        mask_tensor = tf.expand_dims(color_action_mask_tensor, axis=-1)
        pred_probs, pred_vals = self.actor(inputs)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        pred_log_probs = pred_log_probs * mask_tensor
        probs_full = probs_full * mask_tensor
        logprobs_full = logprobs_full * mask_tensor
        true_kl = tf.reduce_sum(
            probs_full * (logprobs_full - pred_log_probs),
            axis=-1
        )  # shape (9, 280)
        kl = tf.reduce_mean(true_kl)  # shape (1,)

        return kl, entr, loss, policy_loss, value_loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),

        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    ])
    def train_actor(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            action_buffer,
            logprobability_buffer,
            color_action_mask_tensor,
            color_advantage_buffer,
            logprobs_full,
            probs_full
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:
            pred_probs, pred_vals = self.actor(inputs)  # shape: (batch, seq_len, 2)
            pred_probs = tf.nn.softmax(pred_probs, axis=-1)  # shape: (batch, seq_len, 2)

            # --- Find true advantage
            pred_log_probs = tf.math.log(pred_probs + 1e-10)  # shape: (batch, seq_len, 2)
            pred_log_probs = tf.cast(pred_log_probs, dtype=tf.float32)
            one_hot_actions = tf.one_hot(action_buffer, self.num_actions)
            one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
            logprobability = tf.reduce_sum(
                one_hot_actions * pred_log_probs, axis=-1
            )  # shape (batch, seq_len)
            color_logprobs = logprobability * color_action_mask_tensor
            color_logprobs_buffer = logprobability_buffer * color_action_mask_tensor
            white_ratio = tf.exp(
                color_logprobs - color_logprobs_buffer
            )
            white_true_advantage = white_ratio * color_advantage_buffer

            # --- Find min advantage
            white_min_advantage = tf.where(
                color_advantage_buffer > 0,
                (1 + self.clip_ratio) * color_advantage_buffer,
                (1 - self.clip_ratio) * color_advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(white_true_advantage, white_min_advantage)
            )
            loss = 0
            loss += policy_loss

            entr = -tf.reduce_sum(pred_probs * pred_log_probs, axis=-1)  # shape (batch, seq_len)
            entr = tf.reduce_mean(entr)  # Higher positive value means more exploration - shape (batch,)
            loss = loss - (self.entropy_coef * entr)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        # True KL Divergence
        mask_tensor = tf.expand_dims(color_action_mask_tensor, axis=-1)
        pred_probs, pred_vals = self.actor(inputs)
        pred_probs = tf.nn.softmax(pred_probs, axis=-1)
        pred_log_probs = tf.math.log(pred_probs + 1e-10)
        pred_log_probs = pred_log_probs * mask_tensor
        probs_full = probs_full * mask_tensor
        logprobs_full = logprobs_full * mask_tensor
        true_kl = tf.reduce_sum(
            probs_full * (logprobs_full - pred_log_probs),
            axis=-1
        )  # shape (9, 280)
        kl = tf.reduce_mean(true_kl)  # shape (1,)

        return kl, entr, loss, policy_loss

    def plot_model(self):

        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=(14, 6.5))  # default [6.4, 4.8], W x H  9x6, 12x8
        fig.suptitle('Model ' + str(self.model_num), fontsize=16)

        # Returns plot
        plt.subplot(gs[0, 0])
        plt.plot(self.model_info['return'])
        plt.title('Returns')
        plt.xlabel('Epochs')
        plt.ylabel('Return')

        # Checkmates plot
        plt.subplot(gs[0, 1])
        plt.plot(self.model_info['checkmates'])
        plt.title('Checkmates')
        plt.xlabel('Epochs')
        plt.ylabel('Checkmates')

        # Critic Loss plot
        plt.subplot(gs[0, 2])
        plt.plot(self.model_info['critic_loss'])
        plt.title('Critic Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Critic Loss')

        # KL plot
        plt.subplot(gs[1, 0])
        plt.plot(self.model_info['kl'])
        plt.title('KL Divergence')
        plt.xlabel('Epochs')
        plt.ylabel('KL')

        # Entropy plot
        plt.subplot(gs[1, 1])
        plt.plot(self.model_info['entropy'])
        plt.title('Entropy')
        plt.xlabel('Epochs')
        plt.ylabel('Entropy')

        plt.tight_layout()
        save_file = 'model_' + str(self.model_num) + '_plots.png'
        save_path = os.path.join(self.model_dir, save_file)
        plt.savefig(save_path)
        plt.close()











