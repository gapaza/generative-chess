import numpy as np
import tensorflow as tf
from copy import deepcopy
import random
import config
import chess
import chess.engine
import os
from evals.Arch2Evals import Arch2Evals as Evals
import tensorflow_addons as tfa
from evals.plotting.training_comparison import plot_training_comparison
from tasks.selfplay.ActorWrapper import ActorWrapper
import tasks.selfplay.utils as sp_utils

# -------------
# --- Seeds ---
# -------------

seed = 0
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# ----------------
# --- Settings ---
# ----------------

pickup_epoch = 0
global_mini_batch_size = 16
run_evals = True
alternate_colors = False
run_dir = 6
run_num = 0
plot_freq = 10
save_freq = 50
actor_lr = 0.00001
critic_lr = 0.00001
actor_itr = 250
critic_itr = 40

update_kl = 0.0005
update_entropy = 0.00
eval_nodes = 10000

# --------------
# --- Models ---
# --------------

from model import get_rl_models_a2 as get_model
actor_1_path = config.model_path
actor_2_path = config.model_path
critic_path = config.model_path



class Task:

    def __init__(
            self,
            epochs=10000,
    ):

        # Evals
        self.eval_themes = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
        self.a1_evals = None
        self.a2_evals = None
        if run_evals is True:
            self.a1_evals = Evals(themes=self.eval_themes)
            self.a2_evals = Evals(themes=self.eval_themes)

        # Algorithm Parameters
        self.mini_batch_size = global_mini_batch_size
        self.epochs = epochs
        self.max_steps_per_game = config.seq_length - 1
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.target_kl = update_kl
        self.entropy_coef = update_entropy
        self.counter = 0
        self.num_actions = config.vocab_size
        self.curr_epoch = 0
        self.train_actor_itr = actor_itr
        self.train_critic_itr = critic_itr

        # Directories
        self.run_root = config.results_dir
        self.run_dir = os.path.join(self.run_root, 'run_' + str(run_dir))
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        self.pretrain_save_dir = os.path.join(self.run_dir, 'pretrained')
        if not os.path.exists(self.pretrain_save_dir):
            os.makedirs(self.pretrain_save_dir)
        self.games_dir = os.path.join(self.run_dir, 'games')
        if not os.path.exists(self.games_dir):
            os.makedirs(self.games_dir)
        else:
            for file in os.listdir(self.games_dir):
                os.remove(os.path.join(self.games_dir, file))

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 32, "Hash": 4096 * 2})
        self.nodes = eval_nodes
        self.lines = 1
        init_analysis = self.engine.analyse(chess.Board(), chess.engine.Limit(nodes=self.nodes), multipv=1)
        self.init_score = init_analysis[0]["score"]

        # Additional info
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

        # Optimizers
        self.model_1_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=actor_lr)
        self.model_2_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=actor_lr)
        self.critic_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=critic_lr)

        # Critic Loss
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()

        # Models
        self.actor_1, self.actor_2, self.critic = get_model(
            actor_1_path,
            actor_2_path,
            critic_path,
        )
        self.actor_1_wrapper = ActorWrapper(self.actor_1, self.model_1_optimizer, self.critic, global_mini_batch_size, self, 1)
        self.actor_2_wrapper = ActorWrapper(self.actor_2, self.model_2_optimizer, self.critic, global_mini_batch_size, self, 2)

    def run_evals(self):
        if run_evals is False:
            return

        # Model 1 evaluations
        eval_history = self.a1_evals.run_eval(self.actor_1)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'a1_evals_' + str(run_num) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)

        # Model 2 evaluations
        eval_history = self.a2_evals.run_eval(self.actor_2)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'a2_evals_' + str(run_num) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)

    def save_models(self):
        t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch + pickup_epoch))
        t_actor_save_path_2 = os.path.join(self.pretrain_save_dir, 'actor_weights_2_' + str(self.curr_epoch + pickup_epoch))
        t_critic_save_path = os.path.join(self.pretrain_save_dir, 'critic_weights_' + str(self.curr_epoch + pickup_epoch))
        self.actor_1.save_weights(t_actor_save_path)
        self.actor_2.save_weights(t_actor_save_path_2)
        self.critic.save_weights(t_critic_save_path)


    def record(self):
        pass

    def run(self):
        self.run_evals()
        for x in range(self.epochs):
            self.curr_epoch = x

            # 1. Run mini-batch
            self.run_mini_batch()

            # 2. Record mini-batch
            self.record()

            # 3. Save models (if needed)
            if self.curr_epoch % save_freq == 0 and self.curr_epoch > 0:
                self.save_models()

            # 4. Run evals
            if self.curr_epoch % plot_freq == 0 and self.curr_epoch > 0:
                self.run_evals()

        self.save_models()


    def run_mini_batch(self):
        # First, reset wrappers
        self.actor_1_wrapper.reset_batch()
        self.actor_2_wrapper.reset_batch()


        # Determine which color is played by which model
        models = [self.actor_1_wrapper, self.actor_2_wrapper]
        if alternate_colors:
            if self.curr_epoch % 2 == 0:
                models = [self.actor_2_wrapper, self.actor_1_wrapper]
        white_model = models[0]
        black_model = models[1]

        # Play games
        games = [[] for _ in range(self.mini_batch_size)]  # move sequence of each game in uci
        boards = [chess.Board() for x in range(self.mini_batch_size)]
        scores = [self.init_score for x in range(self.mini_batch_size)]
        games_done = [False for x in range(self.mini_batch_size)]

        for move_idx in range(self.max_steps_per_game):
            if move_idx % 2 == 0:  # even number
                actor_wrapper = white_model
                white_turn = True
            else:  # odd number
                actor_wrapper = black_model
                white_turn = False

            # Get actions
            actions, games_done, scores, boards = actor_wrapper.get_actions(games, white_turn, games_done, scores, boards)
            for idx, action in enumerate(actions):
                if action is not None:
                    games[idx].append(action)

        avg_game_len = np.mean([len(game) for game in games])
        if self.curr_epoch % 10 == 0:
            game_file_name = 'game_' + str(self.curr_epoch) + '.pgn'
            sp_utils.save_game_pgn(games[0], self.games_dir, file_name=game_file_name)


        white_model.calc_advantages(games, True)
        black_model.calc_advantages(games, False)

        # Each call returns: kl, entr, loss, policy_loss
        w_kl, w_entr, w_loss, w_policy_loss = white_model.train_ppo(games, True)
        b_kl, b_entr, b_loss, b_policy_loss = black_model.train_ppo(games, False)

        if self.curr_epoch % plot_freq == 0 and self.curr_epoch > 0:
            white_model.plot_model()
            black_model.plot_model()





        # If separate critic model
        w_returns, w_c_mask = white_model.get_critic_info()
        b_returns, b_c_mask = black_model.get_critic_info()
        w_returns = tf.convert_to_tensor(w_returns, dtype=tf.float32)
        b_returns = tf.convert_to_tensor(b_returns, dtype=tf.float32)
        w_c_mask = tf.convert_to_tensor(w_c_mask, dtype=tf.float32)
        b_c_mask = tf.convert_to_tensor(b_c_mask, dtype=tf.float32)

        w_model_moves, w_cross_moves, w_model_turns = sp_utils.get_model_inputs_from_uci_games(games, True)
        b_model_moves, b_cross_moves, b_model_turns = sp_utils.get_model_inputs_from_uci_games(games, False)

        c_model_moves = tf.concat([w_model_moves, b_model_moves], axis=0)
        c_cross_moves = tf.concat([w_cross_moves, b_cross_moves], axis=0)
        c_model_turns = tf.concat([w_model_turns, b_model_turns], axis=0)

        c_returns = tf.concat([w_returns, b_returns], axis=0)
        c_returns = tf.expand_dims(c_returns, axis=-1)
        c_mask = tf.concat([w_c_mask, b_c_mask], axis=0)

        for idx in range(self.train_critic_itr):
            c_loss = self.train_critic(c_model_moves, c_cross_moves, c_model_turns, c_returns, c_mask)
        c_loss = c_loss.numpy()
        print('Critic Loss:', c_loss, 'Avg Game Length:', avg_game_len)








    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ])
    def train_critic(
            self,
            input_tensor, cross_tensor, is_white_tensor,
            return_buffer, mask_tensor
    ):
        inputs = [input_tensor, cross_tensor, is_white_tensor]
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            preds, pred_values = self.critic(inputs)  # (batch, seq_len, 2)
            value_loss = self.critic_loss_fn(return_buffer, pred_values, sample_weight=mask_tensor)

        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return value_loss

















if __name__ == '__main__':
    task = Task()
    task.run()

