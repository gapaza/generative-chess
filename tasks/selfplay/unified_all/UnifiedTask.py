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
from tasks.selfplay.unified_all.UnifiedActorWrapper import ActorWrapper
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
global_mini_batch_size = 64
run_evals = True
alternate_colors = False
run_dir = 16
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
        self.a1_evals = Evals(themes=self.eval_themes)

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
        self.init_analysis = init_analysis

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
        self.actor_2_wrapper = ActorWrapper(self.actor_1, self.model_1_optimizer, self.critic, global_mini_batch_size, self, 2)

    def run_evals(self):
        if run_evals is False:
            return

        # Model 1 evaluations
        eval_history = self.a1_evals.run_eval(self.actor_1)
        eval_history['step_interval'] = global_mini_batch_size
        save_name = 'unified_evals_' + str(run_num) + '.png'
        plot_training_comparison([eval_history], bounds=False, save_name=save_name, local_save_dir=self.run_dir)

    def save_models(self):
        t_actor_save_path = os.path.join(self.pretrain_save_dir, 'actor_weights_' + str(self.curr_epoch + pickup_epoch))
        self.actor_1.save_weights(t_actor_save_path)


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
        analysis = [self.init_analysis for x in range(self.mini_batch_size)]
        games_done = [False for x in range(self.mini_batch_size)]

        for move_idx in range(self.max_steps_per_game):
            if move_idx % 2 == 0:  # even number
                actor_wrapper = white_model
                white_turn = True
            else:  # odd number
                actor_wrapper = black_model
                white_turn = False

            # Get actions
            actions, games_done, scores, boards = actor_wrapper.get_actions(games, white_turn, games_done, scores, boards, analysis)
            for idx, action in enumerate(actions):
                if action is not None:
                    games[idx].append(action)

        avg_game_len = np.mean([len(game) for game in games])
        if self.curr_epoch % 10 == 0:
            game_file_name = 'game_' + str(self.curr_epoch) + '.pgn'
            sp_utils.save_game_pgn(games[0], self.games_dir, file_name=game_file_name)


        white_model.calc_advantages(games, True)
        black_model.calc_advantages(games, False)

        # -----------------------
        # Combined Actor Critic
        # -----------------------

        # # Each call returns: kl, entr, loss, policy_loss
        # white_model.train_ppo_unified(games, True, black_model)
        #
        # if self.curr_epoch % plot_freq == 0 and self.curr_epoch > 0:
        #     white_model.plot_model()
        #
        # print('Avg Game Length:', avg_game_len)

        # -----------------------
        # Separate Actor Critic
        # -----------------------

        white_model.train_ppo(games, True, black_model)

        white_model_moves, white_cross_moves, white_model_turns = sp_utils.get_model_inputs_from_uci_games(games, True)
        white_returns, white_mask_tensor = white_model.get_critic_info()
        white_returns = tf.convert_to_tensor(white_returns, dtype=tf.float32)
        white_returns = tf.expand_dims(white_returns, axis=-1)
        white_mask_tensor = tf.convert_to_tensor(white_mask_tensor, dtype=tf.float32)

        black_model_moves, black_cross_moves, black_model_turns = sp_utils.get_model_inputs_from_uci_games(games, False)
        black_returns, black_mask_tensor = black_model.get_critic_info()
        black_returns = tf.convert_to_tensor(black_returns, dtype=tf.float32)
        black_returns = tf.expand_dims(black_returns, axis=-1)
        black_mask_tensor = tf.convert_to_tensor(black_mask_tensor, dtype=tf.float32)

        # Concatenate critic inputs
        all_model_moves = tf.concat([white_model_moves, black_model_moves], axis=0)
        all_cross_moves = tf.concat([white_cross_moves, black_cross_moves], axis=0)
        all_model_turns = tf.concat([white_model_turns, black_model_turns], axis=0)
        all_returns = tf.concat([white_returns, black_returns], axis=0)
        all_mask_tensor = tf.concat([white_mask_tensor, black_mask_tensor], axis=0)

        # Train critic
        for x in range(self.train_critic_itr):
            value_loss = self.train_critic(all_model_moves, all_cross_moves, all_model_turns, all_returns, all_mask_tensor)
        print('Avg game length:', avg_game_len, 'Value Loss:', value_loss.numpy())


        if self.curr_epoch % plot_freq == 0 and self.curr_epoch > 0:
            white_model.plot_model()
            black_model.plot_model()






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

