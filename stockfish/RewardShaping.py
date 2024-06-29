import config
import tensorflow as tf
import numpy as np
import scipy.signal
from utils import save_game_pgn
from stockfish.utils import get_stockfish, combine_alternate
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os

# from stockfish.rewards.reward_1 import calc_reward
from stockfish.rewards.reward_2 import calc_reward, calc_reward_slice


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




class RewardShaping:

    def __init__(self):
        self.engine = get_stockfish()
        # self.nodes = 200000
        self.gamma = 0.99
        self.lam = 0.95

        self.analysis_dir = os.path.join(config.root_dir, 'stockfish', 'analysis')
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def get_even_mask(self, items):
        mask = [x for x in range(len(items))]
        mask = [x % 2 for x in mask]
        return mask

    def get_odd_mask(self, items):
        mask = [x for x in range(len(items))]
        mask = [x % 2 for x in mask]
        mask = [1 - x for x in mask]
        return mask

    def game_reward_slice(self, uci_moves , nodes=200000):
        rewards, sample_weights = calc_reward_slice(self.engine, uci_moves, [6, 8], n=nodes)
        print(rewards)
        print(sample_weights)
    def game_reward_viz(self, uci_moves, save_file='rewards.png', nodes=200000):
        rewards, info = calc_reward(self.engine, uci_moves, n=nodes, info=True)
        engine_eval_history = info['engine_eval_history']
        trans_eval_history = info['trans_eval_history']
        # eval_history = rewards


        print('Rewards:', np.mean(rewards), rewards)
        returns = discounted_cumulative_sums(
            rewards, self.gamma
        )  # [:-1]
        print('Returns:', np.mean(returns), returns.tolist())


        white_mask = self.get_odd_mask(rewards)   # This masks out black moves
        black_mask = self.get_even_mask(rewards)  # This masks out white moves

        white_rewards = [reward for idx, reward in enumerate(rewards) if white_mask[idx] == 1]
        black_rewards = [reward for idx, reward in enumerate(rewards) if black_mask[idx] == 1]

        print('\nWhite Rewards:', np.mean(white_rewards), white_rewards)
        print('Black Rewards:', np.mean(black_rewards), black_rewards)

        white_returns = discounted_cumulative_sums(
            white_rewards, self.gamma
        )  # [:-1]
        black_returns = discounted_cumulative_sums(
            black_rewards, self.gamma
        )  # [:-1]

        print('\nWhite Returns:', np.mean(white_returns), white_returns.tolist())
        print('Black Returns:', np.mean(black_returns), black_returns.tolist())

        combined_returns = combine_alternate(white_returns.tolist(), black_returns.tolist())
        print('Combined Returns:', np.mean(combined_returns), combined_returns)


        gs = gridspec.GridSpec(2, 3)
        fig = plt.figure(figsize=(10, 5))


        # Plot rewards in one line, and eval history on another
        plt.subplot(gs[0, 0])
        plt.plot(engine_eval_history)
        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('Engine Evaluation')

        plt.subplot(gs[0, 1])
        plt.plot(rewards, label='Rewards')
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('Rewards')

        plt.subplot(gs[0, 2])
        plt.plot(returns)
        plt.xlabel('Moves')
        plt.ylabel('Returns')
        plt.title('Critic Returns')

        # White rewards
        plt.subplot(gs[1, 0])
        plt.plot(white_rewards)
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('White Rewards')

        # Black rewards
        plt.subplot(gs[1, 1])
        plt.plot(black_rewards)
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('Black Rewards')







        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()

        self.engine.close()

    def reward_analysis(self, uci_moves, nodes=50000):
        rewards, info = calc_reward(self.engine, uci_moves, n=nodes, info=True)
        engine_eval_history = info['engine_eval_history']
        transformed_eval_history = info['transformed_eval_history']
        eval_type_history = info['eval_type_history']
        t_eval_history_white = info['t_eval_history_white']
        t_eval_history_black = info['t_eval_history_black']

        self.plot_combined(engine_eval_history, transformed_eval_history, eval_type_history, rewards)
        self.plot_colors(engine_eval_history, transformed_eval_history, eval_type_history, rewards, t_eval_history_white, t_eval_history_black)



    def split_into_colors(self, elements):
        white_elements = []
        black_elements = []
        for idx, element in enumerate(elements):
            if idx % 2 == 0:
                white_elements.append(element)
            else:
                black_elements.append(element)
        return white_elements, black_elements

    def get_plot_split(self, elements):
        white_elements = []
        black_elements = []
        for idx, element in enumerate(elements):
            if idx % 2 == 0:
                white_elements.append([idx, element])
            else:
                black_elements.append([idx, element])
        return white_elements, black_elements


    def plot_colors(self, engine_eval_history, transformed_eval_history, eval_type_history, rewards, t_eval_history_white, t_eval_history_black):
        white_rewards, black_rewards = self.split_into_colors(rewards)
        white_engine_eval_history, black_engine_eval_history = self.split_into_colors(engine_eval_history)
        white_transformed_eval_history, black_transformed_eval_history = self.split_into_colors(transformed_eval_history)
        white_eval_type_history, black_eval_type_history = self.split_into_colors(eval_type_history)

        # Plot white
        # self.plot_combined(white_engine_eval_history, white_transformed_eval_history, white_eval_type_history, white_rewards, title='White Plots', file_name='white_plots.png')
        # self.plot_combined(black_engine_eval_history, black_transformed_eval_history, black_eval_type_history, black_rewards, title='Black Plots', file_name='black_plots.png')

        gs = gridspec.GridSpec(2, 3)  # White is top row, black is bottom row
        fig = plt.figure(figsize=(10, 5))
        fit = plt.suptitle('Color Comparison')
        category_colors = {'forced_mate': 'orange', 'nominal': 'green', 'draw': 'blue', 'checkmate': 'red'}



        plt.subplot(gs[0, 0])
        plt.plot(engine_eval_history)
        for idx, eval_type in enumerate(eval_type_history):
            plt.scatter(idx, engine_eval_history[idx], color=category_colors[eval_type], s=10)

        # plot t_eval_history_white
        plt.plot(t_eval_history_white)

        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('White Engine Evaluation')

        plt.subplot(gs[0, 1])
        plt.plot(white_rewards, label='Rewards')
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('White Rewards')

        plt.subplot(gs[0, 2])
        plt.plot(white_transformed_eval_history)
        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('White Transformed Evaluation')


        # Black plots
        plt.subplot(gs[1, 0])
        plt.plot(black_engine_eval_history)
        for idx, eval_type in enumerate(black_eval_type_history):
            plt.scatter(idx, black_engine_eval_history[idx], color=category_colors[eval_type], s=15)
        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('Black Engine Evaluation')

        plt.subplot(gs[1, 1])
        plt.plot(black_rewards, label='Rewards')
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('Black Rewards')

        plt.subplot(gs[1, 2])
        plt.plot(black_transformed_eval_history)
        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('Black Transformed Evaluation')

        file_name = 'color_comparison.png'
        plt.tight_layout()
        save_file = os.path.join(self.analysis_dir, file_name)
        plt.savefig(save_file)
        plt.close()
        










    def plot_combined(self, engine_eval_history, transformed_eval_history, eval_type_history, rewards, title='Combined Plots', file_name='combined_plots.png'):
        gs = gridspec.GridSpec(1, 3)
        fig = plt.figure(figsize=(10, 3))
        fit = plt.suptitle(title)

        category_colors = {'forced_mate': 'orange', 'nominal': 'green', 'draw': 'blue', 'checkmate': 'red'}

        # Plot rewards in one line, and eval history on another
        plt.subplot(gs[0, 0])
        plt.plot(engine_eval_history)

        # Now plot rewards as dots, with different colors for different categories
        for idx, eval_type in enumerate(eval_type_history):
            plt.scatter(idx, engine_eval_history[idx], color=category_colors[eval_type], s=15)


        white_reward_pts, black_reward_pts = self.get_plot_split(rewards)
        plt.plot([r[0] for r in white_reward_pts], [r[1] for r in white_reward_pts], 'ro')
        plt.plot([r[0] for r in black_reward_pts], [r[1] for r in black_reward_pts], 'bo')

        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('Engine Evaluation')

        plt.subplot(gs[0, 1])
        plt.plot(rewards, label='Rewards')
        plt.xlabel('Moves')
        plt.ylabel('Rewards')
        plt.title('Rewards')

        plt.subplot(gs[0, 2])
        plt.plot(transformed_eval_history)
        plt.xlabel('Moves')
        plt.ylabel('Eval')
        plt.title('Transformed Evaluation')

        plt.tight_layout()
        save_file = os.path.join(self.analysis_dir, file_name)
        plt.savefig(save_file)
        plt.close()


















if __name__ == '__main__':
    uci_moves = """

e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 e1g1 f8g7 d2d4 c5d4 c2c3 d4c3 b1c3 a7a6 b5a4 b7b5 a4b3 g8f6 e4e5 f6g8 c1f4 e7e6 c3e4 d7d5 e5d6 g8f6 e4f6 g7f6 a1c1 c8b7 f1e1 e8g8 d1d2 d8d7 f3e5 c6e5 f4e5 f6e5 e1e5 a8c8 e5c5 c8d8 c5c7 d7d6 d2d6 d8d6 c7b7 d6d2 h2h3 f8d8 c1c7 d8f8 c7f7 f8f7 b7f7 g8f7 g1f1 d2b2 f1e1 a6a5 a2a4 b5b4 b3c4 b2c2 c4b5 b4b3 e1d1 b3b2 d1c2 b2b1q c2d2 b1e4 f2f3 e4f4 d2e2 f4d2 e2f1 d2f4 f1g1 f7f8 g1f2 h7h5 f2e2 h5h4 b5d7 f8e7 d7b5 e6e5 e2f2 e7d6 f2e2 d6c5 e2d3 c5b4 b5e8 b4b3 e8c6 f4d4 d3e2 b3a4

    """



    # uci_moves = """
    #
    # # e2e4 g7g5 b1c3 f7f5 f2f4
    #
    # """
    uci_moves = uci_moves.strip()

    moves_ids = [config.token2id[move] for move in uci_moves.split()]
    save_game_pgn(moves_ids, config.root_dir)


    # pgn_text = """
    #
    # 1. d4 c5 2. e3 e6 3. Nf3 cxd4 4. exd4 Nf6 5. Nc3 d6 6. a3 *
    #
    # """
    # uci_moves = pgn_to_uci(pgn_text)







    client = RewardShaping()
    client.reward_analysis(uci_moves, nodes=50000)


    client.engine.close()


