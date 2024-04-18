import config
import tensorflow as tf
import numpy as np
import scipy.signal
from stockfish.utils import get_stockfish
from matplotlib import pyplot as plt

from stockfish.rewards.reward_1 import calc_reward


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




class RewardShaping:

    def __init__(self):
        self.engine = get_stockfish()

        self.gamma = 0.99
        self.lam = 0.95

    def get_even_mask(self, items):
        mask = [x for x in range(len(items))]
        mask = [x % 2 for x in mask]
        return mask

    def get_odd_mask(self, items):
        mask = [x for x in range(len(items))]
        mask = [x % 2 for x in mask]
        mask = [1 - x for x in mask]
        return mask


    def game_reward_viz(self, uci_moves):
        rewards, eval_history = calc_reward(self.engine, uci_moves)

        print('Rewards:', np.mean(rewards), rewards)
        returns = discounted_cumulative_sums(
            rewards, self.gamma
        )  # [:-1]
        print('Returns:', np.mean(returns), returns)




        white_mask = self.get_odd_mask(rewards)   # This masks out black moves
        black_mask = self.get_even_mask(rewards)  # This masks out white moves

        white_rewards = [reward for idx, reward in enumerate(rewards) if white_mask[idx] == 1]
        black_rewards = [reward for idx, reward in enumerate(rewards) if black_mask[idx] == 1]

        white_returns = discounted_cumulative_sums(
            white_rewards, self.gamma
        )  # [:-1]
        black_returns = discounted_cumulative_sums(
            black_rewards, self.gamma
        )  # [:-1]

        print('White Rewards:', np.mean(white_rewards), white_rewards)
        print('White Returns:', np.mean(white_returns), white_returns)

        print('Black Rewards:', np.mean(black_rewards), black_rewards)
        print('Black Returns:', np.mean(black_returns), black_returns)




        # Plot rewards in one line, and eval history on another
        scaled_eval_history = [x / 100 for x in eval_history]
        plt.plot([0] + rewards, label='Rewards')
        plt.plot(scaled_eval_history, label='Eval History')
        plt.legend()
        plt.savefig('rewards.png')
































if __name__ == '__main__':
    uci_moves = """


    b1c3 b7b6 e2e4 c8b7 f2f4 e7e6 d2d3 d7d6 g1f3 b8d7 g2g3 c7c5 f1g2 g8f6 e1g1 f8e7 c1e3 e8g8 h2h3 a7a6 g3g4 h7h6 d1e1 b6b5 a1d1 a8c8 g4g5 h6g5 f3g5 b5b4 c3e2 f6e8 e1g3 g7g6 e3c1 e8g7 g3e3 a6a5 b2b3 e7f6 c1d2 f8e8 d2e1 d8c7 d3d4 c5d4 e3d4 f6d4 e2d4 c7b6 e1f2 b6a6 h3h4 g7h5 d4b5 c8c2 b5c7 a6e2 c7e8 h5f4 e8f6 d7f6 g5f3 f6e4 f3d4 e2g4 d4c2 g4g2 [black]


    """
    uci_moves = uci_moves.strip()


    client = RewardShaping()
    client.game_reward_viz(uci_moves)

