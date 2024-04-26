import config
import tensorflow as tf
import numpy as np
import scipy.signal
from stockfish.utils import get_stockfish, combine_alternate
from matplotlib import pyplot as plt
from matplotlib import gridspec

# from stockfish.rewards.reward_1 import calc_reward
from stockfish.rewards.reward_2 import calc_reward


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




class RewardShaping:

    def __init__(self):
        self.engine = get_stockfish()
        # self.nodes = 200000
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





from stockfish.utils import pgn_to_uci


if __name__ == '__main__':
    uci_moves = """

e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7 e1g1 e8g8 f1e1 c8e6 e2f3 b8d7 a2a4 a8c8 b3d2 d8c7 d2f1 d7b6 f1e3 f8d8 a4a5 b6c4 e3c4 c7c4 c1g5 h7h6 g5f6 e7f6 c3d5 e6d5 e4d5 c8c5 c2c3 c5b5 f3e2 c4c5 e2b5 c5b5 b2b4 d8c8 e1e3 f6g5 e3f3 g7g6 h2h4 g5h4 g2g3 h4g5 g1g2 g8g7 d1d3 b5d3 f3d3 c8c4 a1b1 f7f5 b1b3 g7f6 b4b5 a6b5 b3b5 e5e4 d3d4 c4d4 c3d4 g5d2 a5a6 b7a6 b5a6                                                 

    """
    uci_moves = uci_moves.strip()


    # pgn_text = """
    #
    # 1. d4 c5 2. e3 e6 3. Nf3 cxd4 4. exd4 Nf6 5. Nc3 d6 6. a3 *
    #
    # """
    # uci_moves = pgn_to_uci(pgn_text)







    client = RewardShaping()

    # print(client.get_even_mask([0 for _ in range(10)]))
    # exit(0)

    client.game_reward_viz(uci_moves, nodes=50000)


    # client.game_reward_viz(uci_moves, save_file='rewards_10k.png', nodes=10000)
    # client.game_reward_viz(uci_moves, save_file='rewards_50k.png', nodes=50000)
    # client.game_reward_viz(uci_moves, save_file='rewards_100k.png', nodes=100000)
    # client.game_reward_viz(uci_moves, save_file='rewards_200k.png', nodes=200000)
    # client.game_reward_viz(uci_moves, save_file='rewards_500k.png', nodes=500000)

