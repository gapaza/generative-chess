import chess
import chess.engine
import multiprocessing
import random
import chess.pgn
import os
import time

from stockfish.generator import save_pgn

from stockfish.generator.selfplay_v1 import generate_games


STOCKFISH_PATH = '/Users/gapaza/Downloads/stockfish/stockfish-macos-m1-apple-silicon'
SAVE_DIR = '/Users/gapaza/repos/gabe/generative-chess/results'




class GameGenerator:
    """
    The purpose of this class is to generate chess games of stockfish playing against itself.
    Specifically, a high elo stockfish engine will play against a low elo stockfish engine.

    There will be one strong stockfish engine, and multiple versions of weaker ones.

    Strong Engine:
    - 2800 elo stockfish engine

    Weak Engines:
    - 1. random play (no engine)
    - 2. 1350 elo stockfish engine
    - 3. 1800 elo stockfish engine


    There will be a "run" method that will take:
    - the number of games to play
    - the number of processes to use for parallel processing
    - which weak engines to use (randomly selected within method)


    After each game is played, it is recorded as a list of UCI moves, with an additional token at the beginning indicating the color of the strong stockfish engine:
    - [white], d2d4, e7e5, g1f3, b8c6, ...
    - [black], e2e4, e7e5, g1f3, b8c6, ...

    Finally, when all games are played, the results are saved to a text file, where each game is written to a line

    """

    def __init__(self):

        # Weak Engines
        self.engine_1 = None
        self.engine_2 = {'Threads': 1, "Hash": 512, 'UCI_LimitStrength': True, 'UCI_Elo': 1350}
        # self.engine_3 = {'Threads': 1, "Hash": 512, 'UCI_LimitStrength': True, 'UCI_Elo': 1800}
        self.engines = [
            self.engine_1,
            self.engine_2,
            # self.engine_3
        ]



    def run(self, num_games=1000, num_processes=1, engine=0):
        engine = self.engines[engine]



        # 1. Collect multiprocessing call parameters
        num_proc_games = num_games // num_processes
        params = []
        for i in range(num_processes):
            f_name = f'example_games_{i}.txt'
            params.append((num_proc_games, STOCKFISH_PATH, engine, SAVE_DIR, f_name))


        # 2. Submit multiprocessing jobs
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(generate_games, params)





        # generate_games(
        #     num_proc_games,
        #     STOCKFISH_PATH,
        #     self.engine_1,
        #     save_dir=SAVE_DIR,
        #     save_file='example_games.txt'
        # )






        pass












if __name__ == '__main__':
    game_gen = GameGenerator()
    game_gen.run(num_games=1000000, num_processes=10, engine=0)


