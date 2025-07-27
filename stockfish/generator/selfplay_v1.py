import chess
import chess.engine
import multiprocessing
import random
import chess.pgn
import os
import time
from tqdm import tqdm

from stockfish.generator import save_pgn

DEBUG = False
SAVE_PGN = False
THINK_TIME = 0.0025  # seconds, time each engine has to think per move

def generate_games(n, engine_path, e2_conf, save_dir, save_file):
    # e1 is the strong engine which the model will mimic

    # Strong Engine
    e1_conf = {'Threads': 1, "Hash": 512,}
    strong_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    strong_engine.configure(e1_conf)

    # Weak Engine
    weak_engine = None if e2_conf is None else chess.engine.SimpleEngine.popen_uci(engine_path)
    if weak_engine:
        weak_engine.configure(e2_conf)

    # Prep save file
    if save_file is None:
        save_file = f'{n}_games.txt'
    games_file = os.path.join(save_dir, save_file)


    # Play n games
    games = []
    for i in range(n):
        last_iter = i == n - 1
        if DEBUG is True:
            print(f'Playing game {i + 1}/{n}')
        game_moves = play_game(strong_engine, weak_engine, save=SAVE_PGN and last_iter)
        games.append(game_moves)

        # Open the file in append mode and write the game moves
        with open(games_file, 'a') as f:
            f.write(' '.join(game_moves) + '\n')

    # Close engines
    strong_engine.quit()
    if weak_engine:
        weak_engine.quit()





def play_game(strong_engine, weak_engine=None, save=False):
    board = chess.Board()
    game_moves = []

    # Randomly assign strong engine color
    strong_color = random.choice(['white', 'black'])
    # strong_color = 'white'
    game_moves.append(f'[{strong_color}]')

    # Play the game
    start_time = time.time()
    while not board.is_game_over():
        if board.turn == (strong_color == 'white'):
            move = strong_engine.play(
                board,
                chess.engine.Limit(time=THINK_TIME)
                # chess.engine.Limit(nodes=100000)
            ).move
        else:
            if weak_engine:
                move = weak_engine.play(
                    board,
                    chess.engine.Limit(time=THINK_TIME)
                ).move
            else:
                move = random.choice(list(board.legal_moves))

        game_moves.append(move.uci())
        board.push(move)
    # if DEBUG is True:
    #     print(f'Game finished in {time.time() - start_time:.2f} seconds')

    # Save the game in PGN format
    if save is True:
        save_pgn(game_moves[1:])  # Exclude the color token

    return game_moves










if __name__ == '__main__':
    engine_path = '/Users/gapaza/Downloads/stockfish/stockfish-macos-m1-apple-silicon'

    # e2_conf_ex = {
    #     'Threads': 1, "Hash": 512,
    #     'UCI_LimitStrength': True, 'UCI_Elo': 1350
    # }
    e2_conf_ex = None

    # Generate 10 games
    generate_games(
        10,
        engine_path,
        e2_conf_ex,
        save_dir='/Users/gapaza/repos/gabe/generative-chess/results',
        save_file='example_games.txt'
    )




