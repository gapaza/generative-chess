import os
import chess
import chess.pgn
from tqdm import tqdm
import multiprocessing as mp

import config

# mp.set_start_method('fork')


base_dir = '/home/ubuntu/games/lc0'
tar_dir = '/home/ubuntu/games/lc0/tar'
set_dir = '/home/ubuntu/games/lc0/set'

chunk_dir = '/home/ubuntu/games/lc0/chunk'
if not os.path.exists(chunk_dir):
    os.makedirs(chunk_dir)

chunk_size = 100000


# Each directory in set_dir will get its own process
# - each process will sequentially process each file in the directory
# - the output of each processed file is a list of UCI moves (one file has one game)
# - once 100k games are stored, save 100k games to chunk dir
# - continue processing games until all games are processed

def chunk_game_set(params):
    set_path, prefix = params
    print('Starting process:', prefix)
    # Get basename of set_path
    set_name = os.path.basename(set_path)
    chunk_save_dir = os.path.join(chunk_dir, set_name)
    if not os.path.exists(chunk_save_dir):
        os.makedirs(chunk_save_dir)

    game_uci_strings = []
    checkmates = 0
    fen_games = 0

    # List all files in set_path that end with .pgn
    all_files = os.listdir(set_path)
    all_files = [f for f in all_files if f.endswith('.pgn')]
    # print('Num files: ', len(all_files))

    if prefix == 0:
        progress_bar = tqdm(total=len(all_files), desc='Processing files')
    for f in all_files:
        if prefix == 0:
            progress_bar.update(1)
        f_path = os.path.join(set_path, f)
        with open(f_path) as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            fen = game.headers.get("FEN", "")
            if fen != "":
                fen_games += 1
                continue
            uci_moves = list(move.uci() for move in game.mainline_moves())
            if game.end().board().is_checkmate():
                checkmates += 1
                result = game.headers.get("Result", "")
                if result == '1-0':
                    uci_moves.append('[white]')
                elif result == '0-1':
                    uci_moves.append('[black]')
            # print(uci_moves)
            uci_moves_str = ' '.join(uci_moves)
            game_uci_strings.append(uci_moves_str)

        if len(game_uci_strings) >= chunk_size:
            num_files = len(os.listdir(chunk_save_dir))
            chunk_save_path = os.path.join(chunk_save_dir, f'chunk_{prefix}_{num_files}_100k.txt')
            with open(chunk_save_path, 'w') as chunk_file:
                for uci_str in game_uci_strings:
                    chunk_file.write(uci_str + '\n')
            game_uci_strings = []

    print('Process', prefix, 'Checkmates:', checkmates, 'Fen games:', fen_games)
    return


def copy_preprocessed_files():
    source_dir = chunk_dir
    target_dir = os.path.join(config.games_dir, 'lc0')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Recursively gather all .txt files in this directory
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))
    print('Total files:', len(all_files))
    # exit(0)

    # Copy all files to target_dir if they don't already exist
    for f in all_files:
        target_path = os.path.join(target_dir, os.path.basename(f))
        if not os.path.exists(target_path):
            os.system(f'cp {f} {target_path}')



def preprocess_chunks():

    # Copy preprocessed files to games_dir
    # copy_preprocessed_files()
    # exit(0)


    # 1. list all directories in set_dir
    all_dirs = os.listdir(set_dir)


    # Linear processing
    # for d in all_dirs:
    #     set_path = os.path.join(set_dir, d)
    #     chunk_game_set(set_path)

    # Parallel processing
    params = []
    for idx, d in enumerate(all_dirs):
        params.append(
            (os.path.join(set_dir, d), idx)
        )
    # params = params[:1]

    num_workers = 24
    pool = mp.Pool(num_workers)
    pool.map(chunk_game_set, params)
    pool.close()






if __name__ == '__main__':
    preprocess_chunks()

