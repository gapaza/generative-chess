import config
import scipy.signal
import chess.pgn
import tensorflow as tf
import os




def save_game_pgn(game_moves, save_dir, file_name='game.pgn'):
    uci_moves = game_moves
    uci_moves_clipped = [x for x in uci_moves if x not in config.non_move_tokens]

    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for idx, move in enumerate(uci_moves_clipped):
        # print('GAME MOVE:', move)
        move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            board.push(move)
            node = node.add_variation(move)
        else:
            break

    game.headers["Event"] = "Example Game"
    game.headers["Site"] = "Internet"
    game.headers["Date"] = "2024.04.13"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player1"
    game.headers["Black"] = "Player2"
    game.headers["Result"] = "*"  # Or use board.result() if the game is over

    pgn_path = os.path.join(save_dir, file_name)
    with open(pgn_path, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    # Append UCI move sequence to the end of the PGN file
    with open(pgn_path, 'a') as pgn_file:
        pgn_file.write('\n\n')
        pgn_file.write(' '.join(uci_moves))
        pgn_file.write('\n\n')





def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]





def pad_list(sequence, pad_token=0, pad_len=None):
    if pad_len is None:
        pad_len = config.seq_length
    while len(sequence) < pad_len:
        sequence.append(pad_token)
    if len(sequence) > pad_len:
        sequence = sequence[:pad_len]
    return sequence



def id2token(ids):
    return [config.id2token[x] for x in ids]

def token2id(tokens):
    return [config.token2id[x] for x in tokens]




def get_model_inputs_from_uci_games(games_move_ids, white_turns):
    model_moves = []
    cross_moves = []
    model_turns = []
    for game_move_ids in games_move_ids:
        mm, cm, inf_idx = get_inputs_from_game(game_move_ids, white_turns, convert_uci=True)
        mm = pad_list(mm, pad_len=config.seq_length)
        cm = pad_list(cm, pad_len=config.seq_length)
        model_moves.append(mm)
        cross_moves.append(cm)
        model_turns.append(white_turns)
    model_moves = tf.convert_to_tensor(model_moves, dtype=tf.int32)
    cross_moves = tf.convert_to_tensor(cross_moves, dtype=tf.int32)
    model_turns = tf.convert_to_tensor(model_turns, dtype=tf.bool)
    return model_moves, cross_moves, model_turns


def get_inputs_from_game(game_move_ids, white_turn, convert_uci=False):
    if convert_uci:
        game_move_ids = [config.token2id[x] for x in game_move_ids]
    game = [x for x in game_move_ids if config.id2token[x] not in config.non_move_tokens]
    white_moves = [config.token2id['[white]']]
    black_moves = [config.token2id['[black]']]
    for idx, obs in enumerate(game):
        if idx % 2 == 0:
            white_moves.append(obs)
        else:
            black_moves.append(obs)

    if white_turn is True:
        model_moves = white_moves
        cross_moves = black_moves
        inf_idx = len(white_moves) - 1
    else:
        model_moves = black_moves
        cross_moves = white_moves
        inf_idx = len(black_moves) - 1

    return model_moves, cross_moves, inf_idx



