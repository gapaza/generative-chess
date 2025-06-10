import config
import chess
import chess.pgn
import os


def save_game_pgn(game_moves, save_dir, file_name='game.pgn'):
    uci_moves = [config.id2token[x] for x in game_moves]
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





def get_inputs_from_game(game_move_ids, white_turn):
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















