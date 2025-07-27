import config
import chess
import chess.pgn
import os
import random


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




def get_inputs_from_game_a3(game_move_ids, white_turn):
    game = [x for x in game_move_ids if config.id2token[x] not in config.non_move_tokens]


    if white_turn is True:
        self_attn = [config.token2id['[start]']]   # white moves from white perspective
        cross_attn = [config.token2id['[black]']]  # black moves from white perspective
    else:
        self_attn = [config.token2id['[start]']]   # black moves from black perspective
        cross_attn = []  # white moves from black perspective

    for idx, obs in enumerate(game):
        if idx % 2 == 0:
            if white_turn is True:
                self_attn.append(obs)
            else:
                cross_attn.append(obs)
        else:
            if white_turn is True:
                cross_attn.append(obs)
            else:
                self_attn.append(obs)

    inf_idx = len(self_attn) - 1
    return self_attn, cross_attn, inf_idx


def get_engine_move(game_move_ids, engine, elo=1200):
    game = [x for x in game_move_ids if config.id2token[x] not in config.non_move_tokens]

    board = chess.Board()

    for idx, obs in enumerate(game):
        move = chess.Move.from_uci(config.id2token[obs])
        if move not in board.legal_moves:
            return None
        board.push(move)



    # --- ENGINE MOVE
    result = engine.play(board, chess.engine.Limit(time=0.002))
    if result.move is None:
        # play a random legal move if no best move is found
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        result.move = random.choice(legal_moves)

        uci = str(result.move.uci())
        print('Random move selected:', uci)
    else:
        uci = str(result.move.uci())
        # print('Engine move selected:', uci)

    # # --- RANDOM MOVE
    # legal_moves = list(board.legal_moves)
    # if not legal_moves:
    #     return None
    # uci = str(random.choice(legal_moves).uci())




    return uci







