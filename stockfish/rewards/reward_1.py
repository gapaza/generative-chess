import config
import chess
import chess.engine
from multiprocessing import Process, Queue
from stockfish.utils import get_stockfish
# Game is a string of UCI moves


# Rewards
correct_pred_win = 0.05
incorrect_pred_win = -0.1


def calc_reward(engine, game, n=100000, info=False):
    uci_moves = game.split(' ')
    rewards = []  # Always from white's perspective
    pad_len = config.seq_length - 1
    eval_history = [0.2]
    eval_type = ['eval']  # eval or mtm (moves until mate)

    board = chess.Board()
    for uci_move in uci_moves:
        white_turn = (board.turn == chess.WHITE)
        move_reward = 0

        # Check if checkmate
        if board.is_checkmate():
            # print('Turn:', white_turn, uci_move)
            if white_turn is True and uci_move == '[black]':
                move_reward = 0.05  # Correctly predicted white won
            elif not white_turn and uci_move == '[white]':
                move_reward = 0.05  # Correctly predicted white won
            else:
                move_reward = -0.1  # Incorrect prediction of winning side
            rewards.append(move_reward)
            break

        # Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            if uci_move == '[draw]':
                move_reward = 0.05
            else:
                move_reward = -0.1
            rewards.append(move_reward)
            break

        # Check if illegal
        if uci_move in config.non_move_tokens:
            move_reward = -0.1
            rewards.append(move_reward)
            break
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            move_reward = -0.1
            rewards.append(move_reward)
            break

        # -------------------------------------
        # Push move
        # -------------------------------------
        board.push_uci(uci_move)

        # Check if checkmating move
        if board.is_checkmate():
            move_reward = 0.05
            rewards.append(move_reward)
            continue

        # Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            move_reward = -0.05
            rewards.append(move_reward)
            continue

        # -------------------------------------
        # Engine-based reward
        # -------------------------------------
        nodes = n
        analysis = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=1)
        top_line = analysis[0]
        top_line_score = top_line["score"]

        # If forced mate in 'n'
        if top_line_score.is_mate():
            if white_turn is True:
                moves_to_mate = top_line_score.white().mate()
            else:
                moves_to_mate = top_line_score.black().mate()

            # Restrict moves_to_mate to 10
            if moves_to_mate > 10:
                moves_to_mate = 10
            elif moves_to_mate < -10:
                moves_to_mate = -10

            if eval_type[-1] == 'mtm':
                move_reward = (eval_history[-1] - moves_to_mate) / 15.0
            else:
                move_reward = 0.03


            eval_history.append(moves_to_mate)
            eval_type.append('mtm')
            rewards.append(move_reward)
            continue

        # If not forced mate, use eval
        prev_eval = eval_history[-1]
        new_eval = top_line_score.white().score() / 100.0  # Convert centipawns to pawns
        eval_norm = 20.0
        reward_center = 0.03
        if eval_type[-1] == 'mtm':
            # For prev_eval, use the last value of eval_history that is of type eval in eval_type
            prev_eval = 0.0


        # # Pure positional evaluation
        # eval_history.append(new_eval)
        # eval_type.append('eval')
        # move_reward = new_eval / 10.0
        # rewards.append(move_reward)


        # Difference in eval w.r.t. previous eval
        reward_diff = abs(new_eval - prev_eval) / eval_norm
        if white_turn is True:
            if prev_eval > new_eval:  # Position worsened for white
                move_reward = reward_center - reward_diff
            else:
                move_reward = reward_center + reward_diff
        else:
            if prev_eval < new_eval:  # Position worsened for black
                move_reward = reward_center - reward_diff
            else:
                move_reward = reward_center + reward_diff  # Position improved for black
        eval_history.append(new_eval)
        eval_type.append('eval')
        rewards.append(move_reward)



    # Pad rewards to seq_length
    if len(rewards) < pad_len:
        rewards += ([0.0] * (pad_len - len(rewards)))

    if info is True:
        return_info = {
            'eval_history': eval_history,
        }
        return rewards, return_info
    else:
        return rewards





def calc_reward_batch(games_uci, engine, n=100000):
    games = []
    for uci_game in games_uci:
        game = [config.id2token[token] for token in uci_game]
        game = ' '.join(game)
        games.append(game)
    rewards = []
    for game in games:
        rewards.append(calc_reward(engine, game, n=n))
    return rewards









