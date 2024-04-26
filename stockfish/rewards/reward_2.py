import config
import chess
import chess.engine
from multiprocessing import Process, Queue
from stockfish.utils import get_stockfish
# Game is a string of UCI moves

# Rewards
# correct_pred_win = 0.1
# incorrect_pred_win = -0.1
# correct_pred_draw = 0.1
# incorrect_pred_draw = -0.1
# illegal_move = -0.0  # was -0.1
# checkmate_reward = 0.0  # was 0.3
# draw_reward = 0.0  # was -0.1
# move_bonus = 0.02
# eval_lb = -1
# eval_ub = 1
# mate_steps_ub = 10

# Rewards
correct_pred_win = 0.1
incorrect_pred_win = -0.1
correct_pred_draw = 0.1
incorrect_pred_draw = -0.1
illegal_move = -1.0  # was -0.1
checkmate_reward = 100.0  # was 0.3
draw_reward = -0.1  # was -0.1
move_bonus = 0.02
eval_lb = -1
eval_ub = 1
mate_steps_ub = 10


def clip_eval(eval):
    eval = eval / 1000.0
    if eval < eval_lb:
        return eval_lb
    elif eval > eval_ub:
        return eval_ub
    else:
        return eval

def clip_mate_steps(mate):
    if mate > mate_steps_ub:
        return mate_steps_ub
    elif mate < -mate_steps_ub:
        return -mate_steps_ub
    else:
        return mate


def get_moves_to_mate_eval(moves_to_mate):
    moves_to_mate = clip_mate_steps(moves_to_mate)
    if moves_to_mate < 0:
        eval = -1 - ((10-abs(moves_to_mate)) * 0.01)
        return eval
    else:
        eval = 1 + ((10-abs(moves_to_mate)) * 0.01)
        return eval




def calc_reward(engine, game, n=100000, info=False):
    uci_moves = game.split(' ')
    rewards = []  # Always from white's perspective
    engine_eval_history = []
    trans_eval_history = []
    pad_len = config.seq_length - 1


    # Rewards:
    # 1. positional eval after making move minus positional eval before making move
    board = chess.Board()
    analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
    prev_score = analysis[0]["score"]
    # engine_eval_history.append(prev_score.white().score())

    for uci_move in uci_moves:
        white_turn = (board.turn == chess.WHITE)
        color_turn = 'white'
        if board.turn == chess.BLACK:
            color_turn = 'black'


        if board.is_checkmate():
            if white_turn is True and uci_move == '[black]':
                move_reward = correct_pred_win  # Correctly predicted white won
            elif not white_turn and uci_move == '[white]':
                move_reward = correct_pred_win  # Correctly predicted white won
            else:
                move_reward = incorrect_pred_win  # Incorrect prediction of winning side
            rewards.append(move_reward)
            break

        # Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            if uci_move == '[draw]':
                move_reward = correct_pred_draw
            else:
                move_reward = incorrect_pred_draw
            rewards.append(move_reward)
            break

        # Check if illegal
        if uci_move in config.non_move_tokens:
            move_reward = illegal_move
            rewards.append(move_reward)
            break
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            move_reward = illegal_move
            rewards.append(move_reward)
            break

        # -------------------------------------
        # Push move
        # -------------------------------------
        board.push(move)

        # Check if checkmating move
        if board.is_checkmate():
            move_reward = checkmate_reward
            rewards.append(move_reward)
            continue

        # Check if draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            move_reward = draw_reward
            rewards.append(move_reward)
            continue

        # -------------------------------------
        # Engine-based reward
        # -------------------------------------
        # rewards.append(0.0)
        # continue
        analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
        top_line = analysis[0]
        top_line_score = top_line["score"]

        # Cases
        # 1. forced_mate --> forced_mate
        # 2. forced_mate --> normal
        # 3. normal --> forced_mate
        # 4. normal --> normal
        if prev_score.is_mate() and top_line_score.is_mate():
            prev_moves_to_mate = prev_score.white().mate()
            moves_to_mate = top_line_score.white().mate()
            if color_turn == 'black':
                prev_moves_to_mate = prev_score.black().mate()
                moves_to_mate = top_line_score.black().mate()
            prev_eval = get_moves_to_mate_eval(prev_moves_to_mate)
            eval = get_moves_to_mate_eval(moves_to_mate)
            move_reward = eval - prev_eval
        elif prev_score.is_mate() and not top_line_score.is_mate():
            prev_moves_to_mate = prev_score.white().mate()
            eval = top_line_score.white().score()
            if color_turn == 'black':
                prev_moves_to_mate = prev_score.black().mate()
                eval = top_line_score.black().score()
            prev_eval = get_moves_to_mate_eval(prev_moves_to_mate)
            eval = clip_eval(eval)
            move_reward = eval - prev_eval
        elif not prev_score.is_mate() and top_line_score.is_mate():
            prev_eval = prev_score.white().score()
            moves_to_mate = top_line_score.white().mate()
            if color_turn == 'black':
                prev_eval = prev_score.black().score()
                moves_to_mate = top_line_score.black().mate()
            prev_eval = clip_eval(prev_eval)
            eval = get_moves_to_mate_eval(moves_to_mate)
            move_reward = eval - prev_eval
        else:
            prev_eval = prev_score.white().score()
            eval = top_line_score.white().score()
            if color_turn == 'black':
                prev_eval = prev_score.black().score()
                eval = top_line_score.black().score()
            prev_eval = clip_eval(prev_eval)
            eval = clip_eval(eval)
            move_reward = eval - prev_eval


        move_reward += move_bonus
        rewards.append(move_reward)
        prev_score = top_line_score
        trans_eval_history.append(eval)
        if top_line_score.white().score() is not None:
            engine_eval_history.append(top_line_score.white().score() / 100.0)
        else:
            engine_eval_history.append(0.0)

    if info is True:
        info = {}
        info['engine_eval_history'] = engine_eval_history
        info['trans_eval_history'] = trans_eval_history
        return rewards, info
    else:
        if len(rewards) < pad_len:
            rewards += ([0.0] * (pad_len - len(rewards)))
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









