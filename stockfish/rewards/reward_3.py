import config
import chess
import chess.engine
from multiprocessing import Process, Queue
from stockfish.utils import get_stockfish
import numpy as np
import time
# Game is a string of UCI moves

# Rewards (old)
# correct_pred_win = 0.1
# incorrect_pred_win = -0.1
# correct_pred_draw = 0.1
# incorrect_pred_draw = -0.1
# illegal_move = -0.0  # was -0.1
# checkmate_reward = 0.0  # was 0.3
# draw_reward = 0.0  # was -0.1
# move_bonus = 0.02

# Rewards (old)
# correct_pred_win = 0.1
# incorrect_pred_win = -0.1
# correct_pred_draw = 0.1
# incorrect_pred_draw = -0.1
# illegal_move = -1000.0  # was -0.1
# checkmate_reward = 1000.0  # was 0.3
# draw_reward = -1.0  # was -0.1
# move_bonus = 1.0  # was 0.02

# Rewards (new)
correct_pred_win = 0.0
incorrect_pred_win = 0.0
correct_pred_draw = 0.0
incorrect_pred_draw = 0.0
illegal_move = -1.0
checkmate_reward = 1.0  # was 0.3
draw_reward = -1.0  # was -0.1
move_bonus = 0.01  # was 0.02

# Rewards
# correct_pred_win = 0.0
# incorrect_pred_win = 0.0
# correct_pred_draw = 0.0
# incorrect_pred_draw = 0.0
#
# illegal_move = -1.0
# checkmate_reward = 0.0
# draw_reward = 0.0
# move_bonus = 0.0




# Eval Clipping
eval_lb = -1
eval_ub = 1
mate_steps_ub = 10




def clip_eval(eval):  # Clips between -1 and 1 (1 is 10 pawns)
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
        eval = -1 - ((11-abs(moves_to_mate)) * 0.01)
        return eval
    else:
        eval = 1 + ((11-abs(moves_to_mate)) * 0.01)
        return eval




def calc_reward(engine, game, n=100000, info=False, pad=True):
    # Ok so we are only giving a sparse reward for the final move for both colors
    # - if illegal move, then reward is -1.0
    # - if checkmate, then reward is 1.0 for the winning side and -1.0 for the losing side
    # - if draw, then reward is -0.2 for both sides
    # - if game is on-going, reward is the positional eval normalized to [-1, 1] where negative if losing and positive if winning

    
    uci_moves = game.split(' ')

    # validate it is longer than 1 move
    if len(uci_moves) == 1:
        # only white made a move
        board = chess.Board()
        uci_move = chess.Move.from_uci(uci_moves[0])
        if uci_move in config.non_move_tokens or uci_move not in board.legal_moves:
            white_rewards = [illegal_move]
            black_rewards = []
        else:
            white_rewards = [0.0]
            black_rewards = []
        # padding 
        if pad is True:
            pad_len = config.seq_length - 1
            white_rewards += ([0.0] * (pad_len - len(white_rewards)))
            black_rewards += ([0.0] * (pad_len - len(black_rewards)))
        return white_rewards, black_rewards



    last_move = uci_moves[-1]       # the last move is NOT guaranteed to be legal
    scnd_last_move = uci_moves[-2]  # the second to last move is guaranteed to be legal
    prev_moves = uci_moves[:-2] # the previous moves are guaranteed to be legal

    black_rewards = []
    white_rewards = []

    # - push previous moves to board and initialize rewards
    board = chess.Board()
    for idx, uci_move in enumerate(prev_moves):
        move = chess.Move.from_uci(uci_move)
        board.push(move)
        if idx % 2 == 0: # White's turn
            white_rewards.append(move_bonus)
        else:
            black_rewards.append(move_bonus)

    # - push second to last move to board
    white_turn = (board.turn == chess.WHITE)
    board.push(chess.Move.from_uci(scnd_last_move))

    curr_time = time.time()
    analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
    # print("Time taken for analysis:", time.time() - curr_time)
    score = analysis[0]["score"]
    if score.is_mate():
        if white_turn:
            moves_to_mate = score.white().mate()
        else:
            moves_to_mate = score.black().mate()
        pos_eval = get_moves_to_mate_eval(moves_to_mate)
        if white_turn:
            white_rewards.append(pos_eval)
            # white_rewards.append(0.0)
        else:
            black_rewards.append(pos_eval)
            # black_rewards.append(0.0)
    else:
        if white_turn:
            pos_eval = score.white().score()
            white_rewards.append(clip_eval(pos_eval))
            # white_rewards.append(0.0)
        else:
            pos_eval = score.black().score()
            black_rewards.append(clip_eval(pos_eval))
            # black_rewards.append(0.0)


    # - push last move to board
    white_turn = (board.turn == chess.WHITE)
    last_move_uci = chess.Move.from_uci(last_move)
    # check if illegal
    if last_move_uci in config.non_move_tokens or last_move_uci not in board.legal_moves:
        if white_turn:
            white_rewards.append(illegal_move)
        else:
            black_rewards.append(illegal_move)
    else:
        board.push(last_move_uci)
        analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
        score = analysis[0]["score"]
        if score.is_mate():
            if white_turn:
                moves_to_mate = score.white().mate()
            else:
                moves_to_mate = score.black().mate()
            pos_eval = get_moves_to_mate_eval(moves_to_mate)
            if white_turn:
                white_rewards.append(pos_eval)
                # white_rewards.append(0.0)
            else:
                black_rewards.append(pos_eval)
                # black_rewards.append(0.0)
        else:
            if white_turn:
                pos_eval = score.white().score()
                white_rewards.append(clip_eval(pos_eval))
                # white_rewards.append(0.0)
            else:
                pos_eval = score.black().score()
                black_rewards.append(clip_eval(pos_eval))
                # black_rewards.append(0.0)


    pad_len = config.seq_length - 1
    if len(white_rewards) < pad_len and pad is True:
        white_rewards += ([0.0] * (pad_len - len(white_rewards)))
    if len(black_rewards) < pad_len and pad is True:
        black_rewards += ([0.0] * (pad_len - len(black_rewards)))

    # Clip all rewards between -1 and 1
    white_rewards = [np.clip(reward, -1.0, 1.0) for reward in white_rewards]
    black_rewards = [np.clip(reward, -1.0, 1.0) for reward in black_rewards]

    # white_rewards = [reward / np.abs(illegal_move) for reward in white_rewards]
    # black_rewards = [reward / np.abs(illegal_move) for reward in black_rewards]

    return white_rewards, black_rewards



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



def calc_reward_move_simple(color_turn, prev_score, top_line_score):
    if top_line_score.is_mate():
        if color_turn == 'white':
            moves_to_mate = top_line_score.white().mate()
        else:
            moves_to_mate = top_line_score.black().mate()
        moves_to_mate = clip_mate_steps(moves_to_mate)
        if moves_to_mate > 0:  # You are checkmating the opponent
            reward = 11 - moves_to_mate
        else:  # You are getting checkmated
            reward = -11 + abs(moves_to_mate)
    else:
        if color_turn == 'white':
            eval = top_line_score.white().score()
        else:
            eval = top_line_score.black().score()
        reward = clip_eval(eval)
    return reward # Reward is between -10 and 10





def calc_reward_move(color_turn, prev_score, top_line_score):
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

    return move_reward





def calc_reward_slice(engine, game, slice, n=100000, info=False):
    uci_moves = game.split(' ')
    rewards = []  # Always from white's perspective
    sample_weights = []
    engine_eval_history = []
    trans_eval_history = []
    pad_len = config.seq_length


    # Rewards:
    # 1. positional eval after making move minus positional eval before making move
    board = chess.Board()
    analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
    prev_score = analysis[0]["score"]
    # engine_eval_history.append(prev_score.white().score())

    for idx, uci_move in enumerate(uci_moves):
        white_turn = (board.turn == chess.WHITE)
        color_turn = 'white'
        if board.turn == chess.BLACK:
            color_turn = 'black'


        if idx == (slice[0]-1):
            analysis = engine.analyse(board, chess.engine.Limit(nodes=n), multipv=1)
            prev_score = analysis[0]["score"]
            board.push_uci(uci_move)
            rewards.append(0)
            sample_weights.append(0)
            continue
        elif idx < (slice[0]-1):
            board.push_uci(uci_move)
            rewards.append(0)
            sample_weights.append(0)
            continue
        elif idx > (slice[1]):
            rewards.append(0)
            sample_weights.append(0)
            break
        sample_weights.append(1)





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


        if prev_score.white().score() is not None:
            engine_eval_history.append(prev_score.white().score() / 100.0)
        else:
            engine_eval_history.append(0.0)


        move_reward += move_bonus
        rewards.append(move_reward)
        prev_score = top_line_score
        trans_eval_history.append(eval)


    if info is True:
        info = {}
        info['engine_eval_history'] = engine_eval_history
        info['trans_eval_history'] = trans_eval_history
        return rewards, info
    else:
        if len(rewards) < pad_len:
            rewards += ([0] * (pad_len - len(rewards)))
        if len(sample_weights) < pad_len:
            sample_weights += ([0] * (pad_len - len(sample_weights)))
        return rewards, sample_weights




if __name__ == '__main__':

    nodes = 10000000

    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    threads = config.AVAILABLE_CPUS
    engine.configure({'Threads': threads, "Hash": 4096 * 2})


    # uci_game = 'g1f3 c7c5 g2g3 g8f6 f1g2 g7g6 c2c4 f8g7 e1g1 e8g8 d2d4 b8c6 d4d5 c6a5 b1a3 d7d6 a1b1 a7a6 b2b4 c5b4'
    # uci_game = 'e2e4 f7f5 d2d4 g7g5 d1h5'
    # uci_game = 'e2e4'
    uci_game = 'e2e3 f7f5 d2d4 g7g5 f1c4 g8h6'

    curr_time = time.time()
    w_rewards, b_rewards = calc_reward(engine, uci_game, n=nodes, pad=False)
    print("Time taken:", time.time() - curr_time)
    print("White rewards:", w_rewards)
    print("Black rewards:", b_rewards)

    # uci_game = 'e2e4 f7f5 d2d4 g7g5 d1h5'

    # curr_time = time.time()
    # w_rewards, b_rewards = calc_reward(engine, uci_game, n=nodes, pad=False)
    # print("Time taken:", time.time() - curr_time)
    # print("White rewards:", w_rewards)
    # print("Black rewards:", b_rewards)



    # Close engine
    engine.quit()







