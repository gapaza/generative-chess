import chess
import chess.engine
import multiprocessing
import config


class RewardProcessor(multiprocessing.Process):
    def __init__(self):
        super(RewardProcessor, self).__init__()
        self.non_move_tokens = config.end_of_game_tokens + config.special_tokens + ['']

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 12, "Hash": 1024})
        self.nodes = 200000
        self.lines = 1

    def run(self, games):
        all_rewards = []

        for game in games:
            game_ended = False
            rewards = []
            board = chess.Board()
            curr_eval = 0.2  # starting position evaluation
            uci_moves = [config.id2token[x] for x in game]

            for uci_move in uci_moves:
                if game_ended is False:
                    if uci_move in config.end_of_game_tokens or uci_move in config.special_tokens:
                        game_ended = True
                        rewards.append(-1)
                        continue
                    move = chess.Move.from_uci(uci_move)
                    if move not in board.legal_moves:  # ILLEGAL MOVE
                        game_ended = True
                        rewards.append(-1)
                        continue
                    white_turn = (board.turn == chess.WHITE)
                    board.push(move)

                    if board.is_checkmate():
                        game_ended = True
                        rewards.append(1)
                    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                        game_ended = True
                        rewards.append(-0.1)
                    else:
                        analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=1)
                        line = analysis[0]
                        new_eval = line["score"].white().score()
                        if new_eval is None:
                            new_eval = curr_eval
                        else:
                            new_eval = new_eval / 100
                        eval_norm = 20.0
                        reward = abs(new_eval - curr_eval) / eval_norm
                        if white_turn and new_eval < curr_eval:
                            reward *= -1
                        elif not white_turn and new_eval > curr_eval:
                            reward *= -1
                        rewards.append(reward)

            while len(rewards) < config.seq_length - 1:
                rewards.append(0)

            all_rewards.append(rewards)
        return all_rewards

    def eval_game(self, game):
        # Reward Params
        curr_eval = 0.2
        invalid_move_penalty = -1
        checkmate_reward = 1
        draw_reward = -0.1
        eval_norm = 20.0
        move_reward = 0.1

        game_ended = False
        rewards = []
        board = chess.Board()

        # Iterate over moves
        uci_moves = [config.id2token[x] for x in game]
        for uci_move in uci_moves:

            # Illegal moves
            if uci_move in self.non_move_tokens:
                rewards.append(invalid_move_penalty)
                break
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                rewards.append(invalid_move_penalty)
                break

            # Legal move
            board.push(move)

            # Case 1: Checkmate
            if board.is_checkmate():
                rewards.append(checkmate_reward)
                break
            # Case 2: Stalemate, Insufficient Material, 75 moves
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                rewards.append(draw_reward)
                break
            # Case 3: Normal move
            else:

                # 1. Get evaluation (white perspective)
                analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)
                line = analysis[0]
                new_eval = line["score"].white().score()
                if new_eval is None:
                    new_eval = curr_eval
                else:
                    new_eval = new_eval / 100

                # 2. Determine reward
                reward = abs(new_eval - curr_eval) / eval_norm
                if board.turn == chess.WHITE and new_eval < curr_eval:
                    reward *= -1
                elif board.turn == chess.BLACK and new_eval > curr_eval:
                    reward *= -1
                reward += move_reward
                rewards.append(reward)
                curr_eval = new_eval





    def eval_game(self, game):
        # Reward Params
        curr_eval = 0.2
        invalid_move_penalty = -1
        checkmate_reward = 1
        draw_reward = -0.1



        game_ended = False
        rewards = []
        board = chess.Board()

        # Iterate over moves
        uci_moves = [config.id2token[x] for x in game]
        for uci_move in uci_moves:

            # Illegal moves
            if uci_move in self.non_move_tokens:
                rewards.append(invalid_move_penalty)
                break
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                rewards.append(invalid_move_penalty)
                break

            # Legal move
            board.push(move)

            # Case 1: Checkmate
            if board.is_checkmate():
                rewards.append(checkmate_reward)
                break
            # Case 2: Stalemate, Insufficient Material, 75 moves
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                rewards.append(draw_reward)
                break
            # Case 3: Normal move
            else:
                analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)
                line = analysis[0]
                new_eval = line["score"].white().score()


                if new_eval is None:
                    new_eval = curr_eval
                else:
                    new_eval = new_eval / 100
                eval_norm = 20.0
                reward = abs(new_eval - curr_eval) / eval_norm
                if board.turn == chess.WHITE and new_eval < curr_eval:
                    reward *= -1
                elif board.turn == chess.BLACK and new_eval > curr_eval:
                    reward *= -1
                rewards.append(reward)
                curr_eval = new_eval













if __name__ == '__main__':
    proc = RewardProcessor()



