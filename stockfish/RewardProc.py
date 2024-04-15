import chess
import chess.engine
import multiprocessing
import config



class RewardProcessor(multiprocessing.Process):
    def __init__(self):
        super(RewardProcessor, self).__init__()

        # Stockfish Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': 12, "Hash": 1024})
        self.nodes = 100000
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





if __name__ == '__main__':
    proc = RewardProcessor()



