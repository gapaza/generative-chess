from multiprocessing import Process, Queue
from stockfish.utils import get_stockfish
import config
from stockfish.rewards.reward_1 import calc_reward




class StockfishProcess(Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.engine = get_stockfish()



    def run(self):
        pass








