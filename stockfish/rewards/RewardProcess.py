from stockfish.utils import get_stockfish
from stockfish.rewards.reward_1 import calc_reward, calc_reward_batch
import time

from multiprocessing import Process, Queue
import multiprocessing as mp
# mp.set_start_method('spawn', force=True)



class StockfishProcess(Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.engine = None
        self.threads = 12
        self.nodes = 200000


    def run(self):
        if self.engine is None:
            print('Starting stockfish')
            self.engine = get_stockfish(threads=self.threads)

        running = True
        while running:
            try:
                # Check for new task
                if not self.task_queue.empty():
                    task = self.task_queue.get_nowait()
                    # print('Task:', task)
                    if task is None:
                        # None is a signal to stop the process
                        break
                    # Process the task using process_input
                    result = calc_reward_batch(task, self.engine, n=self.nodes)
                    if result == 'exit':
                        running = False
                    self.result_queue.put(result)
                else:
                    # Sleep for a short while to avoid busy waiting
                    time.sleep(0.1)
            except Exception as e:
                # Handle possible exceptions (e.g., queue issues)
                print(f"An error occurred: {e}")

        # Properly close the engine before exiting
        self.engine.quit()



