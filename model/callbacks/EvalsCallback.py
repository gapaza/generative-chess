import tensorflow as tf
from evals.AbstractEval import AbstractEval
import json
import os
import config

class EvalsCallback(tf.keras.callbacks.Callback):
    def __init__(self, step_interval, weights_path, model_type='v2'):
        super().__init__()
        self.step_interval = step_interval
        self.model_type = model_type
        self.weights_path = weights_path
        self.eval = AbstractEval()

    def on_batch_end(self, batch, logs=None):
        if batch % self.step_interval == 0:
            # print('EVALS CALLBACK:', batch)
            evals = ['advantage', 'mate', 'fork', 'pin', 'equality', 'opening', 'middlegame', 'endgame']
            eval_history = self.eval.run_eval(self.model, type=self.model_type, themes=evals)
            f_path = os.path.join(config.results_dir, 'evals.json')
            with open(f_path, 'w') as f:
                json.dump(eval_history, f, indent=4)

            # Save model weights
            self.model.save_weights(self.weights_path+'-eval')















