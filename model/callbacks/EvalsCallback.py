import tensorflow as tf
from evals.AbstractEval import AbstractEval
import json
import os
import config

class EvalsCallback(tf.keras.callbacks.Callback):
    def __init__(self, step_interval, weights_path):
        super().__init__()
        self.step_interval = step_interval
        self.weights_path = weights_path
        self.weights_path_basename = os.path.basename(weights_path)
        self.eval = AbstractEval()

    def on_batch_end(self, batch, logs=None):
        if batch % self.step_interval == 0:
            evals = [ 'opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
            eval_history = self.eval.run_eval(self.model, themes=evals)
            eval_history['step_interval'] = self.step_interval
            f_path = os.path.join(config.results_dir, 'evals', self.weights_path_basename+'.json')
            with open(f_path, 'w') as f:
                json.dump(eval_history, f, indent=4)

            # Save model weights
            self.model.save_weights(self.weights_path+'-eval')















