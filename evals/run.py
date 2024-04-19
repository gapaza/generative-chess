import config, os
from evals.AbstractEval import AbstractEval

from model import get_pretrain_model as get_model
# from model import get_pretrain_model_v2 as get_model

if __name__ == '__main__':

    model_name = 'chess-gpt-v3'
    checkpoint_path = os.path.join(config.weights_dir, model_name)
    model = get_model(checkpoint_path=checkpoint_path)

    ae = AbstractEval()

    # results = ae.run_eval(model, type='v1', save_name=model_name)

    compare_files = ['chess-gpt-v4-1', 'chess-gpt-v4-2', 'chess-gpt-v3']
    compare_themes = ['advantage', 'mate', 'fork', 'pin', 'equality', 'opening', 'middlegame', 'endgame']
    ae.histogram_comparison(compare_files, themes=compare_themes)





