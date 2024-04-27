from model.api import ChessGPT_API
from model.api_a2 import ChessGPTa2_API
import config
import os



def run_old():
    rl_model_path = os.path.join(config.results_dir, 'run_10', 'pretrained', 'actor_weights_400')
    model_path = os.path.join(config.weights_dir, 'chess-gpt-v6')
    # model_path = rl_model_path

    api = ChessGPT_API(model_path=model_path, user_plays_white=True)
    api.play_interactive_game()




if __name__ == '__main__':
    model_path = os.path.join(config.weights_dir, 'chess-gpt-a5')
    api = ChessGPTa2_API(model_path=model_path, user_plays_white=True)
    api.play_interactive_game()







