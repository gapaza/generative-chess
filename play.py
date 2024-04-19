from model.api import ChessGPT_API
import config
import os



rl_model_path = os.path.join(config.results_dir, 'run_2', 'pretrained', 'actor_weights_250')




if __name__ == '__main__':
    model_path = config.model_path
    # model_path = rl_model_path


    api = ChessGPT_API(model_path=model_path, user_plays_white=False)
    api.play_interactive_game()







