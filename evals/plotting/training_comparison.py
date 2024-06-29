import config
import json
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

save_dir = os.path.join(config.evals_dir, 'out')

def plot_training_comparison(history_files, labels=None, save_name=None, bounds=True, local_save_dir=None):
    evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']

    if type(history_files[0]) is not str:
        history_files_json = history_files
        if labels is None:
            labels = []
            for idx, history_json in enumerate(history_files_json):
                labels.append(str(idx))
    else:
        if labels is None:
            labels = []
            for history_file in history_files:
                basename = os.path.basename(history_file)
                labels.append(basename.split('.')[0])
        history_files_json = []
        for history_file in history_files:
            with open(os.path.join(config.evals_dir, history_file), 'r') as f:
                history_files_json.append(json.load(f))



    all_history_steps = []
    for history_json in history_files_json:
        history_steps = [x * history_json['step_interval'] for x in range(len(history_json['opening']))]
        all_history_steps.append(history_steps)

    gs = gridspec.GridSpec(2, 4)
    gs_pos = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle('Evals', fontsize=16)

    for i, eval_name in enumerate(evals):
        gs_x, gs_y = gs_pos[i]
        plt.subplot(gs[gs_x, gs_y])
        for j, history_steps in enumerate(all_history_steps):
            last_val = history_files_json[j][eval_name][-1]
            label = labels[j] + ' (' + str(round(last_val, 2)) + ')'
            y_vals = history_files_json[j][eval_name]
            plt.plot(history_steps, history_files_json[j][eval_name], label=label)
            diff = y_vals[-1] - y_vals[0]

        plt.title(eval_name + ' ' + str(round(diff, 2)))
        # plt.title(eval_name)
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        if bounds is True:
            plt.ylim(-0.01, 1)
        plt.legend()

    plt.tight_layout()
    if local_save_dir is None:
        local_save_dir = save_dir

    if save_name is not None:
        save_path = os.path.join(local_save_dir, save_name)
    else:
        save_path = os.path.join(local_save_dir, 'training_comparison.png')
    plt.savefig(save_path)



def plot_training_comparison_grouped(history_files, labels=None, save_name=None):
    evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']


    if labels is None:
        labels = []
        for history_file in history_files:
            basename = os.path.basename(history_file)
            labels.append(basename.split('.')[0])
    history_files_json = []
    for history_file in history_files:
        with open(os.path.join(config.evals_dir, history_file), 'r') as f:
            history_files_json.append(json.load(f))

    all_history_steps = []
    for history_json in history_files_json:
        history_steps = [x * history_json['step_interval'] for x in range(len(history_json['opening']))]
        all_history_steps.append(history_steps)

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(13, 4))
    fig.suptitle('Evals', fontsize=16)

    # Subplot: opening, middlegame, and endgame
    plt.subplot(gs[0, 0])
    for eval in evals[:3]:
        for j, history_steps in enumerate(all_history_steps):
            label = labels[j] + ' (' + eval + ')'
            plt.plot(history_steps, history_files_json[j][eval], label=label)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(-0.01, 1)
    plt.legend()
    plt.title('Game Phases')

    # Subplot: equality, advantage
    plt.subplot(gs[0, 1])
    for eval in evals[3:5]:
        for j, history_steps in enumerate(all_history_steps):
            label = labels[j] + ' (' + eval + ')'
            plt.plot(history_steps, history_files_json[j][eval], label=label)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(-0.01, 1)
    plt.legend()
    plt.title('Equal / Winning Positions')

    # Subplot: mate, fork, pin
    plt.subplot(gs[0, 2])
    for eval in evals[5:]:
        for j, history_steps in enumerate(all_history_steps):
            label = labels[j] + ' (' + eval + ')'
            plt.plot(history_steps, history_files_json[j][eval], label=label)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(-0.01, 1)
    plt.legend()
    plt.title('Basic Tactics')

    plt.tight_layout()
    if save_name is not None:
        save_path = os.path.join(save_dir, save_name)
    else:
        save_path = os.path.join(save_dir, 'training_comparison_grouped.png')
    plt.savefig(save_path)











if __name__ == '__main__':
    compare_files = [
        os.path.join(config.results_dir, 'evals', 'chess-gpt-v3.json'),
        os.path.join(config.results_dir, 'evals', 'chess-gpt-v6.json')
    ]
    model_labels = ['v3', 'v6']

    plot_training_comparison(compare_files, labels=model_labels)
    plot_training_comparison_grouped(compare_files, labels=model_labels)





















