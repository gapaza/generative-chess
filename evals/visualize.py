import numpy as np
import pandas as pd
import config
import json
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec



def plot_compare_training_evals(eval_history_file_1, eval_history_file_2):
    # open json file
    file_basename_1 = os.path.basename(eval_history_file_1).replace('.json', '')
    file_basename_2 = os.path.basename(eval_history_file_2).replace('.json', '')
    history_1 = json.load(open(eval_history_file_1))
    history_2 = json.load(open(eval_history_file_2))
    step_size = int(history_1['step_interval'])
    step_size_2 = int(history_2['step_interval'])

    evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
    steps_1 = [x * step_size for x in range(len(history_1['opening']))]
    steps_2 = [x * step_size_2 for x in range(len(history_2['opening']))]

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(13, 4))
    fig.suptitle('Evals', fontsize=16)

    # Subplot for opening, middlegame, and endgame
    plt.subplot(gs[0, 0])
    for eval in evals[:3]:
        plt.plot(steps_1, history_1[eval], label=eval+' | ' + file_basename_1)
        plt.plot(steps_2, history_2[eval], label=eval+' | ' + file_basename_2)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    # plt.legend()
    plt.title('Game Phases')

    # Subplot for equality and advantage
    plt.subplot(gs[0, 1])
    for eval in evals[3:5]:
        plt.plot(steps_1, history_1[eval], label=eval+' | ' + file_basename_1)
        plt.plot(steps_2, history_2[eval], label=eval+' | ' + file_basename_2)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    # plt.legend()
    plt.title('Even / Winning Positions')

    # Subplot for mate, fork, and pin
    plt.subplot(gs[0, 2])
    for eval in evals[5:]:
        plt.plot(steps_1, history_1[eval], label=eval+' | ' + file_basename_1)
        plt.plot(steps_2, history_2[eval], label=eval+' | ' + file_basename_2)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    # plt.legend()
    plt.title('Basic Tactics')

    plt.tight_layout()
    # if '.json' in file_basename:
    #     file_basename = file_basename.replace('.json', '')
    save_path = os.path.join(config.results_dir, 'evals', 'plots', 'comparison-training-evals.png')
    plt.savefig(save_path)









def plot_training_evals(eval_history_file):
    # open json file
    file_basename = os.path.basename(eval_history_file)
    history = json.load(open(eval_history_file))
    step_size = int(history['step_interval'])
    # extract history
    history = extract_history(history)

    evals = ['opening', 'middlegame', 'endgame', 'equality', 'advantage', 'mate', 'fork', 'pin']
    steps = [x * step_size for x in range(len(history['opening']))]

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(13, 4))
    fig.suptitle('Evals', fontsize=16)


    # Subplot for opening, middlegame, and endgame
    plt.subplot(gs[0, 0])
    for eval in evals[:3]:
        plt.plot(steps, history[eval], label=eval)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Game Phases')

    # Subplot for equality and advantage
    plt.subplot(gs[0, 1])
    for eval in evals[3:5]:
        plt.plot(steps, history[eval], label=eval)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Even / Winning Positions')

    # Subplot for mate, fork, and pin
    plt.subplot(gs[0, 2])
    for eval in evals[5:]:
        plt.plot(steps, history[eval], label=eval)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Basic Tactics')

    plt.tight_layout()

    # remove .json from basename if exists
    if '.json' in file_basename:
        file_basename = file_basename.replace('.json', '')
    save_path = os.path.join(config.results_dir, 'evals', 'plots', file_basename+'-training-evals.png')
    plt.savefig(save_path)




def extract_history(history):
    new_history = {}
    for key in history.keys():
        if not isinstance(history[key], list):
            continue
        elif len(history[key]) == 0:
            continue
        else:
            new_history[key] = history[key]
    return new_history



if __name__ == '__main__':
    eval_history_file = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-s.json')
    eval_history_file_2 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-s-puzzles.json')
    eval_history_file_3 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-s-puzzles-2.json')
    eval_history_file_4 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-m-puzzles.json')
    eval_history_file_5 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-l-puzzles.json')
    eval_history_file_6 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-l-puzzles-8h.json')
    eval_history_file_7 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-xl-puzzles.json')
    eval_history_file_8 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v1-xxl-puzzles.json')
    eval_history_file_9 = os.path.join(config.results_dir, 'evals', 'chess-gpt-v6.json')

    # plot_training_evals(eval_history_file_2)

    # plot_compare_training_evals(eval_history_file_3, eval_history_file_4)

    # plot_compare_training_evals(eval_history_file_4, eval_history_file_5)

    # plot_compare_training_evals(eval_history_file_5, eval_history_file_6)

    # plot_compare_training_evals(eval_history_file_6, eval_history_file_7)

    # plot_compare_training_evals(eval_history_file_7, eval_history_file_8)

    plot_compare_training_evals(eval_history_file_2, eval_history_file_9)

    print('Done')
























