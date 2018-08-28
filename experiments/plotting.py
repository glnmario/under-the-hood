import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from itertools import chain
from collections import defaultdict
import os


MODE = 'classification'
CONTEXT_SIZE = 6
NUM_BEFORE_SUBJECT = 2
NUM_AFTER_VERB = 1
NUM_ATTRACTORS = 0
NUM_HELPFUL = 0


data_dict_paths_wrong = 'results/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_wrong_test/test/regression/plotting_dict.pickle'
data_dict_paths_correct = 'results/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_correct_test/test/regression/plotting_dict.pickle'


def plot_data(data_dict_paths_correct, data_dict_paths_wrong, plot_savefile, mode):
    with open(data_dict_paths_correct, 'rb') as f:
        data_correct = pickle.load(f)

    with open(data_dict_paths_wrong, 'rb') as f:
        data_wrong = pickle.load(f)

    data_correct['lstm_pred'] = np.array(['correct'] * len(data_correct['scores']))
    data_wrong['lstm_pred'] = np.array(['wrong'] * len(data_wrong['scores']))

    data_merged = defaultdict(list)

    for k, v in chain(data_correct.items(), data_wrong.items()):
        data_merged[k].append(v)

    overall_scores = data_merged['mean_score']

    data_merged_new = defaultdict(list)
    for key in data_merged.keys():
        if str(key) == 'mean_score':
            continue
        else:
            data_merged_new[key] = data_merged[key]

    for key in list(data_merged_new.keys()):
        data_merged_new[key] = [item for sublist in data_merged_new[key] for item in sublist]

    #data_merged['squared error'] = data_merged.pop('scores')
    print(data_merged_new.keys())
    data = pd.DataFrame(data_merged_new, columns=list(data_merged_new.keys()))

    if mode == 'regression':
        sns.set(font_scale=1.5)
        data.columns = ['squared error', 'timesteps', 'activation', 'layer', 'lstm prediction']
        plot = sns.factorplot(x="timesteps", y="squared error", hue="lstm prediction", palette={'correct': 'green', 'wrong': 'blue'}, row='layer', col="activation", legend_out=True, data=data)
        print('MSE from top left to bottom right:', overall_scores)
    else:
        sns.set(font_scale=1.5)
        data.columns = ['accuracy', 'timesteps', 'activation', 'layer', 'lstm prediction']
        plot = sns.factorplot(x="timesteps", y="accuracy", hue="lstm prediction", palette={'correct': 'green', 'wrong': 'blue'}, row='layer', col="activation", legend_out=True, data=data)
        print('Mean accuracies correct:', overall_scores[0])
        print('Mean accuracies false:', overall_scores[1])

    plot.savefig(plot_savefile + '.png')
    
def plot_modified_data(data_dict_paths_correct, data_dict_paths_wrong, modified_correct_plotting_dict, modified_wrong_plotting_dict, plot_savefile, mode):
    with open(data_dict_paths_correct, 'rb') as f:
        data_correct = pickle.load(f)

    with open(data_dict_paths_wrong, 'rb') as f:
        data_wrong = pickle.load(f)
        
    with open(modified_correct_plotting_dict, 'rb') as f:
        modified_data_correct = pickle.load(f)

    with open(modified_wrong_plotting_dict, 'rb') as f:
        modified_data_wrong = pickle.load(f)

    data_correct['lstm_pred'] = np.array(['correct'] * len(data_correct['scores']))
    data_wrong['lstm_pred'] = np.array(['wrong'] * len(data_wrong['scores']))
    modified_data_correct['lstm_pred'] = np.array(['modified_correct'] * len(modified_data_correct['scores']))
    modified_data_wrong['lstm_pred'] = np.array(['modified_wrong'] * len(modified_data_wrong['scores']))

    data_merged = defaultdict(list)

    for k, v in chain(data_correct.items(), data_wrong.items(), modified_data_correct.items(), modified_data_wrong.items()):
        data_merged[k].append(v)

    overall_scores = data_merged['mean_score']

    data_merged_new = defaultdict(list)
    for key in data_merged.keys():
        if str(key) == 'mean_score':
            continue
        else:
            data_merged_new[key] = data_merged[key]

    for key in list(data_merged_new.keys()):
        data_merged_new[key] = [item for sublist in data_merged_new[key] for item in sublist]

    #data_merged['squared error'] = data_merged.pop('scores')
    print(data_merged_new.keys())
    data = pd.DataFrame(data_merged_new, columns=list(data_merged_new.keys()))

    if mode == 'regression':
        sns.set(font_scale=1.5)
        data.columns = ['squared error', 'timesteps', 'activation', 'layer', 'lstm prediction']
        plot = sns.factorplot(x="timesteps", y="squared error", hue="lstm prediction", palette={'correct': 'darkgreen', 'modified_correct': 'lightgreen', 'wrong': 'crimson', 'modified_wrong': 'salmon'}, row='layer', col="activation", legend_out=True, data=data)
        print('MSE from top left to bottom right:', overall_scores)
    else:
        sns.set(font_scale=1.5)
        data.columns = ['accuracy', 'timesteps', 'activation', 'layer', 'lstm prediction']
        plot = sns.factorplot(x="timesteps", y="accuracy", hue="lstm prediction", palette={'correct': 'darkgreen', 'modified_correct': 'lightgreen', 'wrong': 'crimson', 'modified_wrong': 'salmon'}, row='layer', col="activation", legend_out=True, data=data)
        print('Mean accuracies correct:', overall_scores[0])
        print('Mean accuracies false:', overall_scores[1])

    plot.savefig(plot_savefile + '.png')
    
def plot_prediction_data(data_dict_paths_correct, data_dict_paths_wrong, plot_savefile, mode):
    with open(data_dict_paths_correct, 'rb') as f:
        data_correct = pickle.load(f)

    with open(data_dict_paths_wrong, 'rb') as f:
        data_wrong = pickle.load(f)

    data_correct['lstm_pred'] = ['correct_sing', 'correct_plur'] * (int(len(data_correct['predictions'])/2))
    data_correct['lstm_pred'] = np.array(data_correct['lstm_pred'])
    
    data_wrong['lstm_pred'] = ['wrong_sing', 'wrong_plur'] * (int(len(data_wrong['predictions'])/2))
    data_wrong['lstm_pred'] = np.array(data_wrong['lstm_pred'])

    data_merged = defaultdict(list)

    for k, v in chain(data_correct.items(), data_wrong.items()):
        data_merged[k].append(v)

    data_merged_new = defaultdict(list)
    for key in data_merged.keys():
        data_merged_new[key] = data_merged[key]

    for key in list(data_merged_new.keys()):
        print(data_merged_new[key])
        data_merged_new[key] = [item for sublist in data_merged_new[key] for item in sublist]
        print(data_merged_new[key])

    print(data_merged_new.keys())
    data = pd.DataFrame(data_merged_new, columns=list(data_merged_new.keys()))

    sns.set(font_scale=1.5)
    data.columns = ['prob. verb singular', 'timesteps', 'activation', 'layer', 'lstm prediction']
    plot = sns.factorplot(x="timesteps", y='prob. verb singular', hue="lstm prediction", palette={'correct_sing': 'darkgreen', 'correct_plur': 'lightgreen', 'wrong_sing': 'crimson', 'wrong_plur': 'salmon'}, row='layer', col="activation", legend_out=True, data=data)

    plot.savefig(plot_savefile + '.png')


def plot_heatmap(data_dict_path, plot_savefile, lstm_mode, mode):
    c_map = 'YlOrBr'

    with open(data_dict_path, 'rb') as f:
        data = pickle.load(f)

    data = pd.DataFrame(data, columns=list(data.keys()))
    sns.set()

    activations = ["hx", "cx", "f_g", "i_g", "o_g", "c_tilde"]
    layers = ['l0', 'l1']

    for layer in layers:
        for act in activations:
            activation_column = data['activations'] == act
            layer_column = data['layers'] == layer
            heatmap_data = data[activation_column & layer_column]
#            print(len(heatmap_data))
            heatmap_data = heatmap_data.pivot('timestep_test', 'timestep_train', 'scores')
            if mode == 'classification':
                plot = sns.heatmap(heatmap_data, annot=True, linewidths=.5, cmap=c_map, vmin=0, vmax=1)
            else:
                plot = sns.heatmap(heatmap_data, annot=True, linewidths=.5, cmap=c_map, center=0.5)
            plot.invert_yaxis()
            plt.show()

            fig = plot.get_figure()

            fig.savefig("{}-{}-{}-{}.png".format(plot_savefile, lstm_mode, layer, act))


def plot_heatmap_component_wise(data_dict_path, plot_savefile, lstm_mode, num_timesteps, mode):
    with open(data_dict_path, 'rb') as f:
        data = pickle.load(f)

    data = pd.DataFrame(data, columns=list(data.keys()))
    sns.set()

    activations = ["hx", "cx", "f_g", "i_g", "o_g", "c_tilde"]
    layers = ['l0', 'l1']

    for timestep in range(num_timesteps):
        timestep_column = data['timestep'] == timestep
        heatmap_data = data[timestep_column]

        heatmap_data = heatmap_data.pivot('test_act', 'train_act', 'scores')
        if mode == 'classification':
            plot = sns.heatmap(heatmap_data, annot=True, linewidths=.5, cmap=c_map, vmin=0, vmax=1)
        else:
            plot = sns.heatmap(heatmap_data, annot=True, linewidths=.5, cmap=c_map, center=0.5)
        plot.invert_yaxis()
        plt.show()

        fig = plot.get_figure()

        fig.savefig("{}-{}-{}_component_wise.png".format(plot_savefile, lstm_mode, timestep))
