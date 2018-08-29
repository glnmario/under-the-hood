import torch
import pickle
import numpy as np

import os
import sys
sys.path.append("../experiments/")

from lstm import Forward_LSTM
from collections import Counter, defaultdict
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable


# Parameters
####################
VOCAB_SIZE = 50001
HIDDEN_SIZE = 650

AVG_INIT = False
####################


def extract_data(filepath, mode, vocab_size, hidden_size, context_len_before_subj, context_len_after_verb, w2i_dict_path, model_path):

    if mode not in ['train', 'test']:
        raise ValueError('Mode: train or test')

    try:
        idx_words_before_subj = filepath.index('words_before_subject')
    except ValueError:
        raise ValueError('Filepath should contain "words_before_subject".')

    try:
        idx_words_after_verb = filepath.index('words_after_verb')
    except ValueError:
        raise ValueError('Filepath should contain "words_after_verb".')

    num_words_before_subj_in_filename = int(filepath[idx_words_before_subj + len('words_before_subject')])
    num_words_after_verb_in_filename = int(filepath[idx_words_after_verb + len('words_after_verb')])

    if context_len_before_subj > num_words_before_subj_in_filename:
        raise ValueError('Dataset does not contain this many words before the subject.')

    if context_len_after_verb > num_words_after_verb_in_filename:
        raise ValueError('Dataset does not contain this many words after the verb.')

    with open(filepath) as corpus_f:
        lines = corpus_f.readlines()

    output_path_prefix = 'activations/{}/{}'.format(filepath[18:-4], mode)

    if os.path.exists(output_path_prefix):
        return output_path_prefix

    data = []
    for line in lines:
        split_line = line.split('\t')

        sent = split_line[0].split()
        label = 0 if split_line[2] == 'sing' else 1
        subj_idx = int(split_line[3])
        verb_idx = int(split_line[4])

        data.append([sent, label, subj_idx, verb_idx])

    lstm = Forward_LSTM(vocab_size, hidden_size, hidden_size, vocab_size, w2i_dict_path, model_path)

    if AVG_INIT:
        with open('../models/our_hidden.pickle', 'rb') as f_hid:
            h_list = pickle.load(f_hid)
        h0_l0_ = h_list[0]
        c0_l0_ = h_list[1]
        h0_l1_ = h_list[2]
        c0_l1_ = h_list[3]
    else:
        h0_l0 = torch.Tensor(torch.zeros(hidden_size))
        c0_l0 = torch.Tensor(torch.zeros(hidden_size))
        h0_l1 = torch.Tensor(torch.zeros(hidden_size))
        c0_l1 = torch.Tensor(torch.zeros(hidden_size))

    activation_names = ['hx', 'cx', 'f_g', 'i_g', 'o_g', 'c_tilde']
    hx_l0_list, cx_l0_list, f_g_l0_list = [], [], []
    i_g_l0_list, o_g_l0_list, c_tilde_l0_list = [], [], []
    hx_l1_list, cx_l1_list, f_g_l1_list = [], [], []
    i_g_l1_list, o_g_l1_list, c_tilde_l1_list = [], [], []

    l0_lists = [hx_l0_list, cx_l0_list, f_g_l0_list,
                i_g_l0_list, o_g_l0_list, c_tilde_l0_list]

    l1_lists = [hx_l1_list, cx_l1_list, f_g_l1_list,
                i_g_l1_list, o_g_l1_list, c_tilde_l1_list]

    labels = []

    for i, data_point in enumerate(data):
        sentence, label, subj_idx, verb_idx = data_point

        if AVG_INIT:
            h0_l0, c0_l0, h0_l1, c0_l1 = h0_l0_, c0_l0_, h0_l1_, c0_l1_

        for t, word_t in enumerate(sentence):
            out, layer0, layer1 = lstm(word_t, h0_l0, c0_l0, h0_l1, c0_l1)

            h0_l0, c0_l0 = layer0[:2]
            h0_l1, c0_l1 = layer1[:2]

            if t >= subj_idx - context_len_before_subj and t <= verb_idx + context_len_after_verb:

                labels.append(label)

                for idx, l in enumerate(l0_lists):
                    l.append(layer0[idx].detach().numpy())

                for idx, l in enumerate(l1_lists):
                    l.append(layer1[idx].detach().numpy())

    # Store data and labels

    output_path_prefix = 'activations/{}/{}'.format(filepath[18:-4], mode)

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    for idx, l in enumerate(l0_lists):
        filename = '{}/{}_l0.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    for idx, l in enumerate(l1_lists):
        filename = '{}/{}_l1.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    filename = '{}/labels.pickle'.format(output_path_prefix)
    with open(filename, 'wb') as f_out:
        pickle.dump(np.array(labels), f_out)

    return output_path_prefix

def extract_and_modify_data(filepath, mode, vocab_size, hidden_size, context_len_before_subj, context_len_after_verb, num_timesteps, w2i_dict_path, model_path, trained_classifier_path):

    if mode not in ['test']:
        raise ValueError('Mode must be test')

    try:
        idx_words_before_subj = filepath.index('words_before_subject')
    except ValueError:
        raise ValueError('Filepath should contain "words_before_subject".')

    try:
        idx_words_after_verb = filepath.index('words_after_verb')
    except ValueError:
        raise ValueError('Filepath should contain "words_after_verb".')

    num_words_before_subj_in_filename = int(filepath[idx_words_before_subj + len('words_before_subject')])
    num_words_after_verb_in_filename = int(filepath[idx_words_after_verb + len('words_after_verb')])

    if context_len_before_subj > num_words_before_subj_in_filename:
        raise ValueError('Dataset does not contain this many words before the subject.')

    if context_len_after_verb > num_words_after_verb_in_filename:
        raise ValueError('Dataset does not contain this many words after the verb.')

    with open(filepath) as corpus_f:
        lines = corpus_f.readlines()

    output_path_prefix = 'modified_activations/{}/{}'.format(filepath, mode)

    if os.path.exists(output_path_prefix):
        return output_path_prefix

    data = []
    for line in lines:
        split_line = line.split('\t')

        sent = split_line[0].split()
        label = 0 if split_line[2] == 'sing' else 1
        subj_idx = int(split_line[3])
        verb_idx = int(split_line[4])

        data.append([sent, label, subj_idx, verb_idx])

    lstm = Forward_LSTM(vocab_size, hidden_size, hidden_size, vocab_size, w2i_dict_path, model_path)

    if AVG_INIT:
        with open('../models/our_hidden.pickle', 'rb') as f_hid:
            h_list = pickle.load(f_hid)
        h0_l0_ = h_list[0]
        c0_l0_ = h_list[1]
        h0_l1_ = h_list[2]
        c0_l1_ = h_list[3]
    else:
        h0_l0 = torch.Tensor(torch.zeros(hidden_size))
        c0_l0 = torch.Tensor(torch.zeros(hidden_size))
        h0_l1 = torch.Tensor(torch.zeros(hidden_size))
        c0_l1 = torch.Tensor(torch.zeros(hidden_size))

    activation_names = ['hx', 'cx', 'f_g', 'i_g', 'o_g', 'c_tilde']
    layers = ['l0', 'l1']
    
    classifiers = defaultdict()
    
    for layer in layers:
        for act in activation_names:
            for time_step in range(num_timesteps):
                with open("{}/{}.pickle".format(trained_classifier_path, act + '_' + layer), 'rb') as trained_classifier:
                    classifiers[act + '_' + layer] = pickle.load(trained_classifier)
    
    hx_l0_list, cx_l0_list, f_g_l0_list = [], [], []
    i_g_l0_list, o_g_l0_list, c_tilde_l0_list = [], [], []
    hx_l1_list, cx_l1_list, f_g_l1_list = [], [], []
    i_g_l1_list, o_g_l1_list, c_tilde_l1_list = [], [], []

    l0_lists = [hx_l0_list, cx_l0_list, f_g_l0_list,
                i_g_l0_list, o_g_l0_list, c_tilde_l0_list]

    l1_lists = [hx_l1_list, cx_l1_list, f_g_l1_list,
                i_g_l1_list, o_g_l1_list, c_tilde_l1_list]

    labels = []

    for i, data_point in enumerate(data):
        sentence, label, subj_idx, verb_idx = data_point

        if AVG_INIT:
            h0_l0, c0_l0, h0_l1, c0_l1 = h0_l0_, c0_l0_, h0_l1_, c0_l1_

        for t, word_t in enumerate(sentence):
            out, layer0, layer1 = lstm(word_t, h0_l0, c0_l0, h0_l1, c0_l1)

            h0_l0, c0_l0, f_g_l0, i_g_l0, o_g_l0, c_tilde_l0 = layer0
            h0_l1, c0_l1, f_g_l1, i_g_l1, o_g_l1, c_tilde_l1 = layer1

            if t >= subj_idx - context_len_before_subj and t <= verb_idx + context_len_after_verb:
                time_step_under_consideration = t - subj_idx + context_len_before_subj
                labels.append(label)

                for idx, l in enumerate(l0_lists):
                    act_str = activation_names[idx]
                    weight, bias = classifiers[act_str + '_l0']
                    
                    weight = Variable(torch.tensor(weight, dtype=torch.double).squeeze(0), requires_grad=False)
                    bias = Variable(torch.tensor(bias, dtype=torch.double), requires_grad=False)
                    current_activation = Variable(torch.tensor(layer0[idx], dtype=torch.double), requires_grad=True)

                    if True:
                        
                        class_0_prob_a = torch.dot(weight, current_activation) + bias
                        
                        class_0_prob_b = F.sigmoid(class_0_prob_a)
                        total_prob = torch.tensor(1.0, dtype=torch.double)
                        class_1_prob_b = total_prob - class_0_prob_b
                        
                        class_0_prob_c = torch.log(class_0_prob_b)
                        class_1_prob_c = torch.log(class_1_prob_b)
                        
                        prediction = torch.tensor(torch.cat((class_1_prob_c, class_0_prob_c))).unsqueeze(0)
                        
                        params = [current_activation]
                        optimiser = torch.optim.SGD(params, lr=0.1)
                        optimiser.zero_grad()
                        
                        criterion = nn.NLLLoss()
                        gold_label = torch.tensor(label).unsqueeze(0)
                        loss = criterion(prediction, gold_label)
                        
                        loss.backward()
                        optimiser.step()
                        
                        if act_str == 'h0':
                            h0_l0 = current_activation
                        elif act_str == 'c0':
                            c0_l0 = current_activation
                    
                    l.append(current_activation.detach().numpy())

                for idx, l in enumerate(l1_lists):
                    act_str = activation_names[idx]
                    weight, bias = classifiers[act_str + '_l1']
                    
                    weight = Variable(torch.tensor(weight, dtype=torch.double).squeeze(0), requires_grad=False)
                    bias = Variable(torch.tensor(bias, dtype=torch.double), requires_grad=False)
                    current_activation = Variable(torch.tensor(layer1[idx], dtype=torch.double), requires_grad=True)
                    
                    if True:
                        
                        class_0_prob_a = torch.dot(weight, current_activation) + bias
                        
                        class_0_prob_b = F.sigmoid(class_0_prob_a)
                        total_prob = torch.tensor(1., dtype=torch.double)
                        class_1_prob_b = total_prob - class_0_prob_b
                        
                        class_0_prob_c = torch.log(class_0_prob_b)
                        class_1_prob_c = torch.log(class_1_prob_b)
                        
                        prediction = torch.tensor(torch.cat((class_1_prob_c, class_0_prob_c))).unsqueeze(0)
                        
                        params = [current_activation]
                        optimiser = torch.optim.SGD(params, lr=0.1)
                        optimiser.zero_grad()
                        
                        criterion = nn.NLLLoss()
                        gold_label = torch.tensor(label).unsqueeze(0)
                        loss = criterion(prediction, gold_label)
                        
                        loss.backward()
                        optimiser.step()
                        
                        if act_str == 'h0':
                            h0_l1 = current_activation
                        elif act_str == 'c0':
                            c0_l1 = current_activation
                        
                    l.append(current_activation.detach().numpy())
                    

    # Store data and labels

    output_path_prefix = 'modified_activations/{}/{}'.format(filepath, mode)

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    for idx, l in enumerate(l0_lists):
        filename = '{}/{}_l0.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    for idx, l in enumerate(l1_lists):
        filename = '{}/{}_l1.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    filename = '{}/labels.pickle'.format(output_path_prefix)
    with open(filename, 'wb') as f_out:
        pickle.dump(np.array(labels), f_out)

    return output_path_prefix

def extract_multi_class_data(filepath, mode, vocab_size, hidden_size, context_len_before_subj, context_len_after_verb, w2i_dict_path, model_path, labels_type, attractor_idx):

    if mode not in ['train', 'test']:
        raise ValueError('Mode: train or test')

    try:
        idx_words_before_subj = filepath.index('words_before_subject')
    except ValueError:
        raise ValueError('Filepath should contain "words_before_subject".')


    try:
        idx_words_after_verb = filepath.index('words_after_verb')
    except ValueError:
        raise ValueError('Filepath should contain "words_after_verb".')


    num_words_before_subj_in_filename = int(filepath[idx_words_before_subj + len('words_before_subject')])
    num_words_after_verb_in_filename = int(filepath[idx_words_after_verb + len('words_after_verb')])


    if context_len_before_subj > num_words_before_subj_in_filename:
        raise ValueError('Dataset does not contain this many words before the subject.')

    if context_len_after_verb > num_words_after_verb_in_filename:
        raise ValueError('Dataset does not contain this many words after the verb.')

    with open(filepath) as corpus_f:
        lines = corpus_f.readlines()

    data = []
    for line in lines:
        split_line = line.split('\t')

        sent = split_line[0].split()
        pos_sent = split_line[1].split()
        label = 0 if split_line[2] == 'sing' else 1
        subj_idx = int(split_line[3])
        verb_idx = int(split_line[4])

        data.append([sent, pos_sent, label, subj_idx, verb_idx])

    lstm = Forward_LSTM(vocab_size, hidden_size, hidden_size, vocab_size, w2i_dict_path, model_path)

    if AVG_INIT:
        with open('../models/our_hidden.pickle', 'rb') as f_hid:
            h_list = pickle.load(f_hid)

        h0_l0_ = h_list[0]
        c0_l0_ = h_list[1]
        h0_l1_ = h_list[2]
        c0_l1_ = h_list[3]
    else:
        h0_l0 = torch.Tensor(torch.zeros(hidden_size))
        c0_l0 = torch.Tensor(torch.zeros(hidden_size))
        h0_l1 = torch.Tensor(torch.zeros(hidden_size))
        c0_l1 = torch.Tensor(torch.zeros(hidden_size))

    activation_names = ['hx', 'cx', 'f_g', 'i_g', 'o_g', 'c_tilde']
    hx_l0_list,  cx_l0_list,  f_g_l0_list     = [], [], []
    i_g_l0_list, o_g_l0_list, c_tilde_l0_list = [], [], []
    hx_l1_list,  cx_l1_list,  f_g_l1_list     = [], [], []
    i_g_l1_list, o_g_l1_list, c_tilde_l1_list = [], [], []

    l0_lists = [hx_l0_list, cx_l0_list, f_g_l0_list,
               i_g_l0_list, o_g_l0_list, c_tilde_l0_list]

    l1_lists = [hx_l1_list, cx_l1_list, f_g_l1_list,
                i_g_l1_list, o_g_l1_list, c_tilde_l1_list]

    labels = []
    for i, data_point in enumerate(data):
        sentence, pos_sentence, label, subj_idx, verb_idx = data_point

        if AVG_INIT:
            h0_l0, c0_l0, h0_l1, c0_l1 = h0_l0_, c0_l0_, h0_l1_, c0_l1_

        for t, word_t in enumerate(sentence):
            out, layer0, layer1 = lstm(word_t, h0_l0, c0_l0, h0_l1, c0_l1)

            h0_l0, c0_l0 = layer0[:2]
            h0_l1, c0_l1 = layer1[:2]

            if t >= subj_idx - context_len_before_subj and t <= verb_idx + context_len_after_verb:

                if labels_type == 'timestep':
                    labels.append(t - subj_idx + context_len_before_subj)

                elif labels_type == 'subj_attractor_region':
                    if t < subj_idx:
                        labels.append(0)
                    elif t >= subj_idx and t < subj_idx + attractor_idx:
                        labels.append(1)
                    elif t >= subj_idx + attractor_idx and t < verb_idx:
                        labels.append(2)
                    elif t >= verb_idx:
                        labels.append(3)
                        
                elif labels_type == 'POS_tags':
                    if word_t == '<unk>':
                        labels.append('UNK')
                    else:
                        word_tag = pos_sentence[t]
                        if word_tag in ['NN', 'NNS']:
                            pos_label = 'NOUN'
                        elif word_tag in ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                            pos_label = 'VERB'
                        elif word_tag in ['AFX', 'JJ', 'JJR', 'JJS']:
                            pos_label = 'ADJ'
                        elif word_tag in ['EX', 'PRP', 'WP']:
                            pos_label = 'PRON'
                        elif word_tag in ['', '.', ',', '-LRB-', '-RRB-', ':', 'HYPH', '``', ';']:
                            pos_label = 'PUNCT'
#                        elif word_tag in ['SYM', '$', '#']:
#                            pos_label = 'SYM'
#                        elif word_tag in ['CD']:
#                            pos_label = 'NUM'
                        elif word_tag in ['IN', 'RP']:
                            pos_label = 'ADP'
                        elif word_tag in ['RB', 'RBR', 'RBS', 'WRB']:
                            pos_label = 'ADV'
#                        elif word_tag in []:
#                            pos_label = 'AUX'
                        elif word_tag in ['CC']:
                            pos_label = 'CCONJ'
                        elif word_tag in ['DT', 'PDT', 'PRP$', 'WDT', 'WP$']:
                            pos_label = 'DET'  
#                        elif word_tag in ['UH']:
#                            pos_label = 'INTJ'
#                        elif word_tag in ['POS', 'TO']:
#                            pos_label = 'PART'
                        elif word_tag in ['NNP', 'NNPS']:
                            pos_label = 'NOUN'
#                        elif word_tag in []:
#                            pos_label = 'SCONJ'
                        else:
                            pos_label = 'X'
                        labels.append(pos_label)
                
                else:
                    raise ValueError('Invalid label type')

                for idx, l in enumerate(l0_lists):
                    l.append(layer0[idx].detach().numpy())

                for idx, l in enumerate(l1_lists):
                    l.append(layer1[idx].detach().numpy())

    # Store data and labels
    print(Counter(labels))
    output_path_prefix = 'activations/{}/{}'.format(filepath[18:-4], mode)

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    for idx, l in enumerate(l0_lists):
        filename = '{}/{}_l0.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    for idx, l in enumerate(l1_lists):
        filename = '{}/{}_l1.pickle'.format(output_path_prefix,
                                            activation_names[idx])
        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    filename = '{}/labels.pickle'.format(output_path_prefix)
    with open(filename, 'wb') as f_out:
        pickle.dump(np.array(labels), f_out)

    return output_path_prefix


if __name__ == '__main__':

    extract_data('../balanced_datasets/fixed_context_size/lstm_mixed/context_size_6/neither_attractor_nor_helpful/datasize1000_split1_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_mixed_train.tsv', 'train', VOCAB_SIZE, HIDDEN_SIZE, 2, 1)

    extract_data('../balanced_datasets/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_correct_test.tsv', 'test', VOCAB_SIZE, HIDDEN_SIZE, 2, 1)

    extract_data('../balanced_datasets/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_wrong_test.tsv', 'test', VOCAB_SIZE, HIDDEN_SIZE, 2, 1)
