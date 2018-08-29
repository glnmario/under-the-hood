from __future__ import division
import torch
import math
import random
import pickle
import sklearn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from lstm import Forward_LSTM
import argparse
from collections import defaultdict


def main(
    data_path,
    model_path,
    w2i_path,
    hidden_size,
    classifier_path=None,
    intervention=False,
    learning_rate=None,
    component_names=None,
    generate_labels=False):
    
    # load word-to-index vocabulary
    with open(w2i_path, 'r') as f:
        vocab_lines = f.readlines()

        w2i = {}
        for i, line in enumerate(vocab_lines):
            w2i[line.strip()] = i
        unk_idx = w2i['<unk>']


    vocab_size = len(w2i)

    # load and initialise model
    lstm = Forward_LSTM(vocab_size,
                        hidden_size,
                        hidden_size,
                        vocab_size,
                        w2i_path,
                        model_path)

    # initialise hidden state for time step -1
    # the hidden state will not be reset for each sentence
    relevant_activations = {}
    relevant_activations['hx_l0'] = torch.Tensor(torch.zeros(hidden_size))
    relevant_activations['cx_l0'] = torch.Tensor(torch.zeros(hidden_size))
    relevant_activations['hx_l1'] = torch.Tensor(torch.zeros(hidden_size))
    relevant_activations['cx_l1'] = torch.Tensor(torch.zeros(hidden_size))


    # load testing data
    with open(data_path, 'r') as f_in:
        test_set = f_in.readlines()[1:]


    # collect scores for all subcategories
    scores_original_nvv = []
    scores_nonce_nvv = []
    scores_original_vnpcv = []
    scores_nonce_vnpcv = []
    scores_original = []
    scores_nonce = []
    scores = []


    if intervention:
        classifiers = defaultdict()

        # load diagnostic classifiers for the intervention
        for act in component_names:
            with open("{}/{}.pickle".format(classifier_path, act), 'rb') as trained_classifier:
                classifiers[act] = pickle.load(trained_classifier)


    # process sentences
    for line_idx in range(0, len(test_set), 2):

        # read two consecutive lines with the following structure:
        # 0:pattern 1:constr_id 2:sent_id 3:correct_number 4:form 5:class 6:type
        # 7:prefix 8:n_attr 9:punct 10:freq 11:len_context 12:len_prefix 13:sent

        sent_data1 = test_set[line_idx].split('\t')
        sent_data2 = test_set[line_idx + 1].split('\t')

        # L__NOUN_VERB_VERB or R__VERB_NOUN_CCONJ_VERB
        pattern1 = sent_data1[0]
        pattern2 = sent_data2[0]

        assert(pattern1[0] in ['R', 'L'] and pattern2[0] in ['R', 'L'])
        assert(pattern1[0] == pattern2[0])
        construction_id = 0 if pattern1[0] == 'R' else 1

        assert(sent_data1[3] == sent_data2[3])
        if not generate_labels:
            label = 0 if sent_data1[3].strip() == 'sing' else 1

        assert(sent_data1[5] != sent_data2[5])
        if sent_data1[5] == 'correct':
            correct_form = sent_data1[4]
            wrong_form = sent_data2[4]
        else:
            correct_form = sent_data2[4]
            wrong_form = sent_data1[4]

        assert(sent_data1[6] == sent_data2[6])
        type_of_sent = sent_data1[6]

        assert(sent_data1[11] == sent_data2[11])
        context_length = int(sent_data1[11])

        assert(sent_data1[12] == sent_data2[12])
        target_idx = int(sent_data1[12])

        subject_idx = target_idx - context_length

        assert(sent_data1[13] == sent_data2[13])
        sentence = sent_data1[13].split()

        # process sentence
        for t, word in enumerate(sentence):
            output, layer0, layer1 = lstm(word,
                                          relevant_activations['hx_l0'],
                                          relevant_activations['cx_l0'],
                                          relevant_activations['hx_l1'],
                                          relevant_activations['cx_l1'])

            relevant_activations['hx_l0'] = layer0[0]
            relevant_activations['hx_l1'] = layer1[0]
            relevant_activations['cx_l0'] = layer0[1]
            relevant_activations['cx_l1'] = layer1[1]

            if t == target_idx - 1:
                vocab_probs = F.log_softmax(
                    output.view(-1, len(w2i)), dim=1)[0]

            # intervention at subject timestep
            if intervention and t == subject_idx:
                for act in component_names:
                    weight, bias = classifiers[act]
                    weight = Variable(torch.tensor(
                        weight, dtype=torch.double).squeeze(0), requires_grad=False)
                    bias = Variable(torch.tensor(
                        bias, dtype=torch.double), requires_grad=False)
                    current_activation = Variable(torch.tensor(
                        relevant_activations[act], dtype=torch.double), requires_grad=True)

                    total_prob = torch.tensor(1.0, dtype=torch.double)
                    class_1_prob = torch.dot(weight, current_activation) + bias
                    class_1_prob = F.sigmoid(class_1_prob)
                    class_0_prob = total_prob - class_1_prob

                    class_0_log_prob = torch.log(class_0_prob)
                    class_1_log_prob = torch.log(class_1_prob)

                    params = [current_activation]
                    optimiser = torch.optim.SGD(params, lr=learning_rate) 
                    optimiser.zero_grad()

                    prediction = (class_0_log_prob, class_1_log_prob)
                    prediction = torch.tensor(
                        torch.cat(prediction)).unsqueeze(0)

                    # unsupervised intervention requires generated labels
                    if generate_labels:
                        label = 0 if class_0_prob > class_1_prob else 1

                    gold_label = torch.tensor(label).unsqueeze(0)

                    criterion = nn.NLLLoss()
                    loss = criterion(prediction, gold_label)
                    loss.backward()
                    optimiser.step()

                    relevant_activations[act] = torch.tensor(current_activation, dtype=torch.float)


        correct_form_score = vocab_probs[w2i[correct_form]].data
        wrong_form_score = vocab_probs[w2i[wrong_form]].data

        if (correct_form_score > wrong_form_score).all():
            score = 1
        else:
            score = 0

        scores.append(score)

        if construction_id == 0 and type_of_sent == 'original':
            scores_original_vnpcv.append(score)
            scores_original.append(score)

        if construction_id == 1 and type_of_sent == 'original':
            scores_original_nvv.append(score)
            scores_original.append(score)

        if construction_id == 0 and type_of_sent == 'generated':
            scores_nonce_vnpcv.append(score)
            scores_nonce.append(score)

        if construction_id == 1 and type_of_sent == 'generated':
            scores_nonce_nvv.append(score)
            scores_nonce.append(score)

    assert(len(scores) == len(test_set) / 2)


    # Print accuracy results
    print ('Original   V NP Conj V   ', np.sum(
        scores_original_vnpcv) / len(scores_original_vnpcv))
    print ('Nonce      V NP Conj V   ', np.sum(
        scores_nonce_vnpcv) / len(scores_nonce_vnpcv))
    print ('Original   N V V         ', np.sum(
        scores_original_nvv) / len(scores_original_nvv))
    print ('Nonce      N V V         ', np.sum(
        scores_nonce_nvv) / len(scores_nonce_nvv))
    print ('Original   Overall       ', np.sum(
        scores_original) / len(scores_original))
    print ('Nonce      Overall       ', np.sum(scores_nonce) / len(scores_nonce))
    print ('Overall                  ', np.sum(scores) / len(scores))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='../data/colorlessgreenRNNs/agreement/generated.tab', help='The path of the Linzen validation set')
    parser.add_argument('--model', type=str, default='../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt', help='The path of the LSTM model')
    parser.add_argument('--w2i', type=str, default='../models/colorlessgreenRNNs/vocab_hidden650_batch128_dropout0.2_lr20.0.txt', help='The path of the word-to-index dictionary.')
    parser.add_argument('--nhid', type=int, default=650, help='Number of LSTM hidden dimensions')
    parser.add_argument('--lr', type=float, default=0.5, help='The backprop learning rate used for the intervention')
    # parser.add_argument('--classifier', type=str, default='trained-classifier/activations/no_fixed_context_size/lstm_mixed/context_size_3-15/neither_attractor_nor_helpful/datasize500_split1_attractor_positionNone_helpful_positionNone_words_before_subject0_words_after_verb0_mixed_remove_unk-True_train/train', help='The path of the diagnostic classifier to be used for unsupervised intervention')
    parser.add_argument('--classifier', type=str, default='../trained-classifiers/train', help='The path of the diagnostic classifier to be used for unsupervised intervention')
    parser.add_argument('--intervention', action='store_true', default=False, help='Whether to apply an intervention using diagnostic classifiers')
    parser.add_argument('--components', type=list, default=['hx_l0', 'hx_l1', 'cx_l0', 'cx_l1'], help='A list of names of the LSTM components on which the intervention should be applied')
    parser.add_argument('--unsupervised', action='store_true', help='Whether to use generated labels for the intervention')

    args = parser.parse_args()
    
    main(data_path=args.data, 
        model_path=args.model,
        w2i_path=args.w2i, 
        hidden_size=args.nhid,
        classifier_path=args.classifier,
        intervention=args.intervention,
        learning_rate=args.lr,
        component_names=args.components,
        generate_labels=args.unsupervised)
