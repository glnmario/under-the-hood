import torch
import math
import os
import random
import torch.nn.functional as F

# Parameters
##################################################################################

dataset_path_ = '../data/treebank/parsed_subj_verb_dataset.tsv'
w2i_dict_path_ = '../models/colorlessgreenRNNs/vocab_hidden650_batch128_dropout0.2_lr20.0.txt'
model_path_ = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
##################################################################################


def generate_wrong_form(w2i, correct_form, correct_num, wrong2correct, correct2wrong):

    if correct_form in wrong2correct:
        wrong_form = wrong2correct[correct_form]
    elif correct_form in correct2wrong:
        wrong_form = correct2wrong[correct_form]
    else:
        if correct_num == 'sing':
            wrong_form = correct_form[:-1]
            if wrong_form not in w2i:
                wrong_form = correct_form[:-2]
                if wrong_form not in w2i:
                    print(correct_form)
                    return ''
        else:
            wrong_form = correct_form + 's'
            if wrong_form not in w2i:
                wrong_form = correct_form + 'es'
                if wrong_form not in w2i:
                    print(correct_form)
                    return ''

    return wrong_form


def create_dataset(lstm_mode,
                   fixed_context,
                   context_size,
                   max_class_size,
                   split,
                   num_attractors,
                   num_helpful_nouns,
                   attractor_idx,
                   helpful_noun_idx,
                   num_words_before_subject,
                   num_words_after_verb,
                   dont_overlap_with_datasets,
                   dataset_path,
                   w2i_dict_path,
                   model_path,
                   remove_unk):
    """
    params:
        lstm_mode: correct, wrong, or mixed
        fixed_context: true or false
        context_size: context length or range of context lengths (int or pair)
        max_class_size: size of the total data
        split: proportion of train data points in total data
        num_attractors: number of agreement attractors between subject and verb
        num_helpful_noun: number of nouns between subject and verb having the same number as the subject
        attractor_idx: number of words between subject and attractor
        helpful_noun_idx: number of words between subject and helpful noun
        num_words_before_subject: filter out sentences with fewer than num_words_before_subject words before the subject
        num_words_after_verb: filter out sentences with fewer than num_words_after_verb words after the verb
        dont_overlap_with_datasets: list of paths the new dataset should not overlap with
    """
    if num_attractors is None and attractor_idx is not None:
        raise ValueError('No attractors to index')

    if num_helpful_nouns is None and helpful_noun_idx is not None:
        raise ValueError('No attractors to index')

    if attractor_idx is not None and attractor_idx == helpful_noun_idx:
        raise ValueError('The same word cannot be both an attractor and a helpful noun.')

    if attractor_idx is not None and type(context_size) == int and attractor_idx >= context_size - 1:
        raise ValueError('Attractor must be between subject and verb.')

    if helpful_noun_idx is not None and type(context_size) == int and helpful_noun_idx >= context_size - 1:
        raise ValueError('Helpful noun must be between subject and verb.')

    with open(dataset_path, 'r') as f_in:
        dataset = f_in.readlines()[1:]

    if dont_overlap_with_datasets is not None:
        dataset_overlap = []

        for dont_overlap_with_dataset in dont_overlap_with_datasets:
            with open(dont_overlap_with_dataset, 'r') as f_in:
                dataset_overlap.extend(f_in.readlines())

    # Load the pretrained model
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location='cpu')
    # and its vocabulary
    with open(w2i_dict_path, 'r') as f_in:
        vocab_lines = f_in.readlines()

    w2i = {}
    for i, line in enumerate(vocab_lines):
        w2i[line.strip()] = i
    unk_idx = w2i['<unk>']

    if lstm_mode != 'mixed':
        with open('../data/linzen-testset/subj_agr_filtered.gold', 'r') as f:
            lines = f.readlines()[1:]

        correct2wrong = {}
        for line in lines:
            line = line.split('\t')
            if line[1] == line[2]:
                # print(line[1:3])
                continue
            correct2wrong[line[1]] = line[2]

        wrong2correct = {w: c for c, w in correct2wrong.items()}

    if fixed_context:
        if context_size < 0:
            raise ValueError('Context size cannot be negative.')

        context_sizes = [context_size]
        context_size_label = context_size
    else:
        if context_size[0] < 0:
            raise ValueError('Context size cannot be negative.')

        context_sizes = range(context_size[0], context_size[1])
        context_size_label = '{}-{}'.format(*context_size)

    fixed_context_label = 'fixed_context_size' if fixed_context else 'no_fixed_context_size'
    lstm_mode_label = 'lstm_mixed' if lstm_mode == 'mixed' else 'lstm_correct_wrong'

    if num_attractors is None and num_helpful_nouns is None:
        intervening_nouns_label = 'neither_attractor_nor_helpful'

    elif (num_attractors is None or num_attractors == 0) and (num_helpful_nouns is not None and num_helpful_nouns > 0):
        intervening_nouns_label = 'helpful_only'

    elif (num_helpful_nouns == None or num_helpful_nouns == 0) and (num_attractors is not None and num_attractors > 0):
        intervening_nouns_label = 'attractors_only'

    elif num_attractors > 0 and num_helpful_nouns > 0:
        intervening_nouns_label = 'attractors_and_helpful'

    else:
        intervening_nouns_label = 'neither_attractor_nor_helpful'

    output_folder = 'balanced_datasets/{}/{}/context_size_{}/{}'.format(fixed_context_label,
                                                                        lstm_mode_label,
                                                                        context_size_label,
                                                                        intervening_nouns_label)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = '/datasize{}_split{}_attractor_position{}_helpful_position{}_words_before_subject{}_words_after_verb{}_{}_remove_unk-{}_'.format(max_class_size,
                                                                                                                                         split,
                                                                                                                                         attractor_idx,
                                                                                                                                         helpful_noun_idx,
                                                                                                                                         num_words_before_subject,
                                                                                                                                         num_words_after_verb,
                                                                                                                                         lstm_mode,
                                                                                                                                         remove_unk)

    output_path = output_folder + output_filename

    if split < 1 and split > 0:
        exist_path = output_path + 'train.tsv'
    elif split == 1:
        exist_path = output_path + 'train.tsv'
    elif split == 0:
        exist_path = output_path + 'test.tsv'

    if os.path.isfile(exist_path):
        if split == 1:
            return exist_path, ''
        elif split == 0:
            return '', exist_path

    if split < 1 and split > 0:
        f_test = open(output_path + 'test.tsv', 'w')
        f_train = open(output_path + 'train.tsv', 'w')
    elif split == 1:
        f_train = open(output_path + 'train.tsv', 'w')
    elif split == 0:
        f_test = open(output_path + 'test.tsv', 'w')
    else:
        raise ValueError('Not a valid split.')

    if lstm_mode != 'mixed':
        model.eval()
        hidden = model.init_hidden(1)

    for cs in context_sizes:

        singular_count_train = 0
        singular_count_test = 0
        class_size = 1

        for i, line in enumerate(dataset):

            if dont_overlap_with_datasets and line in dataset_overlap:
                continue

            # sentence, POS_sentence, number, subj_index, verb_index, n_helpful_nouns, and n_attractors.

            line_list = line.split('\t')
            sentence = line_list[0].split()
            pos_sentence = line_list[1].split()
            label = line_list[2].strip()
            subj_idx = int(line_list[3])
            verb_idx = int(line_list[4])
            n_helpful_nouns = int(line_list[5])
            n_attractors = int(line_list[6])

            num_intervening_words = verb_idx - subj_idx - 1

            if num_intervening_words != cs:
                continue

            if num_attractors is not None and n_attractors != num_attractors:
                continue

            if num_helpful_nouns is not None and n_helpful_nouns != num_helpful_nouns:
                continue

            if attractor_idx is not None:
                if num_attractors == 0:
                    raise ValueError('If the number of attractors equals 0 the position of the attractors must be None.')
                    if split > 0:
                        f_train.close()
                    if split < 1:
                        f_test.close()
                if sentence[subj_idx + attractor_idx] not in w2i or sentence[subj_idx + attractor_idx] == '<unk>':
                    continue

            if helpful_noun_idx is not None:
                if num_helpful_nouns == 0:
                    raise ValueError('If the number of helpful nouns equals 0 the position of the helpful noun must be None.')
                    if split > 0:
                        f_train.close()
                    if split < 1:
                        f_test.close()
                if sentence[subj_idx + helpful_noun_idx] not in w2i or sentence[subj_idx + helpful_noun_idx] == '<unk>':
                    continue

            if sentence[subj_idx] not in w2i or sentence[verb_idx] not in w2i or sentence[subj_idx] == '<unk>' or sentence[verb_idx] == '<unk>':
                continue
            
            if remove_unk:
                continuing = False
                for word in sentence[subj_idx:verb_idx+1]:
                    if word not in w2i or word == '<unk>':
                        continuing = True
                
                if continuing:
                    continue

            if subj_idx < num_words_before_subject or verb_idx >= len(sentence) - num_words_after_verb:
                continue

            if attractor_idx is not None:
                candidate_attractor_pos = pos_sentence[subj_idx + attractor_idx]
                if candidate_attractor_pos in ['NN', 'NNP']:
                    if label == 'sing':
                        continue
                elif candidate_attractor_pos in ['NNS', 'NNPS']:
                    if label == 'plur':
                        continue
                else:
                    continue

            if helpful_noun_idx is not None:
                candidate_helpful_noun_pos = pos_sentence[subj_idx + helpful_noun_idx]
                if candidate_helpful_noun_pos in ['NN', 'NNP']:
                    if label != 'sing':
                        continue
                elif candidate_helpful_noun_pos in ['NNS', 'NNPS']:
                    if label != 'plur':
                        continue
                else:
                    continue

            if lstm_mode != 'mixed':
                correct_form = sentence[verb_idx]

                wrong_form = generate_wrong_form(w2i, correct_form, label, wrong2correct, correct2wrong)
                
                if wrong_form == '':
                    continue

                sentence_ids = [w2i[w] if w in w2i else unk_idx for w in sentence]
                sentence_ids = torch.tensor(sentence_ids).unsqueeze(1)

                output, hidden = model(sentence_ids, hidden)
                softmax_output = F.log_softmax(output.view(-1, len(w2i)), dim=-1)

                vocab_probs = softmax_output[verb_idx - 1, :]
                correct_form_score = vocab_probs[w2i[correct_form]].item()
                wrong_form_score = vocab_probs[w2i[wrong_form]].item()


                if lstm_mode == 'correct':
                    if correct_form_score < wrong_form_score:
                        continue
                else:
                    if correct_form_score > wrong_form_score:
                        continue

            if split == 1:
                test_set_interval = 100000000
            elif split == 0:
                test_set_interval = 1
            else:
                test_set_interval = math.floor(1 / (1 - split))

            # Add to test set
            if class_size % test_set_interval == 0:

                if label == 'sing':
                    if singular_count_test < max_class_size / 2 * (1 - split):
                        print(line, file=f_test, end="")
                        singular_count_test += 1
                        class_size += 1
                else:
                    print(line, file=f_test, end="")
                    class_size += 1

            # Add to training set
            else:
                if label == 'sing':
                    if singular_count_train < max_class_size / 2 * split:
                        print(line, file=f_train, end="")
                        singular_count_train += 1
                        class_size += 1
                else:
                    print(line, file=f_train, end="")
                    class_size += 1

            if class_size - 1 == max_class_size:
                break

        print('Generated {} out of {} for context size {}'.format(class_size - 1, max_class_size, cs))

    if split > 0:
        f_train.close()
    if split < 1:
        f_test.close()

    if split < 1 and split > 0:
        return output_path + 'train.tsv', output_path + 'test.tsv'
    elif split == 1:
        return output_path + 'train.tsv', ""
    elif split == 0:
        return "", output_path + 'test.tsv'


if __name__ == '__main__':

    print(
        _dataset(lstm_mode='mixed',
                 fixed_context=True,
                 context_size=6,
                 max_class_size=1000,
                 split=1,
                 num_attractors=None,
                 num_helpful_nouns=None,
                 attractor_idx=None,
                 helpful_noun_idx=None,
                 num_words_before_subject=2,
                 num_words_after_verb=1,
                 dont_overlap_with_datasets=['../balanced_datasets/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_correct_test.tsv', '../balanced_datasets/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_wrong_test.tsv'],
                 dataset_path=dataset_path_,
                 w2i_dict_path=w2i_dict_path_,
                 model_path=model_path_))
