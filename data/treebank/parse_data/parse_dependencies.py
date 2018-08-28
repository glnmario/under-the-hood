# This file can be run as a python script after setting the paths below.
# It parses the dependency dataset and outputs a clean .tsv version thereof
# which includes: sentence, POS_sentence, number, subj_index, verb_index,
# n_helpful_nouns, and n_attractors.


# Paths
############################################################
dependency_dataset_path = '../../unparsed_wiki_agr_50_mostcommon_10K.tsv'
output_dataset_path = '../../parsed_subj_verb_dataset.tsv'
############################################################


def count_nouns(number, pos_sequence, sentence, w2i):
    """
    params:
        number: 'sing' or 'plur'
        pos_sequence: sequence of POS tags between subject and verb (non-inclusive)
        sentence: sequence of words between subject and verb (non-inclusive)
        w2i: word-to-index vocabulary
    """
    if number == 'sing':
        pos_tags = ['NN', 'NNP']
    else:
        pos_tags = ['NNS', 'NNPS']

    count_relevant_nouns = 0
    for i, pos_tag in enumerate(pos_sequence):
        if pos_tag in pos_tags and sentence[i] in w2i:
            count_relevant_nouns += 1

    return count_relevant_nouns


if __name__ == '__main__':

    with open('../../../models/vocab_hidden650_batch128_dropout0.2_lr20.0.txt') as f_in:
        vocab_lines = f_in.readlines()
    w2i = {}
    for i, line in enumerate(vocab_lines):
        w2i[line.strip()] = i


    with open(dependency_dataset_path, 'r') as f_in:
        lines = f_in.readlines()[1:]

    with open(output_dataset_path, 'w') as f_out:

        print('\t'.join(['sentence',
                        'pos_sentence',
                        'number',
                        'subj_index',
                        'verb_index',
                        'n_helpful_nouns',
                        'n_attractors']), file=f_out)

        for i, line in enumerate(lines):

            split_line = line.split('\t')
            sentence = split_line[1].split()
            pos_sentence = split_line[2].split()
            subj_index = int(split_line[9]) - 1
            verb_index = int(split_line[10]) - 1

            intervening_pos_seq = pos_sentence[subj_index + 1: verb_index]
            intervening_word_seq = sentence[subj_index + 1: verb_index]
            verb_pos = pos_sentence[verb_index]
            subj_pos = pos_sentence[subj_index]
            
            for j, word in enumerate(sentence):
                if word not in w2i:
                   sentence[j]  = '<unk>'

            if verb_pos == 'VBZ':
                number = 'sing'
                n_helpful_nouns = count_nouns('sing', intervening_pos_seq, intervening_word_seq, w2i)
                n_attractors = count_nouns('plur', intervening_pos_seq, intervening_word_seq, w2i)
            elif verb_pos == 'VBP':
                number = 'plur'
                n_helpful_nouns = count_nouns('plur', intervening_pos_seq, intervening_word_seq, w2i)
                n_attractors = count_nouns('sing', intervening_pos_seq, intervening_word_seq, w2i)
            else:
                continue

            fields = [' '.join(sentence),
                      ' '.join(pos_sentence),
                      number,
                      subj_index,
                      verb_index,
                      n_helpful_nouns,
                      n_attractors]

            print('\t'.join(list(map(str, fields))), file=f_out)
