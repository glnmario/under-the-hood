import os
import numpy as np
import pickle
from sklearn import linear_model, metrics, utils
from collections import defaultdict

# Parameters
###################################################################################

TRAIN_FILEPATH = '../../activations/fixed_context_size/lstm_mixed/context_size_6/neither_attractor_nor_helpful/datasize1000_split1_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_mixed_train/train'

TEST_FILEPATHS = ['../../activations/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_correct_test/test', '../../activations/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_wrong_test/test']

RESULTS_PATHS = ['../../results/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_correct_test/test', '../../results/fixed_context_size/lstm_correct_wrong/context_size_6/neither_attractor_nor_helpful/datasize200_split0_attractor_positionNone_helpful_positionNone_words_before_subject2_words_after_verb1_wrong_test/test']

MODE = 'regression'
CONTEXT_SIZE_LIST = [6, 6]
NUM_WORDS_BEFORE_SUBJECT = 2
NUM_WORDS_AFTER_VERB = 1
PRINT_PLOTTING_INFO = True
ACTIVATIONS = ["hx_l0", "cx_l0", "f_g_l0", "i_g_l0", "o_g_l0", "c_tilde_l0",
               "hx_l1", "cx_l1", "f_g_l1", "i_g_l1", "o_g_l1", "c_tilde_l1"]

###################################################################################

def train_and_store_classifier(train_filepath,
                               output_path,
                               num_time_steps,
                               activation_names
                               ):
    
    if os.path.exists(output_path):
        return output_path

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)
        
    for act in activation_names:
        print("--------------------------------------------------------------------------")
        print(act)
        print("--------------------------------------------------------------------------")
    
        with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
            data_ = pickle.load(f_train)
            
            
#        for time_step in range(num_time_steps):
#            train_time_step_slice = data_[time_step::num_time_steps, :]
#            train_labels_slice = train_labels_[time_step::num_time_steps]
        data, train_labels = data_, train_labels_
        data, train_labels = utils.shuffle(data, train_labels)
    
            # l2-regularised logistic regression with 10-fold cross-validation
        logreg = linear_model.LogisticRegressionCV()
        logreg.fit(data, train_labels)
            
        weights = logreg.coef_ 
            
        bias = logreg.intercept_   

        if not os.path.exists('{}/{}/'.format(output_path, act)):
            os.makedirs('{}/{}/'.format(output_path, act))
        
        with open('{}/{}.pickle'.format(output_path, act), 'wb') as f_out:
            pickle.dump([weights, bias], f_out)
    
    return output_path


def run_diagnostic_experiment(train_filepath,
                              test_filepaths,
                              results_paths,
                              type_of_experiment,
                              mode,
                              context_size_list,
                              num_words_before_subject,
                              num_words_after_verb,
                              activation_names,
                              print_plotting_info,
                              ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):

        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)
        # params_activations = defaultdict(list)
        for act in activation_names:
            print("--------------------------------------------------------------------------")
            print(act)
            print("--------------------------------------------------------------------------")

          # Training

            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                data = pickle.load(f_train)

          # Load labels at each iteration to ensure that shuffling is done correctly
                train_labels = train_labels_
                data, train_labels = utils.shuffle(data, train_labels)

            if mode == 'classification':
                # l2-regularised logistic regression with 10-fold cross-validation
                logreg = linear_model.LogisticRegressionCV()
                logreg.fit(data, train_labels)
            elif mode == 'regression':
                # Ridge regression with generalised cross-validation
                logreg = linear_model.RidgeCV()
                logreg.fit(data, train_labels)

                # params_activations[act] = logreg.coef_

            else:
                raise ValueError('Mode does not exist.')

            with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                test_data = pickle.load(f_test)

                activation_accuracy = logreg.score(test_data, test_labels)
                print("Overall accuracy for component {} was {}".format(act, activation_accuracy))
                print("--------------------------------------------------------------------------")

          # Testing: we obtain scores
          # for sentences where the distance between subject and verb ranges from 0 to MAX_CONTEXT_SIZE
          # and for each time step starting from 2 positions before the subject up to the verb

            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
            all_scores = []
            for time_step in range(num_time_steps):
                time_step_slice = test_data[time_step::num_time_steps, :]
                labels_slice = test_labels[time_step::num_time_steps]

                if mode == 'classification':
                    accuracy = logreg.score(time_step_slice, labels_slice)
                    predictions = logreg.predict(time_step_slice)
                    scores = np.equal(labels_slice, predictions).astype(int)
                else:
                    scores = []
                    preds = logreg.predict(time_step_slice)
                    scores = ((preds - labels_slice) ** 2)

                if print_plotting_info:

                    scores = list(scores)
                    plotting_info['scores'].extend(scores)
                    plotting_info['timestep'].extend([time_step] * len(scores))
                    plotting_info['activations'].extend([act[:-3]] * len(scores))
                    plotting_info['layers'].extend([act[-2:]] * len(scores))
                    all_scores.extend(np.round(scores, 3))

                if mode == 'classification':
                    print("Time step {:2}   accuracy: {:06.3f}".format(time_step, accuracy * 100))
                else:
                    print("Time step {:2}   R^2: {:06.4f}".format(time_step, np.mean(scores)))

            if print_plotting_info:
                plotting_info['mean_score'].append(np.mean(scores))

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

        # for key in params_activations.keys():
        #     for key_ in params_activations.keys():
        #         score = np.equal(params_activations[key], params_activations[key_]).sum()
        #         zeros = params_activations[key] == 0
        #         zeros_ = params_activations[key_] == 0
        #         zeros = zeros.sum()
        #         zeros_ = zeros_.sum()

        #         print(key, key_, score, zeros, zeros_)

    return plotting_locations

def run_diagnostic_experiment_timestep_training(train_filepath,
                              test_filepaths,
                              results_paths,
                              type_of_experiment,
                              mode,
                              context_size_list,
                              num_words_before_subject,
                              num_words_after_verb,
                              activation_names,
                              print_plotting_info,
                              ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):

        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)
        params_activations = defaultdict(list)
        for act in activation_names:
            print("--------------------------------------------------------------------------")
            print(act)
            print("--------------------------------------------------------------------------")

          # Training

            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                data_ = pickle.load(f_train)
                
            with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                test_data = pickle.load(f_test)

          # Load labels at each iteration to ensure that shuffling is done correctly
                
            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
#            all_scores = []
            accuracies = []
            for time_step in range(num_time_steps):
                train_time_step_slice = data_[time_step::num_time_steps, :]
                train_labels_slice = train_labels_[time_step::num_time_steps]
                
                data, train_labels = utils.shuffle(train_time_step_slice, train_labels_slice)

                if mode == 'classification':
                    # l2-regularised logistic regression with 10-fold cross-validation
                    logreg = linear_model.LogisticRegressionCV()
                    logreg.fit(data, train_labels)
                elif mode == 'regression':
                    # Ridge regression with generalised cross-validation
                    logreg = linear_model.RidgeCV()
                    logreg.fit(data, train_labels)
    
                else:
                    raise ValueError('Mode does not exist.')
    

          # Testing: we obtain scores
          # for sentences where the distance between subject and verb ranges from 0 to MAX_CONTEXT_SIZE
          # and for each time step starting from 2 positions before the subject up to the verb

                time_step_slice = test_data[time_step::num_time_steps, :]
                labels_slice = test_labels[time_step::num_time_steps]

                if mode == 'classification':
                    accuracy = logreg.score(time_step_slice, labels_slice)
                    predictions = logreg.predict(time_step_slice)
                else:
                    scores = []
                    preds = logreg.predict(time_step_slice)
                    scores = ((preds - labels_slice) ** 2)
                    
                if print_plotting_info:
                    accuracies.append(accuracy)
                    plotting_info['scores'].append(accuracy)
                    plotting_info['timestep'].append(time_step)
                    plotting_info['activations'].append(act[:-3])
                    plotting_info['layers'].append(act[-2:])
#                    all_scores.extend(np.round(scores, 3))

                if mode == 'classification':
                    print("Time step {:2}   accuracy: {:06.3f}".format(time_step, accuracy * 100))
                else:
                    print("Time step {:2}   R^2: {:06.4f}".format(time_step, np.mean(scores)))

            if print_plotting_info:
                mean_scores = np.mean(accuracies)
                print("Average score for component {} was {}".format(act, mean_scores))
                plotting_info['mean_score'].append(np.round(mean_scores, 3))

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

        for key in params_activations.keys():
            for key_ in params_activations.keys():
                score = np.equal(params_activations[key], params_activations[key_]).sum()
                zeros = params_activations[key] == 0
                zeros_ = params_activations[key_] == 0
                zeros = zeros.sum()
                zeros_ = zeros_.sum()

                print(key, key_, score, zeros, zeros_)

    return plotting_locations

def run_diagnostic_experiment_prediction_training(train_filepath,
                              test_filepaths,
                              results_paths,
                              type_of_experiment,
                              mode,
                              context_size_list,
                              num_words_before_subject,
                              num_words_after_verb,
                              activation_names,
                              print_plotting_info,
                              ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):

        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)
        params_activations = defaultdict(list)
        for act in activation_names:
            print("--------------------------------------------------------------------------")
            print(act)
            print("--------------------------------------------------------------------------")

          # Training

            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                data_ = pickle.load(f_train)
                
            with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                test_data = pickle.load(f_test)

          # Load labels at each iteration to ensure that shuffling is done correctly
                
            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
#            all_scores = []
            accuracies = []
            for time_step in range(num_time_steps):
                train_time_step_slice = data_[time_step::num_time_steps, :]
                train_labels_slice = train_labels_[time_step::num_time_steps]
                
                data, train_labels = utils.shuffle(train_time_step_slice, train_labels_slice)

                if mode == 'classification':
                    # l2-regularised logistic regression with 10-fold cross-validation
                    logreg = linear_model.LogisticRegressionCV()
                    logreg.fit(data, train_labels)
                elif mode == 'regression':
                    # Ridge regression with generalised cross-validation
                    logreg = linear_model.RidgeCV()
                    logreg.fit(data, train_labels)
    
                else:
                    raise ValueError('Mode does not exist.')
    

          # Testing: we obtain scores
          # for sentences where the distance between subject and verb ranges from 0 to MAX_CONTEXT_SIZE
          # and for each time step starting from 2 positions before the subject up to the verb

                time_step_slice = test_data[time_step::num_time_steps, :]
                labels_slice = test_labels[time_step::num_time_steps]

                if mode == 'classification':
                    predictions = logreg.predict_proba(time_step_slice)
                    plural_idx = [x for x in range(len(predictions)) if labels_slice[x] == 1]
                    singular_idx = [x for x in range(len(predictions)) if labels_slice[x] == 0]
                    
                    sing_predictions = np.mean(predictions[singular_idx, 0])
                    plur_predictions = np.mean(predictions[plural_idx, 0])
                    
                if print_plotting_info:
                    plotting_info['predictions'].extend([sing_predictions, plur_predictions])
                    plotting_info['timestep'].extend([time_step]*2)
                    plotting_info['activations'].extend([act[:-3]]*2)
                    plotting_info['layers'].extend([act[-2:]]*2)
                    
                print("component-{}-{}-{}-timestep-{}".format(act, np.mean(sing_predictions), np.mean(plur_predictions), time_step))

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

        for key in params_activations.keys():
            for key_ in params_activations.keys():
                score = np.equal(params_activations[key], params_activations[key_]).sum()
                zeros = params_activations[key] == 0
                zeros_ = params_activations[key_] == 0
                zeros = zeros.sum()
                zeros_ = zeros_.sum()

                print(key, key_, score, zeros, zeros_)

    return plotting_locations


def diagnostic_experiment_sentence_level(train_filepath,
                                         test_filepaths,
                                         results_paths,
                                         type_of_experiment,
                                         mode,
                                         context_size_list,
                                         num_words_before_subject,
                                         num_words_after_verb,
                                         activation_names,
                                         balanced_dataset_path
                                         ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    for idx, path in enumerate(test_filepaths):

        with open(balanced_dataset_path[idx]) as corpus_f:
            lines = corpus_f.readlines()

        print(path)

        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        for act in activation_names:
            print("--------------------------------------------------------------------------")
            print(act)
            print("--------------------------------------------------------------------------")

          # Training

            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                data = pickle.load(f_train)

          # Load labels at each iteration to ensure that shuffling is done correctly
                train_labels = train_labels_
                data, train_labels = utils.shuffle(data, train_labels)

            if mode == 'classification':
                # l2-regularised logistic regression with 10-fold cross-validation
                logreg = linear_model.LogisticRegressionCV()
                logreg.fit(data, train_labels)
            elif mode == 'regression':
                # Ridge regression with generalised cross-validation
                logreg = linear_model.RidgeCV()
                logreg.fit(data, train_labels)

                # params_activations[act] = logreg.coef_

            else:
                raise ValueError('Mode does not exist.')

            with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                test_data = pickle.load(f_test)

          # Testing: we obtain scores
          # for sentences where the distance between subject and verb ranges from 0 to MAX_CONTEXT_SIZE
          # and for each time step starting from 2 positions before the subject up to the verb

            num_time_steps = context_size_list[0] + 2 + num_words_before_subject + num_words_after_verb

            for i in range(len(lines)):

                test_sentence = test_data[i * num_time_steps:(i + 1) * num_time_steps, :]
                test_sentence_labels = test_labels[i * num_time_steps:(i + 1) * num_time_steps]

                predictions = logreg.predict_proba(test_sentence)

                split_line = lines[i].split('\t')
                sent = split_line[0].split()
                label = 0 if split_line[2] == 'sing' else 1
                subj_idx = int(split_line[3])
                verb_idx = int(split_line[4])

                for idx, w in enumerate(sent):

                    # if w not in w2i:
                    #    w = '<unk>'

                    if idx >= subj_idx - 1 and idx <= verb_idx + 1:
                        print('{:15}  {:6}'.format(w, str(predictions[idx - subj_idx + 1])))
                    else:
                        print('{:15}'.format(w))
                print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------")


def run_multi_class_diagnostic_experiment(train_filepath,
                                          test_filepaths,
                                          results_paths,
                                          type_of_experiment,
                                          mode,
                                          context_size_list,
                                          num_words_before_subject,
                                          num_words_after_verb,
                                          activation_names,
                                          print_plotting_info,
                                          attractor_idx
                                          ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):

        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)

        attractor_errors = dict()
        for act in activation_names:
            print("--------------------------------------------------------------------------")
            print(act)
            print("--------------------------------------------------------------------------")

          # Training

            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                data = pickle.load(f_train)

          # Load labels at each iteration to ensure that shuffling is done correctly
                train_labels = train_labels_
                data, train_labels = utils.shuffle(data, train_labels)

            if mode == 'classification':
                # l2-regularised logistic regression with 10-fold cross-validation
                logreg = linear_model.LogisticRegressionCV(multi_class='multinomial')
                logreg.fit(data, train_labels)
            elif mode == 'regression':
                # Ridge regression with generalised cross-validation
                logreg = linear_model.RidgeCV()
                logreg.fit(data, train_labels)
                params = logreg.get_params(deep=True)
                print('PARAMS:', params)
            else:
                raise ValueError('Mode does not exist.')

            with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                test_data = pickle.load(f_test)


#                predictions = logreg.predict(test_data)
#                print(predictions)

                activation_accuracy = logreg.score(test_data, test_labels)
                print("Overall accuracy for component {} was {}".format(act, activation_accuracy))
                print("--------------------------------------------------------------------------")

          # Testing: we obtain scores
          # for sentences where the distance between subject and verb ranges from 0 to MAX_CONTEXT_SIZE
          # and for each time step starting from 2 positions before the subject up to the verb

            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
            all_scores = []
            for time_step in range(num_time_steps):
                time_step_slice = test_data[time_step::num_time_steps, :]
                labels_slice = test_labels[time_step::num_time_steps]

                if mode == 'classification':
                    scores = logreg.score(time_step_slice, labels_slice).item()
                    predictions = logreg.predict(time_step_slice)

                    print("Predictions for Activation {} and timestep {} were as follows:{}".format(act, time_step, Counter(predictions)))
                    if time_step == num_words_before_subject + attractor_idx:
                        attractor_errors[act] = (Counter(predictions)[num_words_before_subject], len(labels_slice))

                else:
                    preds = logreg.predict(time_step_slice)
                    scores = np.mean(((preds - labels_slice) ** 2))

                if print_plotting_info:

                    scores = [scores]
                    plotting_info['scores'].extend(scores)
                    plotting_info['timestep'].extend([time_step])
                    plotting_info['activations'].extend([act[:-3]])
                    plotting_info['layers'].extend([act[-2:]])
                    all_scores.extend(np.round(scores, 3))

                if mode == 'classification':
                    print("Time step {:2}   accuracy: {:06.3f}".format(time_step, scores[0] * 100))
                else:
                    print("Time step {:2}   R^2: {:06.4f}".format(time_step, np.mean(scores)))

            if print_plotting_info:
                plotting_info['mean_score'].append(np.mean(scores))

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

        print(attractor_errors)

    return plotting_locations


def run_diagnostic_experiment_timestep_wise(train_filepath,
                                            test_filepaths,
                                            results_paths,
                                            type_of_experiment,
                                            mode,
                                            context_size_list,
                                            num_words_before_subject,
                                            num_words_after_verb,
                                            activation_names,
                                            print_plotting_info
                                            ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):
        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)

        for act in activation_names:
            print("here")
            with open('{}/{}.pickle'.format(train_filepath, act), 'rb') as f_train:
                train_data = pickle.load(f_train)

            train_labels = train_labels_
            # data, train_labels = utils.shuffle(data, train_labels)

            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
            for time_step_train in range(num_time_steps):
                time_step_slice_train = train_data[time_step_train::num_time_steps, :]
                labels_slice_train = train_labels[time_step_train::num_time_steps]

                if mode == 'classification':
                    logreg = linear_model.LogisticRegressionCV()
                    logreg.fit(time_step_slice_train, labels_slice_train)
                elif mode == 'regression':
                    # Ridge regression with generalised cross-validation
                    logreg = linear_model.RidgeCV()
                    logreg.fit(time_step_slice_train, labels_slice_train)
                else:
                    raise ValueError('Mode does not exist.')

                with open('{}/{}.pickle'.format(path, act), 'rb') as f_test:
                    test_data = pickle.load(f_test)

                for time_step_test in range(num_time_steps):
                    time_step_slice_test = test_data[time_step_test::num_time_steps, :]
                    labels_slice_test = test_labels[time_step_test::num_time_steps]

                    if mode == 'classification':
                        mean_score = logreg.score(time_step_slice_test, labels_slice_test)
                        predictions = logreg.predict(time_step_slice_test)

                    else:
                        preds = logreg.predict(time_step_slice_test)
                        mean_score = np.mean((preds - labels_slice_test) ** 2)

                    if print_plotting_info:
                        plotting_info['scores'].append(mean_score)
                        plotting_info['timestep_train'].append(time_step_train)
                        plotting_info['timestep_test'].append(time_step_test)
                        plotting_info['activations'].append(act[:-3])
                        plotting_info['layers'].append(act[-2:])

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

    return plotting_locations


def run_diagnostic_experiment_component_wise(train_filepath,
                                             test_filepaths,
                                             results_paths,
                                             type_of_experiment,
                                             mode,
                                             context_size_list,
                                             num_words_before_subject,
                                             num_words_after_verb,
                                             activation_names,
                                             print_plotting_info
                                             ):

    with open('{}/labels.pickle'.format(train_filepath), 'rb') as f_in:
        train_labels_ = pickle.load(f_in)

    plotting_locations = []
    for idx, path in enumerate(test_filepaths):
        print(path)
        print("--------------------------------------------------------------------------")

        with open('{}/labels.pickle'.format(path), 'rb') as f_in:
            test_labels_ = pickle.load(f_in)

        # Diagnostic classification:
        # for each model component under scrutiny, a classifier is trained and tested

        if print_plotting_info:
            plotting_info = defaultdict(list)

        num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
        for time_step in range(num_time_steps):

            for act_train in activation_names:

                with open('{}/{}.pickle'.format(train_filepath, act_train), 'rb') as f_train:
                    train_data_ = pickle.load(f_train)

                train_data = train_data_[time_step::num_time_steps, :]
                train_labels = train_labels_[time_step::num_time_steps]

                train_data, train_labels = utils.shuffle(train_data, train_labels)

                if mode == 'classification':
                    logreg = linear_model.LogisticRegressionCV()
                    logreg.fit(train_data, train_labels)
                elif mode == 'regression':
                        # Ridge regression with generalised cross-validation
                    logreg = linear_model.RidgeCV()
                    logreg.fit(train_data, train_labels)
                else:
                    raise ValueError('Mode does not exist.')

    #            num_time_steps = context_size_list[idx] + 2 + num_words_before_subject + num_words_after_verb
                for act_test in activation_names:

                    with open('{}/{}.pickle'.format(path, act_test), 'rb') as f_test:
                        test_data_ = pickle.load(f_test)

                    test_data = test_data_[time_step::num_time_steps, :]
                    test_labels = test_labels_[time_step::num_time_steps]

                    if mode == 'classification':
                        mean_score = logreg.score(test_data, test_labels)
                        predictions = logreg.predict(test_data)
                    else:
                        preds = logreg.predict(test_data)
                        mean_score = np.mean((preds - test_labels) ** 2)

                    if print_plotting_info:
                        plotting_info['scores'].append(mean_score)
                        plotting_info['train_act'].append(act_train)
                        plotting_info['test_act'].append(act_test)
                        plotting_info['timestep'].append(time_step)

        if print_plotting_info:
            for list_ in plotting_info.values():
                list_ = np.array(list_)

        output_path = results_paths[idx] + '/' + mode

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open('{}/plotting_dict_component_wise.pickle'.format(output_path, mode), 'wb') as f_dict:
            pickle.dump(plotting_info, f_dict)

        plotting_locations.append('{}/plotting_dict_component_wise.pickle'.format(output_path))
        print("--------------------------------------------------------------------------")

    return plotting_locations


if __name__ == '__main__':

    run_diagnostic_experiment(TRAIN_FILEPATH,
                              TEST_FILEPATHS,
                              RESULTS_PATHS,
                              None,
                              MODE,
                              CONTEXT_SIZE_LIST,
                              NUM_WORDS_BEFORE_SUBJECT,
                              NUM_WORDS_AFTER_VERB,
                              ACTIVATIONS,
                              PRINT_PLOTTING_INFO
                              )
