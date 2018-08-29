import create_balanced_datasets.data_creator_subj_verb as data_creator
from create_activation_data import create_activation_data
from diagnostic_classifiers import diagnostic_experiment
import plotting

DATASET_PATH = '../data/treebank/parsed_subj_verb_dataset.tsv'
W2I_DICT_PATH = '../models/colorlessgreenRNNs/vocab_hidden650_batch128_dropout0.2_lr20.0.txt'
MODEL_PATH = '../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt'
VOCAB_SIZE = 50001
HIDDEN_SIZE = 650
NUM_WORDS_BEFORE_SUBJECT = 0
NUM_WORDS_AFTER_VERB = 0

MODE = 'classification'
LABELS_TYPE = 'training_normal'
CONTEXT_SIZE_LIST = [5, 5]
PRINT_PLOTTING_INFO = True
ACTIVATIONS = ["hx_l0", "cx_l0", "f_g_l0", "i_g_l0", "o_g_l0", "c_tilde_l0",
              "hx_l1", "cx_l1", "f_g_l1", "i_g_l1", "o_g_l1", "c_tilde_l1"]
NUM_ATTRACTORS = 1
NUM_HELPFUL_NOUNS = 0
ATTRACTOR_IDX = 3
HELPFUL_NOUN_IDX = None
PLOTTING_MODE = 'graph'
REMOVE_UNK = True
MODIFICATION = True

if __name__ == '__main__':

    num_timesteps = CONTEXT_SIZE_LIST[0] + 2 + NUM_WORDS_BEFORE_SUBJECT + NUM_WORDS_AFTER_VERB
    
    _, test_set_correct = data_creator.create_dataset(lstm_mode='correct',
                                                        fixed_context=True,
                                                        context_size=CONTEXT_SIZE_LIST[0],
                                                        max_class_size=100,
                                                        split=0,
                                                        num_attractors=NUM_ATTRACTORS,
                                                        num_helpful_nouns=NUM_HELPFUL_NOUNS,
                                                        attractor_idx=ATTRACTOR_IDX,
                                                        helpful_noun_idx=HELPFUL_NOUN_IDX,
                                                        num_words_before_subject=NUM_WORDS_BEFORE_SUBJECT,
                                                        num_words_after_verb=NUM_WORDS_AFTER_VERB,
                                                        dont_overlap_with_datasets=[],
                                                        dataset_path=DATASET_PATH,
                                                        w2i_dict_path=W2I_DICT_PATH,
                                                        model_path=MODEL_PATH,
                                                        remove_unk=REMOVE_UNK)
    
    _, test_set_wrong = data_creator.create_dataset(lstm_mode='wrong',
                                                      fixed_context=True,
                                                      context_size=CONTEXT_SIZE_LIST[1],
                                                      max_class_size=100,
                                                      split=0,
                                                      num_attractors=NUM_ATTRACTORS,
                                                      num_helpful_nouns=NUM_HELPFUL_NOUNS,
                                                      attractor_idx=ATTRACTOR_IDX,
                                                      helpful_noun_idx=HELPFUL_NOUN_IDX,
                                                      num_words_before_subject=NUM_WORDS_BEFORE_SUBJECT,
                                                      num_words_after_verb=NUM_WORDS_AFTER_VERB,
                                                      dont_overlap_with_datasets=[],
                                                      dataset_path=DATASET_PATH,
                                                      w2i_dict_path=W2I_DICT_PATH,
                                                      model_path=MODEL_PATH,
                                                      remove_unk=REMOVE_UNK)
    
    training_set, _ = data_creator.create_dataset(lstm_mode='mixed',
                                                    fixed_context=False,
                                                    context_size=[4,7],
                                                    max_class_size=1000,
                                                    split=1,
                                                    num_attractors=None,
                                                    num_helpful_nouns=None,
                                                    attractor_idx=None,
                                                    helpful_noun_idx=None,
                                                    num_words_before_subject=NUM_WORDS_BEFORE_SUBJECT,
                                                    num_words_after_verb=NUM_WORDS_AFTER_VERB,
                                                    dont_overlap_with_datasets=[test_set_correct, test_set_wrong],
                                                    dataset_path=DATASET_PATH,
                                                    w2i_dict_path=W2I_DICT_PATH,
                                                    model_path=MODEL_PATH,
                                                    remove_unk=True)
    
    training_set_2, _ = data_creator.create_dataset(lstm_mode='mixed',
                                                    fixed_context=True,
                                                    context_size=CONTEXT_SIZE_LIST[0],
                                                    max_class_size=1500,
                                                    split=1,
                                                    num_attractors=None,
                                                    num_helpful_nouns=None,
                                                    attractor_idx=None,
                                                    helpful_noun_idx=None,
                                                    num_words_before_subject=NUM_WORDS_BEFORE_SUBJECT,
                                                    num_words_after_verb=NUM_WORDS_AFTER_VERB,
                                                    dont_overlap_with_datasets=[test_set_correct, test_set_wrong, training_set],
                                                    dataset_path=DATASET_PATH,
                                                    w2i_dict_path=W2I_DICT_PATH,
                                                    model_path=MODEL_PATH,
                                                    remove_unk=True)
    
    print("Data sets loaded")
    
    if LABELS_TYPE == 'normal' or LABELS_TYPE == 'training_normal' or LABELS_TYPE == 'poster':
        training_filepath = create_activation_data.extract_data(training_set, 'train', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH)
        training_filepath_2 = create_activation_data.extract_data(training_set_2, 'train', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH)
        if not MODIFICATION:
            testing_filepaths = []
            testing_filepaths.append(create_activation_data.extract_data(test_set_correct, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH))
            testing_filepaths.append(create_activation_data.extract_data(test_set_wrong, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH))
        else:
            classifiers_path = diagnostic_experiment.train_and_store_classifier(training_filepath, "trained_classifier/{}".format(training_filepath), num_timesteps, ACTIVATIONS)
            modified_testing_filepaths = []
            modified_testing_filepaths.append(create_activation_data.extract_and_modify_data(test_set_correct, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, num_timesteps, W2I_DICT_PATH, MODEL_PATH, classifiers_path))
            modified_testing_filepaths.append(create_activation_data.extract_and_modify_data(test_set_wrong, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, num_timesteps, W2I_DICT_PATH, MODEL_PATH, classifiers_path))
            
            testing_filepaths = []
            testing_filepaths.append(create_activation_data.extract_data(test_set_correct, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH))
            testing_filepaths.append(create_activation_data.extract_data(test_set_wrong, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH))
    
    elif LABELS_TYPE == 'timestep' or LABELS_TYPE == 'subj_attractor_region' or LABELS_TYPE == 'POS_tags':
        training_filepath = create_activation_data.extract_multi_class_data(training_set, 'train', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH, LABELS_TYPE, ATTRACTOR_IDX)
        testing_filepaths = []
        testing_filepaths.append(create_activation_data.extract_multi_class_data(test_set_correct, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH, LABELS_TYPE, ATTRACTOR_IDX))
        testing_filepaths.append(create_activation_data.extract_multi_class_data(test_set_wrong, 'test', VOCAB_SIZE, HIDDEN_SIZE, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, W2I_DICT_PATH, MODEL_PATH, LABELS_TYPE, ATTRACTOR_IDX))
    
    print("Activation data extracted")
    plot_savefile = "plots/context_size{}-num_attractors{}-attractor_idx{}-num_before_subj{}-num_after_verb{}-mode{}-labels_type-{}-modification-{}".format(CONTEXT_SIZE_LIST[0], NUM_ATTRACTORS, ATTRACTOR_IDX, NUM_WORDS_BEFORE_SUBJECT, NUM_WORDS_AFTER_VERB, MODE, LABELS_TYPE, MODIFICATION)
    RESULTS_PATHS = ['results/' + test_set_correct, 'results/' + test_set_wrong]
    MODIFIED_RESULTS_PATHS = ['results/modified/' + test_set_correct, 'results/modified/' + test_set_wrong]
    
    if PLOTTING_MODE == 'heatmap':
        if LABELS_TYPE == 'normal':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment_timestep_wise(training_filepath,
                                                                                                                     testing_filepaths,
                                                                                                                     RESULTS_PATHS,
                                                                                                                     None,
                                                                                                                     MODE,
                                                                                                                     CONTEXT_SIZE_LIST,
                                                                                                                     NUM_WORDS_BEFORE_SUBJECT,
                                                                                                                     NUM_WORDS_AFTER_VERB,
                                                                                                                     ACTIVATIONS,
                                                                                                                     PRINT_PLOTTING_INFO)
    
            plotting.plot_heatmap(correct_plotting_dict, plot_savefile, 'correct')
            plotting.plot_heatmap(wrong_plotting_dict, plot_savefile, 'wrong')
    
        else:
            raise NotImplementedError('heatmap does not work for multi-class tasks yet')
    
    elif PLOTTING_MODE == 'graph':
        if LABELS_TYPE == 'normal':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment(training_filepath_2,
                                                                                                       testing_filepaths,
                                                                                                       RESULTS_PATHS,
                                                                                                       None,
                                                                                                       MODE,
                                                                                                       CONTEXT_SIZE_LIST,
                                                                                                       NUM_WORDS_BEFORE_SUBJECT,
                                                                                                       NUM_WORDS_AFTER_VERB,
                                                                                                       ACTIVATIONS,
                                                                                                       PRINT_PLOTTING_INFO)
            
            modified_correct_plotting_dict, modified_wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment(training_filepath_2,
                                                                                                       modified_testing_filepaths,
                                                                                                       MODIFIED_RESULTS_PATHS,
                                                                                                       None,
                                                                                                       MODE,
                                                                                                       CONTEXT_SIZE_LIST,
                                                                                                       NUM_WORDS_BEFORE_SUBJECT,
                                                                                                       NUM_WORDS_AFTER_VERB,
                                                                                                       ACTIVATIONS,
                                                                                                       PRINT_PLOTTING_INFO)
            
            
            if MODIFICATION:
                plotting.plot_modified_data(correct_plotting_dict, wrong_plotting_dict, modified_correct_plotting_dict, modified_wrong_plotting_dict, plot_savefile, MODE)
            else:
                plotting.plot_data(correct_plotting_dict, wrong_plotting_dict, plot_savefile, MODE)
            
        elif LABELS_TYPE == 'training_normal':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment_timestep_training(training_filepath_2,
                                                                                                       testing_filepaths,
                                                                                                       RESULTS_PATHS,
                                                                                                       None,
                                                                                                       MODE,
                                                                                                       CONTEXT_SIZE_LIST,
                                                                                                       NUM_WORDS_BEFORE_SUBJECT,
                                                                                                       NUM_WORDS_AFTER_VERB,
                                                                                                       ACTIVATIONS,
                                                                                                       PRINT_PLOTTING_INFO)
    
            modified_correct_plotting_dict, modified_wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment_timestep_training(training_filepath_2,
                                                                                                       modified_testing_filepaths,
                                                                                                       MODIFIED_RESULTS_PATHS,
                                                                                                       None,
                                                                                                       MODE,
                                                                                                       CONTEXT_SIZE_LIST,
                                                                                                       NUM_WORDS_BEFORE_SUBJECT,
                                                                                                       NUM_WORDS_AFTER_VERB,
                                                                                                       ACTIVATIONS,
                                                                                                       PRINT_PLOTTING_INFO)
            
            if MODIFICATION:
                plotting.plot_modified_data(correct_plotting_dict, wrong_plotting_dict, modified_correct_plotting_dict, modified_wrong_plotting_dict, plot_savefile, MODE)
            else:
                plotting.plot_data(correct_plotting_dict, wrong_plotting_dict, plot_savefile, MODE)
            
        elif LABELS_TYPE == 'poster':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment_prediction_training(training_filepath,
                                                                                                       testing_filepaths,
                                                                                                       RESULTS_PATHS,
                                                                                                       None,
                                                                                                       MODE,
                                                                                                       CONTEXT_SIZE_LIST,
                                                                                                       NUM_WORDS_BEFORE_SUBJECT,
                                                                                                       NUM_WORDS_AFTER_VERB,
                                                                                                       ACTIVATIONS,
                                                                                                       PRINT_PLOTTING_INFO)
    
            plotting.plot_prediction_data(correct_plotting_dict, wrong_plotting_dict, plot_savefile, MODE)
    
        elif LABELS_TYPE == 'timestep' or LABELS_TYPE == 'subj_attractor_region' or LABELS_TYPE == 'POS_tags':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_multi_class_diagnostic_experiment(training_filepath,
                                                                                                                   testing_filepaths,
                                                                                                                   RESULTS_PATHS,
                                                                                                                   None,
                                                                                                                   MODE,
                                                                                                                   CONTEXT_SIZE_LIST,
                                                                                                                   NUM_WORDS_BEFORE_SUBJECT,
                                                                                                                   NUM_WORDS_AFTER_VERB,
                                                                                                                   ACTIVATIONS,
                                                                                                                   PRINT_PLOTTING_INFO,
                                                                                                                   ATTRACTOR_IDX)
    
            plotting.plot_data(correct_plotting_dict, wrong_plotting_dict, plot_savefile, MODE)
    
    elif PLOTTING_MODE == 'component_heatmap':
        if LABELS_TYPE == 'normal':
            correct_plotting_dict, wrong_plotting_dict = diagnostic_experiment.run_diagnostic_experiment_component_wise(training_filepath,
                                                                                                                      testing_filepaths,
                                                                                                                      RESULTS_PATHS,
                                                                                                                      None,
                                                                                                                      MODE,
                                                                                                                      CONTEXT_SIZE_LIST,
                                                                                                                      NUM_WORDS_BEFORE_SUBJECT,
                                                                                                                      NUM_WORDS_AFTER_VERB,
                                                                                                                      ACTIVATIONS,
                                                                                                                      PRINT_PLOTTING_INFO)
    
            plotting.plot_heatmap_component_wise(correct_plotting_dict, plot_savefile, 'correct', num_timesteps)
            plotting.plot_heatmap_component_wise(wrong_plotting_dict, plot_savefile, 'wrong', num_timesteps)
    
    elif PLOTTING_MODE == 'sentence':
        diagnostic_experiment.diagnostic_experiment_sentence_level(training_filepath,
                                                                   testing_filepaths,
                                                                   RESULTS_PATHS,
                                                                   None,
                                                                   MODE,
                                                                   CONTEXT_SIZE_LIST,
                                                                   NUM_WORDS_BEFORE_SUBJECT,
                                                                   NUM_WORDS_AFTER_VERB,
                                                                   ACTIVATIONS,
                                                                   [test_set_correct, test_set_wrong])
    
    else:
        raise NotImplementedError('heatmap does not work for multi-class tasks yet')
