"""
Pseudo code

Initialization:
    Initialize W1 and get token2id dictionary through gensim lda
    Initialize neural network with W1, W2 = ID matrix

Goal: to learn W1 and W2
    1) Choose one source sentence and its corresponding N-best list (preferably randomly)
    Estimate gradients for W1 and W2 with this training sample
        ... procedure

"""

from __future__ import division

import numpy as np
import random, sys
from modules.cptm_neural_network import CPTMNeuralNetwork, get_error_term_dict
from modules.get_lists_and_dictionaries import get_everything
from modules.xbleu import xbleu, get_Ej_translation_probability_list
from gensim import corpora

debug_mode = True
debug_mode_verbose = False

learning_rate = 0.001  # for the neural network
smoothing_factor = 1  # see Equation 6 in the paper


def main(source_file_name, n_best_list_file_name, sbleu_score_list_file_name,
         learning_rate, smoothing_factor):
    print "Loading and initializing neural network"
    W1 = np.loadtxt("data/weight_initialization.gz").transpose()
    W2 = np.identity(100)
    nn = CPTMNeuralNetwork([W1.shape[0], 100, 100], [W1, W2])
    dictionary = corpora.Dictionary.load("data/dictionary.dict")

    training_set_size = 0
    # Each line in source_file is a source sentence.
    # source_file should end with and empty line
    with open(source_file_name, 'r') as source_file:
        for _ in source_file:
            training_set_size += 1
    training_set_size -= 1  # ends with empty line

    # randomize training samples
    # TODO: REMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOVE
    training_set_size = 3
    training_order_list = range(training_set_size)
    # TODO: SHUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUFLE
    #random.shuffle(training_order_list)

    # separate into test and training set
    test_size = max(1, int(0.01*training_set_size))
    test_order_list = training_order_list[:test_size]
    training_order_list = training_order_list[test_size:]

    # initialize variables
    d_theta_old = [0, 0]  # momentum terms

    old_loss_value_test_set = get_average_loss_value_of_test_sample(
        test_order_list, nn, dictionary, source_file_name,
        n_best_list_file_name, sbleu_score_list_file_name,
        smoothing_factor)
    converged = False
    epoch_count = 0

    # For debug
    xBleu_history = []
    xBleu_change_history = []

    # train until overfit (early stop)
    print "Start training"
    while not converged:
        theta_previous = nn.weights
        for i in training_order_list:
            (phrase_pair_dict_all, phrase_pair_dict_n_list,
                total_base_score_list, sbleu_score_list) \
                = get_everything(i, source_file_name,
                                 n_best_list_file_name, sbleu_score_list_file_name)
            xblue_i = xbleu(nn, total_base_score_list, sbleu_score_list,
                            phrase_pair_dict_n_list, dictionary, smoothing_factor)
            Ej_translation_probability_list = get_Ej_translation_probability_list(
                nn, total_base_score_list, phrase_pair_dict_n_list, dictionary, smoothing_factor)

            error_term_dict_i = get_error_term_dict(
                phrase_pair_dict_all, phrase_pair_dict_n_list,
                sbleu_score_list, xblue_i,
                Ej_translation_probability_list)

            if debug_mode_verbose:
                print "Weights before update"
                print
                print "W1"
                print nn.weights[0]
                print
                print "W2"
                print nn.weights[1]
                print
                sum_W1_before = sum(sum(nn.weights[0]))
                print "W1 sum of all elements BEFORE gradient descent:", sum_W1_before
                sum_W2_before = sum(sum(nn.weights[1]))
                print "W2 sum of all elements BEFORE gradient descent:", sum_W2_before

            d_theta_old = nn.update_mini_batch(
                phrase_pair_dict_all, learning_rate, dictionary, error_term_dict_i, d_theta_old)

            if debug_mode_verbose:
                print "Weights after update"
                print
                print "W1"
                print nn.weights[0]
                print
                print "W2"
                print nn.weights[1]
                print
                sum_W1_after = sum(sum(nn.weights[0]))
                print "W1 sum AFTER:", sum_W1_after
                sum_W2_after = sum(sum(nn.weights[1]))
                print "W2 sum AFTER:", sum_W2_after

                print "W1 sum of all elements AFTER gradient descent:", sum_W1_after - sum_W1_before
                print "W2 sum of all elements AFTER gradient descent:", sum_W2_after - sum_W2_before

            # check if xBleu increases
            if debug_mode_verbose:
                xblue_i_after = xbleu(nn, total_base_score_list, sbleu_score_list,
                                      phrase_pair_dict_n_list, dictionary, smoothing_factor)

                xBleu_history.append((xblue_i, xblue_i_after))
                xBleu_change_history.append(xblue_i_after - xblue_i)
                print "-------------------------------------------------------------"
                print "xBleu history: [(xBleu_before_gradient_descent, xBleu_after_gradient_descent)]"
                print xBleu_history
                print
                print "xBleu_change_history [xBleu_after - xBleu_before]"
                print xBleu_change_history
                print "-------------------------------------------------------------"

            print "Finished epoch nr", epoch_count, "training sample nr", i

        epoch_count += 1
        print "Finished epoch number:", epoch_count

        # calculate loss function on test set after each epoch using updated weights
        loss_value_test_set = get_average_loss_value_of_test_sample(
            test_order_list, nn, dictionary,
            source_file_name, n_best_list_file_name, sbleu_score_list_file_name,
            smoothing_factor)
        print "Old average loss function value (-xBleu) over test set from previous epoch:"
        print old_loss_value_test_set
        print "Average loss function value with updated weights:"
        print loss_value_test_set
        print "Difference (new - old):"
        print loss_value_test_set - old_loss_value_test_set
        if loss_value_test_set > old_loss_value_test_set:
            converged = True
            print "CONVERGED!!!!!!!!!!!!"
            print "Saving weights from previous epoch to file"
            np.savetxt('W1.gz', theta_previous[0])
            np.savetxt('W2.gz', theta_previous[1])
        else:
            print "No overfitting, keep training..."


def get_average_loss_value_of_test_sample(test_order_list, nn, dictionary,
                                          source_file_name, n_best_list_file_name,
                                          sbleu_score_list_file_name, smoothing_factor):
    loss_value_test_set = 0
    for t_i in test_order_list:
        (phrase_pair_dict_n_listase_pair_dict_all, phrase_pair_dict_n_list,
            total_base_score_list, sbleu_score_list) \
            = get_everything(t_i, source_file_name, n_best_list_file_name,
                             sbleu_score_list_file_name)
        xBlue_t_i = xbleu(nn, total_base_score_list, sbleu_score_list,
                          phrase_pair_dict_n_list, dictionary, smoothing_factor)
        loss_value_test_set -= xBlue_t_i
    return loss_value_test_set/len(test_order_list)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        learning_rate = sys.argv[3]
    if len(sys.argv) >= 5:
        smoothing_factor = sys.argv[4]
    main(sys.argv[1], sys.argv[2], "data/sbleu.txt", learning_rate, smoothing_factor)
