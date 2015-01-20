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

def main(source_file_name, n_best_list_file_name, sbleu_score_list_file_name):
    W1 = np.loadtxt("data/weight_initialization.txt")
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

    training_order_list = range(training_set_size)
    random.shuffle(training_order_list)
    # TODO: while not converged: how to check convergence? Early stop method

    xBleu_history = []
    xBleu_change_history = []
    d_theta_old = [0, 0]  # momentum terms
    for i in xrange(1):
        (phrase_pair_dict_all, phrase_pair_dict_n_list,
            total_base_score_list, sbleu_score_list) = get_everything(i, source_file_name, n_best_list_file_name, sbleu_score_list_file_name)
        xblue_i = xbleu(nn, total_base_score_list, sbleu_score_list,
                        phrase_pair_dict_n_list, dictionary)
        Ej_translation_probability_list = get_Ej_translation_probability_list(
            nn, total_base_score_list, phrase_pair_dict_n_list, dictionary)

        error_term_dict_i = get_error_term_dict(
            phrase_pair_dict_all, phrase_pair_dict_n_list,
            sbleu_score_list, xblue_i,
            Ej_translation_probability_list)

        if False:
            print "Weights before update"
            print
            print "W1"
            print nn.weights[0]
            print
            print "W2"
            print nn.weights[1]
            print
            sum_W1_before = sum(sum(nn.weights[0]))
            print "W1 sum BEFORE:", sum_W1_before
            sum_W2_before = sum(sum(nn.weights[1]))
            print "W2 sum BEFORE:", sum_W2_before

        d_theta_old = nn.update_mini_batch(
            phrase_pair_dict_all, 0.001, dictionary, error_term_dict_i, d_theta_old)

        if False:
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

            print "W1 sum difference:", sum_W1_after - sum_W1_before
            print "W2 sum difference:", sum_W2_after - sum_W2_before

        # xBleu increases?
        if False:
            xblue_i_after = xbleu(nn, total_base_score_list, sbleu_score_list,
                                  phrase_pair_dict_n_list, dictionary)

            xBleu_history.append((xblue_i, xblue_i_after))
            xBleu_change_history.append(xblue_i_after - xblue_i)
            print "-------------------------------------------------------------"
            print "xBleu history: [(xBleu_before_gradient_descent, xBleu_after_gradient_descent)]"
            print xBleu_history
            print
            print "xBleu_change_history [xBleu_after - xBleu_before]"
            print xBleu_change_history
            print "-------------------------------------------------------------"

    np.savetxt('W1.txt', nn.weights[0])
    np.savetxt('W2.txt', nn.weights[1])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "data/sbleu.txt")
