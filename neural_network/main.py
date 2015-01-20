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
from cptm_neural_network import CPTMNeuralNetwork, get_error_term_dict
from get_lists_and_dictionaries import get_everything
from xbleu import xbleu, get_Ej_translation_probability_list
from gensim import corpora


def main():
    W1 = get_W1_from_text_file("../lda/data/weight_initialization.txt")
    W2 = np.identity(100)
    nn = CPTMNeuralNetwork([W1.shape[0], 100, 100], [W1, W2])
    dictionary = corpora.Dictionary.load("../lda/data/dictionary.dict")

    for i in xrange(1):
        (phrase_pair_dict_all, phrase_pair_dict_n_list,
            total_base_score_list, sbleu_score_list) = get_everything(0)
        xblue_i = xbleu(nn, total_base_score_list, sbleu_score_list, phrase_pair_dict_n_list, dictionary)
        Ej_translation_probability_list = get_Ej_translation_probability_list(
            nn, total_base_score_list, phrase_pair_dict_n_list, dictionary)

        error_term_dict_i = get_error_term_dict(
            phrase_pair_dict_all, phrase_pair_dict_n_list,
            sbleu_score_list, xblue_i,
            Ej_translation_probability_list)

        print "Weights before update"
        print nn.weights
        nn.update_mini_batch(phrase_pair_dict_all, 1000000, dictionary, error_term_dict_i)
        print "Weights after update"
        print nn.weights


class LazyFileReader(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.strip().lower()


def get_W1_from_text_file(path):
    with open(path, 'r') as W1_text_file:
        line = map(lambda x: float(x), (W1_text_file.readline().strip() + " 0.0").split(' '))
        W1 = np.zeros((100, len(line)))  # TODO: FIX W1_TEXT ONE LESS ENTRY
        W1[0] = line

        for i, line in enumerate(W1_text_file):
            line = map(lambda x: float(x), (line.strip() + " 0.0").split(' '))
            W1[i+1] = line
    return W1.transpose()


if __name__ == "__main__":
    main()
