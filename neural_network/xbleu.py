from __future__ import division
import numpy as np
from cptm_neural_network import new_feature_value


def xbleu(nn, n_best_list_base_system_total_score_list, s_bleu_list, phrase_pair_list, dictionary):
    """
    @n_best_list : list of strings, corresponing n best list of the given source_sentence
    @phrase_pair_list : list of dicts
    @dictionary : gensim.corpora.Dictionary
    """

    Ej_translation_probability_list = get_Ej_translation_probability_list(nn, n_best_list_base_system_total_score_list, phrase_pair_list, dictionary)
    
    xbleu = 0
    for j, s_bleu_Ej in enumerate(s_bleu_list):
        xbleu += s_bleu_Ej * Ej_translation_probability_list[j]

    return xbleu

def get_Ej_translation_probability_numerator(nn, n_best_list_base_system_total_score_list, phrase_pair_list, dictionary):
    return np.exp(sum(n_best_list_base_system_total_score_list) + new_feature_value(nn, phrase_pair_list, dictionary))

def get_Ej_translation_probability_list(nn, n_best_list_base_system_total_score_list, phrase_pair_list, dictionary):

    n_best_list_base_system_total_score_list = map(lambda h: np.exp(h), n_best_list_base_system_total_score_list)

    Ej_translation_probability_numerator_list = []
    for j, total_score_j in enumerate(n_best_list_base_system_total_score_list):
        Ej_translation_probability_numerator = get_Ej_translation_probability_numerator(nn, n_best_list_base_system_total_score_list, phrase_pair_list[j], dictionary)
        Ej_translation_probability_numerator_list.append(Ej_translation_probability_numerator)

    Ej_translation_probability_denominator = sum(Ej_translation_probability_numerator_list)

    # print "numerators", Ej_translation_probability_numerator_list
    # print "denominator", Ej_translation_probability_denominator
    Ej_translation_probability_list = map(lambda x: x/Ej_translation_probability_denominator, Ej_translation_probability_numerator_list)
    return Ej_translation_probability_list

    
