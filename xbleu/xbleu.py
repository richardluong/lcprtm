from __future__ import division
import numpy as np

def get

def xbleu(n_best_list_base_system_total_score_list, s_bleu_list, phrase_pair_list):
    """
    @n_best_list : list of strings, corresponing n best list of the given source_sentence
    """

    Ej_translation_probability_list = get_Ej_translation_probability_list(n_best_list_base_system_total_score_list, phrase_pair_list)
    
    xbleu = 0
    for j, s_bleu_Ej in enumerate(s_bleu_list):
        xbleu += s_bleu_Ej * Ej_translation_probability_list[j]

    return xbleu

def Ej_translation_probability_numerator(n_best_list_base_system_total_score_list, phrase_pair_list):
    return np.exp(sum(n_best_list_base_system_total_score_list) + new_feature_value(phrase_pair_list))

def get_Ej_translation_probability_list(n_best_list_base_system_total_score_list, phrase_pair_list):

    Ej_translation_probability_numerator_list = []
    for total_score_j in n_best_list_base_system_total_score_list:
        Ej_translation_probability_numerator = Ej_translation_probability_numerator(n_best_list_base_system_total_score_list, phrase_pair_list)
        Ej_translation_probability_numerator_list.append(Ej_translation_probability_numerator)

    Ej_translation_probability_denominator = sum(Ej_translation_probability_numerator_list)

    Ej_translation_probability_list = map(lambda x: x/Ej_translation_probability_denominator, Ej_translation_probability_numerator_list)
    return Ej_translation_probability_list

    
